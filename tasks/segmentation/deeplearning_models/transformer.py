from typing import Tuple, Dict
from collections import defaultdict
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchmetrics.functional import dice
import numpy as np
import wandb
import math 

from externel_library.mir_eval_simple.mir_eval_segment import pairwise,nce
from externel_library.mir_eval_simple.mir_eval_simple_utils import boundaries_to_intervals

import torch.optim as optim

from tasks.segmentation.deeplearning_models.base import BaseModel


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        """Positional Encoding. shape of input and output are the same
        TODO: may need a new way for position encoding
        Args:
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x

class PositionwiseFeedforwardLayer(nn.Module):
    """
    Just a linear mapping to 'learn more infomation', model_dim -> feedforward_dim -> model_dim
    Params: 
        - model_dim: input dimension of the model
        - feedforward_dim: dimension of inner layer, only being used in this function
        - dropout_rate: set as 0 as defult
    
        Return: shape of input and output are the same: [batch_size, Seqlen, ]

    """
    def __init__(self, model_dim, feedforward_dim, dropout_rate):
        super().__init__()
        self.fc_1 = nn.Linear(model_dim, feedforward_dim)
        self.fc_2 = nn.Linear(feedforward_dim, model_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x))) #[batch size, Seqlen, pf_dim]
        x = self.fc_2(x)
        return x #[batch size, Seqlen, model_dim]



def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)

    batch_size, num_heads, seq_length, _ = attn_logits.shape

    if mask is not None:
        if len(mask.shape) == 2:  # adjust the shape of source mask
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_length]
            mask = mask.expand(batch_size, num_heads, seq_length, -1)

        elif len(mask.shape) == 3:  # adjuest the shape of target mask
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_length + 2, seq_length + 2]
            mask = mask.expand(batch_size, num_heads, -1, -1)

        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, model_dim, embed_dim, num_heads):
        """
        Params:
        - model_dim: Hidden dimensionality of the input
        - embed_dim: we map the model_dim into 3 * embed_dim, but actually outside this function we set embed_dim = model_dim
        - num_head: number of headers
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # if embed_dim = 3, num_heads = 3, then there's 1 dim for each head

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(model_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        # Linear mapping for query, key, value
        self.fc_q = nn.Linear(model_dim, model_dim)
        self.fc_k = nn.Linear(model_dim, model_dim)
        self.fc_v = nn.Linear(model_dim, model_dim)


        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
    
    def forward(self, query, key, value, mask=None, return_attention=False):
        
        batch_size, seq_length, embed_dim = query.size()

        # shape: [batch_size, Seqlen, model_dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # shape: [batch_size, head, Seqlen, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        values, attention = scaled_dot_product(Q, K, V, mask)

        # Concatenate the heads
        values = values.permute(0, 2, 1, 3).contiguous()  # [Batch, SeqLen, Head, head_dim]
        values = values.view(batch_size, -1, self.embed_dim)  # combine the Head and head_dim by reshape:[Batch, Seqlen, ]

        # Apply final linear projection
        output = self.o_proj(values)

        if return_attention:
            return output, attention
        else:
            return output



class EncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads, feedforward_dim, dropout=0.0):
        """EncoderBlock.

        Args:
            model_dim: Dimensionality of the input (after Jie's first linear mapping and positional embedding)
            num_heads: Number of heads to use in the attention block
            feedforward_dim: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        # TODO: change the code below since definition of MultiheadAttention is changed
        self.self_attn = MultiheadAttention(model_dim, model_dim, num_heads)

        self.linear_net = PositionwiseFeedforwardLayer(model_dim, feedforward_dim, dropout)

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x,x,x, mask=mask) #self-attention
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


class DecoderBlock(nn.Module):
    def __init__(self, 
                 model_dim,
                 num_heads,
                 feedforward_dim, 
                 dropout,
                 ):
        super().__init__()
        
        self.self_attention_layer_norm = nn.LayerNorm(model_dim)
        self.cross_attention_layer_norm = nn.LayerNorm(model_dim)
        self.ff_layer_norm = nn.LayerNorm(model_dim)

        self.self_attention = MultiheadAttention(model_dim, model_dim, num_heads) 
        self.cross_attention = MultiheadAttention(model_dim, model_dim, num_heads)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(model_dim, feedforward_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, encoded_source, target, source_mask, target_mask):
        _target = self.self_attention(target, target, target, target_mask)
        target = self.self_attention_layer_norm(target + self.dropout(_target))

        _target, attention = self.cross_attention(target, encoded_source, encoded_source, source_mask,return_attention = True)
        target = self.cross_attention_layer_norm(target + self.dropout(_target))

        _target = self.positionwise_feedforward(target)
        target = self.ff_layer_norm(target + self.dropout(_target))

        return target, attention


class TransformerDecoder(nn.Module):
    def __init__(self,
                 model_dim,
                 num_layers,
                 num_heads,
                 feedforward_dim,
                 dropout,
                 device,
                 decoder_max_length
                 ):
        super().__init__()

        self.model_device = device
        self.position_embedding = PositionalEncoding(model_dim=model_dim, max_len=decoder_max_length)
        self.decoder_layers = nn.ModuleList(DecoderBlock(model_dim, num_heads, feedforward_dim, dropout) for _ in range(num_layers))

        self.fc_out = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([model_dim])).to(device)


    def forward(self, encoded_source, target, source_mask, target_mask):
        """
        Params:
            - target: encoded chord sequence: [batch_size, Seqlen, emb_dim]
            - encoded_source: output come out of TransformerEncoder module [batch_size, Seqlen, emb_dim]  
            - source_mask: as we used padding to make sure Seqlen are the same, source_mask is used to make sure model won't calculate the padding part
            - target_mask: 
            
            
        Return:
            - output: the output of the Decoder Module
            - attention: the attention matrix (batch_size, head, Seqlen, Seqlen) of encoder attention, may not useful
        """
        assert len(target.shape) == 3 #Check if the chord sequence satisfied [batch_size, Seqlen, emb_dim]
        # batch_size, Seqlen = target.shape[0], target.shape[1]
        
        # Pass through each of the decoder layers in sequence
        for layer in self.decoder_layers:
            target, attention = layer(encoded_source, target, source_mask, target_mask)

        # Apply the final fully connected layer to get logits of output_dim size
        output = self.fc_out(target)

        return output, attention
        
    

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class TransformerModel(BaseModel):
    
    def __init__(self,segmentation_train_args):
        """TransformerPredictor.

        Args:
            input_dim: Hidden dimensionality of the input
            model_dim: Hidden dimensionality to use inside the Transformer
            num_classes: Number of classes to predict per sequence element
            num_heads: Number of heads to use in the Multi-Head Attention blocks
            num_layers: Number of encoder blocks to use.
            lr: Learning rate in the optimizer
            warmup: Number of warmup steps. Usually between 50 and 500
            max_iters: Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout: Dropout to apply inside the model
            input_dropout: Dropout to apply on the input features
        """
        super().__init__()
        self.save_hyperparameters(segmentation_train_args)
        
        # Extract the hyperparameter from segmentation_train_args
        # TODO: remove the arg that don't needed
        self.source_input_dim = segmentation_train_args.get("source_input_dim")
        self.model_dim = segmentation_train_args.get("model_dim")
        self.feedforward_dim = segmentation_train_args.get("feedforward_dim")
        self.num_classes = segmentation_train_args.get("num_classes")
        self.num_heads = segmentation_train_args.get("num_heads")
        self.num_layers = segmentation_train_args.get("num_layers")
        self.decoder_max_length = segmentation_train_args.get("decoder_max_length")
        
        self.model_device = segmentation_train_args.get("device")
        self.lr = segmentation_train_args.get("lr")
        self.warmup = segmentation_train_args.get("warmup")
        self.max_iters = segmentation_train_args.get("max_iters")
        self.dropout = segmentation_train_args.get("dropout", 0.0)
        self.input_dropout = segmentation_train_args.get("input_dropout", 0.0)

        self.pad_idx = segmentation_train_args.get("pad_idx", 0) #use 0 as padding


        self._create_model() # call this function to create a Transformer model object


    def _create_model(self):
        # Only being used in def input_net()
        self.input_net = nn.Sequential(
            nn.Dropout(self.input_dropout), nn.Linear(self.source_input_dim, self.model_dim)
        )

        # Add MLP between 
        self.source_mapping = nn.Linear(self.source_input_dim, self.model_dim)
        self.target_mapping = nn.Linear(self.num_classes, self.model_dim)

        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(model_dim=self.model_dim)


        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            num_layers=self.num_layers,
            model_dim=self.model_dim,
            feedforward_dim=self.feedforward_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )

        # Transformer Decoder
        self.transformer_decoder = TransformerDecoder(
            model_dim = self.model_dim,
            num_layers = self.num_layers,
            num_heads = self.num_heads,
            feedforward_dim = self.feedforward_dim,
            dropout = self.dropout,
            device = self.model_device,
            decoder_max_length = self.decoder_max_length
        )


        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.num_classes), 
        )
    

    def _predict(self, batch: Tuple[torch.tensor, torch.tensor, torch.tensor], add_positional_encoding=True) -> Tuple[torch.tensor, torch.tensor]:
        
        source, target, source_mask, target_mask = batch
        """
        Args: 
        - source: A batch of chord sequence: [batch_size, Seqlen, 3] (why 3: each chord is encode into a length=3 vector)
        - target: A batch of target sequence:[batch_size, Seqlen, 14] (why 14: 14 different class)
        - source_mask: 
        - target_mask:
        """
        
        original_target = target.clone()
        class_indices = torch.argmax(original_target, dim=-1)

        # Map source and target to model dim
        source = self.source_mapping(source.float())
        target = self.target_mapping(target.float())

        # TODO: figure out a better way to add position information (note that: 不同的feature应该用不同的处理方法，例如chord和spectogram不应该用相同的方法)
        if add_positional_encoding:
            source = self.positional_encoding(source)
            target = self.positional_encoding(target)
        
        encoded_source = self.transformer_encoder(source, source_mask)
        
        output, _ = self.transformer_decoder(encoded_source, target , source_mask, target_mask)

        output = self.output_net(output)
        
        loss = F.cross_entropy(output.view(-1, 14), class_indices.view(-1), ignore_index=self.pad_idx)

        return output, loss



    def _test(self, batch: Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]) -> Tuple[torch.tensor, Dict[str, float]]:
        """
        Perform the prediction step in the specified batch. When computing the loss
        masked elements are ignored.

        Args:
            batch (Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]): 
            Input batch in the form (data, labels, padding mask)

        Returns:
            Tuple[torch.tensor, Dict[str, float]]: The loss item and the dictionary of metrics
        """
        metrics = defaultdict(list)
        chords, labels, source_mask, target_mask = batch
        
        with torch.no_grad():
            pred, loss = self._predict((chords, labels, source_mask, target_mask))
        
            # 因为我们关心的是labels，所以需要转换为类别索引
            labels_idx = labels.argmax(dim=-1)  # [batch_size, Seqlen+2]

            for pi, yi, mi in zip(pred, labels_idx, target_mask):
                valid_mask = mi.any(dim=-1)  # 沿第二维度压缩，获取有效的掩码
                pi = pi[valid_mask].argmax(axis=-1).cpu().numpy()  # 对预测结果应用mask
                yi = yi[valid_mask].cpu().numpy()  # 对实际标签应用mask

                # Calculate the time interval, in our task there're len(yi) different interval
                # e.g. when len(yi) = 2: intervals = [[0,1], [1,2]]
                # Obviously in our task, pi and yi share the same interval
                intervals = boundaries_to_intervals(np.arange(len(yi) + 1)) 

                # Calculate Precision, Recall Rate, F1-Score by using Agreement matrix
                precision, recall, f1 = pairwise(intervals, yi, intervals, pi)
                metrics["p_precision"].append(precision)
                metrics["p_recall"].append(recall)
                metrics["p_f1"].append(f1)

                # Caslculate S_u (under-segmentation score), S_o (over-segmentation score), and F1 for them 
                over, under, under_over_f1 = nce(intervals, yi, intervals, pi)
                metrics["under"] = under
                metrics["over"] = over
                metrics["under_over_f1"] = under_over_f1
        
        metrics = {k: np.mean(v) for k, v in metrics.items()}
        return loss, metrics

        

    @torch.no_grad()
    def input_net(self, x, mask=None, add_positional_encoding=True):
        """Function for extracting the attention matrices of the whole Transformer for a single batch.

        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup, max_iters=self.max_iters
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration


    def generate_sequence(self, source_seq):
        batch_size, seq_length, _ = source_seq.shape
        source_seq = self.source_mapping(source_seq.float())
        source_seq = self.positional_encoding(source_seq)
        encoded_source = self.transformer_encoder(source_seq)

        # 初始化target_seq为一个全0的tensor，形状与source_seq相同
        target_seq = torch.zeros((batch_size, seq_length), dtype=torch.long).to(self.model_device)

        for i in range(seq_length):
            target_mapped = self.target_mapping(F.one_hot(target_seq, num_classes=self.num_classes).float())
            target_mapped = self.positional_encoding(target_mapped)

            output, _ = self.transformer_decoder(encoded_source, target_mapped, source_mask=None, target_mask=None)
            output = self.output_net(output)

            # 获取每个时间步的预测结果
            next_item = output[:, i, :].argmax(dim=-1)
            target_seq[:, i] = next_item

        return target_seq




"""
Test Code:
------------------------------------------------------------------------------------------------------------
import unittest
import torch
import math
from torch import nn
import tasks.segmentation.deeplearning_models.transformer as Transformer

model_dim = 64
feedforward_dim=128
seq_length = 50
batch_size = 5
num_layers = 6
num_heads = 2
input_dim = 3

positional_encoding = Transformer.PositionalEncoding(model_dim)
dummy_source = torch.zeros(batch_size, seq_length, model_dim)
dummy_target = torch.zeros(batch_size, seq_length+2, model_dim)
dummy_source_mask = torch.zeros(batch_size, seq_length)
dummy_target_mask = torch.zeros(batch_size, seq_length+2, seq_length+2)

------------------------------------------------------------------------------------------------------------
[Test PositionalEncoding]

positional_encoding = Transformer.PositionalEncoding(model_dim)
dummy_input = torch.zeros(batch_size, seq_length, model_dim)

output = positional_encoding(dummy_input)
assert output.shape == dummy_input.shape
------------------------------------------------------------------------------------------------------------
[Test Signal Encoding Block]

pos_output = output
enc_output = enc_blk(pos_output)
assert pos_output.shape == enc_output.shape
------------------------------------------------------------------------------------------------------------
[Test Encoding Module]

encoder = Transformer.TransformerEncoder(
    num_layers=num_layers,
    model_dim=model_dim,
    feedforward_dim=2 * model_dim,
    num_heads=num_heads,
    dropout=0.0,
)

encoder_output = encoder(output)

assert output.shape == encoder_output.shape

------------------------------------------------------------------------------------------------------------
[Test Single Decoder Layer]
position_embedding = Transformer.PositionalEncoding(model_dim=model_dim, max_len=500)
pos_dummy_target = position_embedding(dummy_target)
assert dummy_target.shape == pos_dummy_target.shape

decoder = Transformer.DecoderBlock(model_dim=model_dim, num_heads=num_heads, feedforward_dim=feedforward_dim,dropout=0.0)

decoder_output, _ = decoder(encoder_output, pos_dummy_target, dummy_source_mask, dummy_target_mask)

print(encoder_output.shape)
print(decoder_output.shape)
------------------------------------------------------------------------------------------------------------
[Test Decoder Module]
decoder = Transformer.TransformerDecoder(
                 model_dim,
                 num_layers,
                 num_heads,
                 feedforward_dim,
                 dropout=0.0,
                 device='cpu',
                 decoder_max_length=500
                 )

decoder_output, _ = decoder(encoder_output, dummy_target,dummy_source_mask,dummy_target_mask)

decoder_output.shape

"""


# Legacy Code:

    # This is the old forward function (encoder-only Transformer)
    # def _predict(self, batch: Tuple[torch.tensor, torch.tensor, torch.tensor], add_positional_encoding = True) -> Tuple[torch.tensor, torch.tensor]:
    #     x, y, mask = batch
        
    #     x = self.input_net(x.float())
    #     if add_positional_encoding:
    #         x = self.positional_encoding(x)
        
    #     x = self.transformer_encoder(x, mask=mask)
    #     x = self.output_net(x) 
        
    #     y_class = torch.argmax(y, dim=-1)
    #     x_masked = x[mask != 0].view(-1, x.size(-1))
    #     y_masked = y_class[mask != 0]
    #     loss = F.cross_entropy(x_masked, y_masked)

    #     return x, loss



    # This is the old deinition of forward process of MultiheadAttention (only suitebla for encoder-only architecture)
    # def forward(self, x, mask=None, return_attention=False):
    #     """
    #     from x -> q, k ,v -> value -> output
    #     x:                  [batch_size, SeqLen, emb_dim]
    #     q/k/v:              [batch_size, head, Seqlen, emb_dim / num_heads]
    #     value not reshape:  [batch_size, head, SeqLen, head_dim]
    #     value after reshape:[batch_size, SeqLen, emb_dim]
    #     output:             [batch_size, SeqLen, emb_dim]
    #     """
    #     batch_size, seq_length, embed_dim = x.size()
    #     qkv = self.qkv_proj(x)

    #     # Separate Q, K, V from linear output
    #     qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
    #     qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, 3 * self.head_dim], swap the 2rd and 3th element
    #     q, k, v = qkv.chunk(3, dim=-1) # Chunk qkv into q, k, v： [Batch, Head, SeqLen, self.head_dim]

    #     # Determine value outputs
    #     values, attention = scaled_dot_product(q, k, v, mask=mask)
    #     values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, head_dim]
    #     values = values.reshape(batch_size, seq_length, embed_dim) # combine the Head and head_dim by reshape
    #     o = self.o_proj(values)

    #     if return_attention:
    #         return o, attention
    #     else:
    #         return o