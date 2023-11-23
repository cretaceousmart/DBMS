from typing import Tuple, Dict
from collections import defaultdict
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics.functional import dice
from tasks.segmentation.deeplearning_models.base import BaseModel




# Model V1: Long Short Term Memory (LSTM)
class LSTMBaselineModel(BaseModel):
    """
    Model for the functional segmentation of a music piece.

    Args:
        embedding_dim (int, optional): Dimension of the chord embeddings. Defaults to 10.
        hidden_size (int, optional): LSTM hidden size. Defaults to 100.
        dropout (float, optional): LSTM dropout. Defaults to 0.0.
        num_layers (int, optional): Number of stacked layers in the LSTM. Defaults to 1.
        num_labels (int, optional): Number of sections to be predicted. Defaults to 11.
        learning_rate (float, optional): Default learning rate. Defaults to 0.001.
    """
    def __init__(self, 
        segmentation_train_args: dict = None,
        embedding_dim: int = 10,
        hidden_size: int = 100, 
        dropout: float = 0.0,
        num_layers: int = 1, 
        num_labels: int = 11,
        learning_rate: float = 0.001
        ):

        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_size,
                            dropout=dropout,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.classification = nn.Linear(hidden_size * 2, num_labels)
        self.softmax = nn.Softmax(dim=2)
        self.segmentation_train_args = segmentation_train_args
        self.init_weights(self.segmentation_train_args.get("init_method"))
    
    def init_weights(self, init_method):
        if init_method == "xavier":
            self.apply(self._init_xavier)
        elif init_method == "orthogonal":
            self.apply(self._init_orthogonal)
        # 可以根据需要添加更多的初始化方法

    @staticmethod
    def _init_xavier(m):
        if type(m) == nn.LSTM:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)

    @staticmethod
    def _init_orthogonal(m):
        if type(m) == nn.LSTM:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.orthogonal_(param.data)


    def _predict(self, batch: Tuple[torch.tensor, torch.tensor, torch.tensor]) -> Tuple[torch.tensor, torch.tensor]:
        """
        Perform the prediction step in the specified batch. When computing the loss
        masked elements are ignored.

        Args:
            batch (Tuple[torch.tensor, torch.tensor, torch.tensor]): 
            Input batch in the form (data, labels, padding mask)

        Returns:
            Tuple[torch.tensor, torch.tensor]: The prediction and the loss item
        """
        x, y, mask = batch
        x = x.float()
        x, _ = self.lstm(x)
        x = self.classification(x)
        x = self.softmax(x)

        loss = nn.functional.binary_cross_entropy(x[mask != 0].float(), y[mask != 0].float())

        return x, loss


    def evaluation_forward(self, x):
        x, _ = self.lstm(x)
        x = self.classification(x)
        print(f"Result after classification:{x}")
        y = self.softmax(x)
        print(f"Result after softmax: {x}")
        return (x,y)


    def configure_optimizers(self) -> Dict:
        """
        Configure the optimizers such that after 5 epochs without improvement the 
        learning rate is decreased.

        Returns:
            Dict: Optimizer configuration
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=self.segmentation_train_args.get("factor",0.1),
                patience=self.segmentation_train_args.get("patience",5)
                )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss"
            }
        }

