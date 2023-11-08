import matplotlib.pyplot as plt
import librosa
import numpy as np
import time
from IPython.display import Audio, display
import ruptures as rpt  

def fig_ax(figsize=(15, 5), dpi=150):
    """Return a (matplotlib) figure and ax objects with given size."""
    return plt.subplots(figsize=figsize, dpi=dpi)


def load_and_display(path, sampling_rate):
    duration =400  # in seconds
    mp3_file = path
    signal, sampling_rate = librosa.load(mp3_file,duration = duration, sr = sampling_rate)

    print(f"Sampling rate: {sampling_rate}")

    # listen to the music
    display(Audio(data=signal, rate=sampling_rate))

    # look at the envelope
    fig, ax = fig_ax()
    ax.plot(np.arange(signal.size) / sampling_rate, signal)
    ax.set_xlim(0, signal.size / sampling_rate)
    ax.set_xlabel("Time (s)")
    _ = ax.set(title="Sound envelope")
    
    return signal


def computeFeature_Tempogram(signal, sampling_rate = 22050,hop_length_tempo = 256):
    """ Use librosa to extract the Tempogram from raw signal"""
    # Compute the onset strength
    
    oenv = librosa.onset.onset_strength(
        y=signal, sr=sampling_rate, hop_length=hop_length_tempo
    )
    # Compute the tempogram
    tempogram = librosa.feature.tempogram(
        onset_envelope=oenv,
        sr=sampling_rate,
        hop_length=hop_length_tempo,
    )

    print(tempogram)

    # Display the tempogram
    fig, ax = fig_ax()
    _ = librosa.display.specshow(
        tempogram,
        ax=ax,
        hop_length=hop_length_tempo,
        sr=sampling_rate,
        x_axis="s",
        y_axis="tempo",
    )

    return tempogram


def computeFeature_FourierTempogram(signal, sampling_rate = 22050,hop_length_tempo = 256):
    """ Use librosa to extract the Tempogram from raw signal"""
    # Compute the onset strength
    
    oenv = librosa.onset.onset_strength(
        y=signal, sr=sampling_rate, hop_length=hop_length_tempo
    )
    # Compute the tempogram
    tempogram = librosa.feature.tempogram(
        onset_envelope=oenv,
        sr=sampling_rate,
        hop_length=hop_length_tempo,
    )

    print(tempogram)

    # Display the tempogram
    fig, ax = fig_ax()
    _ = librosa.display.specshow(
        tempogram,
        ax=ax,
        hop_length=hop_length_tempo,
        sr=sampling_rate,
        x_axis="s",
        y_axis="tempo",
    )

    return tempogram



def get_sum_of_cost(algo,bkps) -> float:
    """Return the sum of costs for the change points `n_bkps`"""
    return algo.cost.sum_of_costs(bkps)

def computeSegmentation_and_display(extracted_data, algo, num_optimal_seg, sampling_rate=22050, hop_length_tempo=256):
    """
    Args: 
        extracted_data: feature that extract from raw signal(e.g. tempograh that extracted by librosa)
        algo: change detection algo (e.g. linear kernel method from ruptures)
        sampling_rate: how many sample per second, 22050 as default (as Librosa dose)
        hop_length_tempo: calculate onset strength per 'hop_length_tempo' sample

    Return:
        None
    """
    
    # Segmentation
    bkps = algo.predict(n_bkps=num_optimal_seg)

    # Obtain the cost (loss)
    cost = get_sum_of_cost(algo, bkps)
    print(f"Cost: {cost}")
    # Convert the estimated change points (frame counts) to actual timestamps
    bkps_times = librosa.frames_to_time(bkps, sr=sampling_rate, hop_length=hop_length_tempo)

    print(f"bkps_times: {bkps_times}")

    # Displaying results
    fig, ax = fig_ax()
    _ = librosa.display.specshow(
        extracted_data,
        ax=ax,
        x_axis="s",
        y_axis="tempo",
        hop_length=hop_length_tempo,
        sr=sampling_rate,
    )

    for b in bkps_times[:-1]:
        ax.axvline(b, ls="--", color="white", lw=4)

    return bkps_times 



def obtain_segmentation_time(audio_path, num_optimal_seg, algo_type,algo_name, sampling_rate=22050, hop_length_tempo = 256):
    """
    At the moment, choose Tempogram as the feature that we extract from raw signal
    """
    
    # load the mp3 file from audio_path 
    signal = load_and_display(audio_path,sampling_rate)

    # use selected kernel method and calculated Tempogram to fit the model
    start_time = time.time()
    tempogram = computeFeature_Tempogram(signal, sampling_rate)
    
    print(f"Algo Type: {algo_type}, Algo Name: {algo_name}")
    if algo_type == "kernel":
        if algo_name == "linear_kernel":
            algo = rpt.KernelCPD(kernel="linear").fit(tempogram.T)
        else:   
            algo = rpt.KernelCPD(kernel="rbf").fit(tempogram.T)
    else:
        algo = rpt.Dynp(model="l2").fit(tempogram.T)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"The Algo spends {total_time:.2f} seconds to fit the Tempogram")


    # Slove the optimization problem
    # and devide the signal into 'num_optimal_seg' segmentations
    start_time = time.time()
    bkps_times = computeSegmentation_and_display(tempogram, algo, num_optimal_seg, sampling_rate=sampling_rate, hop_length_tempo=hop_length_tempo)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"The Algo spend {total_time:.2f} seconds to get the segmentation time points")

    return signal, sampling_rate, bkps_times




def output(signal,bkps_times,sampling_rate=22050):
    # Compute change points corresponding indexes in original signal
    bkps_time_indexes = (sampling_rate * bkps_times).astype(int).tolist()
    print(f"bkps_time_indexes: {bkps_time_indexes}")


    for segment_number, (start, end) in enumerate(
        rpt.utils.pairwise([0] + bkps_time_indexes), start=1
    ):
        segment = signal[start:end]
        print(f"Segment n°{segment_number} (duration: {segment.size/sampling_rate:.2f} s)")
        display(Audio(data=segment, rate=sampling_rate))