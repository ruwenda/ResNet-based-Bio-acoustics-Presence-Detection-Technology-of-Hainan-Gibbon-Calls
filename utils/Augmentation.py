import numpy as np
import librosa
import librosa.display
import random


def blend(audio_1, audio_2, w_1=0.9, w_2=0.1):
    """
    有声段与背景噪声段融合
    @param audio_1: 有声段数据
    @param audio_2: 噪声段数据
    @param w_1: 有声段融合权重
    @param w_2: 噪声段融合权重
    @return: 增广有声片
    """
    augmented = w_1 * audio_1 + w_2 * audio_2
    return augmented


def time_shift(audio, time, sample_rate):
    """
    对噪声段进行时间平移增广
    @param audio: 噪声段数据
    @param time:  时间平移时间点
    @param sample_rate: 采样率
    @return: 增广的噪声段
    """
    augmented = np.zeros(len(audio))
    augmented[0:sample_rate*time] = audio[-sample_rate*time:]
    augmented[sample_rate*time:] = audio[:-sample_rate*time]
    return augmented


def convert_to_image(audio):
    n_fft = 1024
    hop_length = 256
    n_mels = 128
    # 改成鸟声时，这两个参数要改
    f_min = 1000    # mel谱的最小频率
    f_max = 2000    # mel谱的最大频率

    X_img = []

    for data in audio:
        S = librosa.feature.melspectrogram(data, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, sr=4800, power=1.0, fmin = f_min, fmax=f_max)
        X_img.append(S)

    X_img = np.asarray(X_img)
    X_img = np.reshape(X_img, (X_img.shape[0], X_img.shape[1], X_img.shape[2], 1))

    return X_img


def augment_data(seed, augmentation_amount, augmentation_probability,
                 gibbon_calls, background_noise, sample_rate, alpha):
    '''
    seed: allows user to specify a random seed
    augmentation_amount: the number of new data points to create using augmentation
    augmentation_probability: the probability to create a new data point
    gibbon_calls: numpy array containing gibbon calls
    background_noise: numpy array containing background noise (no gibbon)
    sample_rate: sample rate of the audio files
    alpha: size of the audio files (seconds)
    '''
    np.random.seed(seed)
    random.seed(seed)
    
    augmented_data = []
    
    # Iterate over each gibbon data point
    for gibbon_data in gibbon_calls:
        
        # Iterate over the number of augmentations needed
        for i in range(0, augmentation_amount):
        
            # Randomly draw a value between 0 and 1
            probability = random.random()
            
            # If the value is less than the user-defined parameter
            # then augment.
            if probability <= augmentation_probability:

                # Randomly select a background noise segment
                random_background = random.randint(0, len(background_noise)-1)

                # Randomly select amount to shift by
                random_time_point = random.randint(1, alpha-1)

                # Create augmented data using time shift
                new_data = time_shift(background_noise[random_background], random_time_point, sample_rate)

                # Blend the two files with equal weighting
                new_data = blend(gibbon_data, new_data, 0.9, 0.1)

                # Append
                augmented_data.append(new_data)


    # Convert to numpy array
    return np.asarray(augmented_data)


def augment_background(seed, augmentation_amount, augmentation_probability,
                 background_noise, sample_rate, alpha):
    
    '''
    seed: allows user to specify a random seed
    augmentation_amount: the number of new data points to create using augmentation
    augmentation_probability: the probability to create a new data point
    background_noise: numpy array containing background noise (no gibbon)
    sample_rate: sample rate of the audio files
    alpha: size of the audio files (seconds)
    '''

    np.random.seed(seed)
    random.seed(seed)
    
    augmented_data = []
    
    # Iterate over each background data point
    for background_data in background_noise:
        
        # Iterate over the number of augmentations needed
        for i in range (0, augmentation_amount):
            
            # Randomly draw a value between 0 and 1
            probability = random.random()
        
            # If the value is less than the user-defined parameter
            # then augment.
            if probability <= augmentation_probability:

                # Randomly select amount to shift by
                random_time_point = random.randint(1, alpha-1)

                # Create augmented data using time shift
                new_data = time_shift(background_data, 
                		  random_time_point, sample_rate)

                # Append
                augmented_data.append(new_data)

    # Convert to numpy array
    return np.asarray(augmented_data)

