import os
import numpy as np
import time
import pandas as pd
import librosa
from torch.utils.data import DataLoader
from utils.Augmentation import convert_to_image
from model.BPDnet import BPDnet
import torch
from utils.Data_Loader import TestDataset
from threading import Thread  # 创建线程的模块
from tqdm import tqdm


class MyThread(Thread):
    def __init__(self, testing_folder, testing_file, sample_rate,
                location_model, weights_name, time_to_extract,
                prediction_folder, model):
        super().__init__()
        self.testing_folder = testing_folder
        self.testing_file = testing_file
        self.sample_rate = sample_rate
        self.location_model = location_model
        self.weights_name = weights_name
        self.time_to_extract = time_to_extract
        self.prediction_folder = prediction_folder
        self.model = model

    def run(self):  # 固定名字run ！！！必须用固定名
        execute_processing(self.testing_folder, self.testing_file, self.sample_rate,
                            self.location_model, self.weights_name, self.time_to_extract,
                            self.prediction_folder, self.model)


def execute_processing(testing_folder, testing_file, sample_rate,
                       location_model, weights_name, time_to_extract,
                       prediction_folder, model):
    """
    预测的总入口程序，完成整个音频文件的预测
    @param testing_folder:      测试文件的文件夹路径
    @param testing_file:        测试文件的文件名
    @param sample_rate:         音频文件的采样率
    @param location_model:      存放模型的文件夹路径
    @param weights_name:        模型文件名
    @param time_to_extract:     预测文件的切片长度
    @param prediction_folder:   存放预测结果的文件夹路径
    @return:                    None
    """

    # 加载音频文件数据
    print('Reading audio file (this can take some time)' + os.path.join(testing_folder, testing_file) + '...')
    test_file_audio, test_file_sample_rate = librosa.load(os.path.join(testing_folder, testing_file),
                                                          sr=sample_rate)
    print()
    print('Reading done.')
    end_reading = time.time()

    # 加载模型并进行预测，返回预测结果
    model_prediction = execute_batches(test_file_audio, time_to_extract, sample_rate, location_model, weights_name, model)

    # 返回每个预测（切片）结果对应的时间开始和结束点
    start_times, end_times = create_time_index(time_to_extract, int(len(test_file_audio) / test_file_sample_rate))
    results = pd.DataFrame(np.column_stack((start_times, end_times, model_prediction[:, 1], model_prediction[:, 0])),
                           columns=['Start(seconds)', 'End(seconds)', 'Pr(presence)', 'Pr(absence)'])

    np.savetxt(prediction_folder + testing_file + '_prediction.txt', model_prediction, fmt='%5f')
    results.to_excel(prediction_folder + testing_file + '_probabilities.xlsx', index=False)

    # 后处理
    segments = post_process(model_prediction, 0.76)
    end_prediction = time.time()
    end = time.time()

    print('---------------------------------------------------')
    print('Predicted segment start and end times:')
    print(segments)


def create_time_index(time_to_extract, file_duration_seconds):
    start = []
    end = []

    # Find out how many chunks of unit size (time_to_extract) can
    # be obtained
    amount_of_chunks = int(file_duration_seconds - time_to_extract + 1)

    # Iterate over each chunk to extract the frequencies
    for i in range(0, amount_of_chunks):
        start.append(i)
        end.append(time_to_extract + (i))

    return np.array(start), np.array(end)


def get_components(values):
    shifted = np.roll(values, 1)
    shifted[0] = 0
    difference = shifted - values
    shifted[difference < -200] = 0

    connected_component = []
    i = 0
    while i < len(shifted):
        # print ('i', i)
        if shifted[i] > 0:
            component = []
            j = 0
            while j < len(shifted) - i:
                if shifted[i + j] > 0:
                    component.append(shifted[i + j])
                else:
                    break
                j = j + 1
            connected_component.append(component)
            i = i + j

        i = i + 1

    return connected_component


def get_connected_components(components, verbose):
    gibbon_indices = []
    for component in components:
        if verbose:
            print('Start ', component[0])
            print('End ', component[-1] + 10)
        gibbon_indices.append([component[0], component[-1] + 10])

    return gibbon_indices


def check(preds):
    cleaned_components = []

    for component in preds:
        # print()
        # print (component)
        # print('len',len(component))

        rolled = component - np.roll(component, 1)
        rolled[0] = 0
        # print ('average', np.average(rolled))

        if len(component) < 20:
            continue

        if np.average(rolled) < 10:
            # print('add')
            cleaned_components.append(component)

    return cleaned_components


def execute_batches(audio, time_to_extract, sample_rate, location_model, weights_name, model):
    """
    预测8小时的音频文件结果
    @param audio:           音频文件数据
    @param time_to_extract: 切片时长
    @param sample_rate:     采样率
    @param location_model:  模型文夹路径
    @param weights_name:    模型文件名
    @return: 模型预测结果
    """
    batch_number = 8        # 8 小时
    model_predictions = []  # 存放预测结果
    start_index = 0         # 开始预测时间 /sec
    end_index = 60 * 60     # 1小时的预测时间 /sec

    for i in range(batch_number):
        print('Processing batch: {} out of {}'.format(i, batch_number))
        batch_prediction = process_batch(audio, start_index, end_index,
                                         time_to_extract, sample_rate, location_model, weights_name, model)
        model_predictions.extend(batch_prediction)
        start_index = end_index - 9
        end_index = end_index + 60 * 60

    return np.array(model_predictions)


def process_batch(audio, start_index, end_index, time_to_extract, sample_rate, location_model, weights_name, model):
    """
    每个 batch 来进行预测
    @param audio:           原始音频文件数据
    @param start_index:     开始预测的时间 /sec
    @param end_index:       结束预测的时间 / sec
    @param time_to_extract: 音频文件的切片长度
    @param sample_rate:     音频文件的采样率
    @param location_model:  模型的文件夹路径
    @param weights_name:    模型文件的名称
    @return:                一个 batch 的预测结果
    """

    # Extract segments from test file(返回音频文件每 10sec 的连续切片(hop len = 1 sec))
    X = create_X_new(audio,
                     time_to_extract,
                     sample_rate, start_index, end_index, verbose=False)

    # Convert data into spetrograms(将切片转换为mei谱图)
    X = convert_to_image(X)

    # ======================加载模型并进行预测=================================================
    # Build the model and load weights
    device = torch.device("cuda")
    model.load_state_dict(torch.load(os.path.join(location_model, weights_name))['model_state_dict'])
    model.to(device)

    ## Predict
    test_DataLoader = DataLoader(TestDataset(X), batch_size=128, num_workers=1, pin_memory=True, shuffle=False)
    del X
    model_prediction = []
    model.eval()
    with torch.no_grad():
        for data in test_DataLoader:
            data, _ = data
            data = data.to(device)
            output = torch.softmax(model(data), dim=-1)
            output = output.to(torch.device("cpu"))
            model_prediction.extend(output.numpy())
    # ========================================================================================
    return model_prediction


def post_process(predictions, threshold):
    values = predictions
    values = values[:, 0] > threshold
    values = values.astype(np.int32)
    values = np.where(values == 1)[0]

    component_prediction = get_components(values)
    predict_components = check(component_prediction)
    predict_components = get_connected_components(predict_components, 0)

    return predict_components


def create_X_new(mono_data, time_to_extract, sampleRate, start_index, end_index, verbose):
    X_frequences = []

    sampleRate = sampleRate
    duration = end_index - start_index - 9
    if verbose:
        print('-----------------------')
        print('start (seconds)', start_index)
        print('end (seconds)', end_index)
        print('duration (seconds)', (duration))
        print()
    counter = 0

    end_index = start_index + 10
    # Iterate over each chunk to extract the frequencies
    for i in range(0, duration):

        if verbose:
            print('Index:', counter)
            print('Chunk start time (sec):', start_index)
            print('Chunk end time (sec):', end_index)

        # Extract the frequencies from the mono file
        extracted = mono_data[int(start_index * sampleRate): int(end_index * sampleRate)]

        X_frequences.append(extracted)

        start_index = start_index + 1
        end_index = end_index + 1
        counter = counter + 1

    X_frequences = np.array(X_frequences)
    print(X_frequences.shape)
    if verbose:
        print()

    return X_frequences


def get_length_in_seconds(librosa_audio, sample_rate):
    return int(len(librosa_audio) / sample_rate)


def process():
    num_channel = [8, 16, 32, 64]
    for channel in tqdm(num_channel):
        model = BPDnet(channel, isSpecAugmentation=True)
        testing_Dir = '../Data/Raw_Data/Test'
        weights_name = "best.pth"
        location_model = f"../log/BPDnet{channel}_specAugment/model"
        result_Dir = f"../BPDnet_Result/BPDnet{channel}_specAugment/"
        time_to_extract = 10
        sample_rate = 4800
        t = []

        os.makedirs(result_Dir, exist_ok=True)

        for testing_folder in os.listdir(testing_Dir):
            torch.cuda.empty_cache()
            print()

            for testing_file in os.listdir(os.path.join(testing_Dir, testing_folder)):
                thread = MyThread(os.path.join(testing_Dir, testing_folder), testing_file, sample_rate,
                                  location_model, weights_name, time_to_extract, result_Dir, model)
                thread.start()
                t.append(thread)

            for thread in t:
                thread.join()

if __name__ == '__main__':
    num_channel = [8, 16, 32, 64]
    for channel in tqdm(num_channel):
        model = BPDnet(channel, isSpecAugmentation=True)
        testing_Dir = '../Data/Raw_Data/Test'
        weights_name = "best.pth"
        location_model = f"../log/BPDnet{channel}_specAugment/model"
        result_Dir = f"../BPDnet_Result/BPDnet{channel}_specAugment/"
        time_to_extract = 10
        sample_rate = 4800
        t = []

        os.makedirs(result_Dir, exist_ok=True)

        for testing_folder in os.listdir(testing_Dir):
            torch.cuda.empty_cache()
            print()

            for testing_file in os.listdir(os.path.join(testing_Dir, testing_folder)):
                thread = MyThread(os.path.join(testing_Dir, testing_folder), testing_file, sample_rate,
                                  location_model, weights_name, time_to_extract, result_Dir, model)
                thread.start()
                t.append(thread)

            for thread in t:
                thread.join()