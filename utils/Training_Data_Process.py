import random
import itertools
import pickle, librosa
import matplotlib.pyplot as plt
from .Extract_Audio_Helper import *
from .Augmentation import augment_data, augment_background, convert_to_image


def execute_audio_extraction(audio_directory, audio_file_name, sample_rate, timestamp_directory,
                             number_seconds_to_extract, save_location):
    """
    根据有声段和无声段标签对原始音频进行切片,并保存为 .plk 文件
    @param audio_directory:     原始音频文件的文件夹路径
    @param audio_file_name:     文件名
    @param sample_rate:         采样率
    @param timestamp_directory: 存放标签的文件夹路径
    @param number_seconds_to_extract: 每个切片的事件长度
    @param save_location:       保存文件的文件夹路径
    @return:
    """

    print('Reading audio file (this can take some time)...')
    # Read in audio file
    librosa_audio, librosa_sample_rate = librosa.load(audio_directory + audio_file_name,
                                                      sr=sample_rate)

    print()
    print('Reading done.')

    # Read gibbon labelled timestamp file
    gibbon_timestamps = read_and_process_gibbon_timestamps(timestamp_directory,
                                                           'g_' + audio_file_name[
                                                                  :audio_file_name.find('.wav')] + '.data',
                                                           sample_rate, sep=',')
    # Read non-gibbon labelled timestamp file
    non_gibbon_timestamps = read_and_process_nongibbon_timestamps(timestamp_directory,
                                                                  'n_' + audio_file_name[
                                                                         :audio_file_name.find('.wav')] + '.data',
                                                                  librosa_sample_rate, sep=',')
    # Extract gibbon calls
    gibbon_extracted = extract_all_gibbon_calls(librosa_audio, gibbon_timestamps,
                                                number_seconds_to_extract, 1, librosa_sample_rate, 0)

    # Extract background noise
    noise_extracted = extract_all_nongibbon_calls(librosa_audio, non_gibbon_timestamps,
                                                  number_seconds_to_extract, 5, librosa_sample_rate, 0)
    # Save the extracted data to disk
    pickle.dump(gibbon_extracted,
                open(save_location + 'g_' + audio_file_name[:audio_file_name.find('.wav')] + '.pkl', "wb"))
    pickle.dump(noise_extracted,
                open(save_location + 'n_' + audio_file_name[:audio_file_name.find('.wav')] + '.pkl', "wb"))

    del librosa_audio
    print()
    print('Extracting segments done. Pickle files saved.')

    return gibbon_extracted, noise_extracted


def execute_augmentation(gibbon_extracted,
                         non_gibbon_extracted, number_seconds_to_extract, sample_rate,
                         augmentation_amount_noise, augmentation_probability,
                         augmentation_amount_gibbon, seed, augment_directory, augment_image_directory,
                         audio_file_name):
    """
    @param gibbon_extracted:            长臂猿叫声片段buff， dtype=np.ndarray
    @param non_gibbon_extracted:        背景噪声静音片段buff, dtype=np.ndarray
    @param number_seconds_to_extract:   每个切片的时间长度 /s
    @param sample_rate:                 采样率
    @param augmentation_amount_noise:   噪声片段的增广副本数
    @param augmentation_probability:    增广概率
    @param augmentation_amount_gibbon:  长臂猿叫声的增广副本数
    @param seed:                        随机种子
    @param augment_directory:           增广音频派那段的保存路径
    @param augment_image_directory:     增广谱图数据的保存路径
    @param audio_file_name:             音频文件名
    @return:                            None
    """
    print()
    print('gibbon_extracted:', gibbon_extracted.shape)
    print('non_gibbon_extracted:', non_gibbon_extracted.shape)

    # 随机的将背景噪声的一段截取前面的一截拼接到最后得到背景噪声的数据增广结果
    non_gibbon_extracted_augmented = augment_background(seed, augmentation_amount_noise,
                                                        augmentation_probability, non_gibbon_extracted,
                                                        sample_rate, number_seconds_to_extract)

    # 对一段有声段，随机选取augmentation_amount_gibbon个不同的背景噪声片段，经过时间的反转拼接之后进行加权求和式的融合
    gibbon_extracted_augmented = augment_data(seed, augmentation_amount_gibbon,
                                              augmentation_probability, gibbon_extracted,
                                              non_gibbon_extracted_augmented, sample_rate,
                                              number_seconds_to_extract)

    # 获得增广之后的有声段的总数
    sample_amount = gibbon_extracted_augmented.shape[0]
    # 得到与有声片段相同数量的背景噪声片段
    non_gibbon_extracted_augmented = non_gibbon_extracted_augmented[
        np.random.choice(non_gibbon_extracted_augmented.shape[0],
                         sample_amount,
                         replace=True)]

    print()
    print('gibbon_extracted_augmented:', gibbon_extracted_augmented.shape)
    print('non_gibbon_extracted_augmented:', non_gibbon_extracted_augmented.shape)

    pickle.dump(gibbon_extracted_augmented,
                open(augment_directory + 'g_' + audio_file_name[:audio_file_name.find('.wav')] + '_augmented.pkl',
                     "wb"))

    pickle.dump(non_gibbon_extracted_augmented,
                open(augment_directory + 'n_' + audio_file_name[:audio_file_name.find('.wav')] + '_augmented.pkl',
                     "wb"))

    # 将数据转化为mel谱
    gibbon_extracted_augmented_image = convert_to_image(gibbon_extracted_augmented)
    non_gibbon_extracted_augmented_image = convert_to_image(non_gibbon_extracted_augmented)

    print()
    print('gibbon_extracted_augmented_image:', gibbon_extracted_augmented_image.shape)
    print('non_gibbon_extracted_augmented_image:', non_gibbon_extracted_augmented_image.shape)

    for i in range(gibbon_extracted_augmented_image.shape[0]):
        pickle.dump(gibbon_extracted_augmented_image[i],
                    open(augment_image_directory + 'g_' +
                         audio_file_name[:audio_file_name.find('.wav')] + '_augmented_img' + f"_{i}" + '.pkl', "wb"))

        pickle.dump(non_gibbon_extracted_augmented_image[i],
                    open(augment_image_directory + 'n_' +
                         audio_file_name[:audio_file_name.find('.wav')] + '_augmented_img' + f"_{i}" + '.pkl', "wb"))

    del non_gibbon_extracted_augmented, gibbon_extracted_augmented

    print()
    print('Augmenting done. Pickle files saved to...')

    return gibbon_extracted_augmented_image, non_gibbon_extracted_augmented_image


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def create_seed():
    return random.randint(1, 1000000)


def execute_preprocessing_all_files(training_file, audio_directory,
                                    sample_rate, timestamp_directory,
                                    number_seconds_to_extract, save_location,
                                    augmentation_amount_noise, augmentation_probability,
                                    augmentation_amount_gibbon, seed, augment_directory, augment_image_directory):
    """
    从原始音频提取特征并进行数据增强
    @param training_file:               记录需要进行训练的音频文件的文件名（一行一个音频文件名），并进行特征提取和数据增强
    @param audio_directory:             存放训练数据的原始音频文件路径
    @param sample_rate:                 采样率
    @param timestamp_directory:         存放对应文件的标签地址（包括静音和非静音段）
    @param number_seconds_to_extract:   每个切片的长度
    @param save_location:               所提取特征的地址
    @param augmentation_amount_noise:   每个10s片段的噪声的数据增广副本数 2
    @param augmentation_probability:    增广概率 1.0
    @param augmentation_amount_gibbon:  每个10s片段的长臂猿叫声的增广副本数 10
    @param seed:                        随机种子 42
    @param augment_directory:           数据增广后的数据存放地址
    @param augment_image_directory:     增广后存放的谱图地址
    @return:
    """

    with open(training_file) as fp:
        line = fp.readline()

        while line:
            file_name = line.strip()
            print('Processing file: {}'.format(file_name))

            # Extract segments from audio files(根据标签对数据进行有声段和无声段的切片并进行保存)
            gibbon_extracted, non_gibbon_extracted = execute_audio_extraction(audio_directory,
                                                                              file_name, sample_rate,
                                                                              timestamp_directory,
                                                                              number_seconds_to_extract, save_location)

            # Augment the extracted segments(对数据进行增广、平衡、及谱图特征的提取)
            _, _ = execute_augmentation(
                gibbon_extracted,
                non_gibbon_extracted, number_seconds_to_extract, sample_rate,
                augmentation_amount_noise, augmentation_probability,
                augmentation_amount_gibbon, seed, augment_directory, augment_image_directory,
                file_name)

            # Read next line
            line = fp.readline()





