from ..utils.Training_Data_Process import execute_preprocessing_all_files

if __name__ == '__main__':
    # 特征提取和数据增广的参数配置
    audio_directory = '../Data/Raw_Data/Train/'
    timestamp_directory = '../Data/Call_Labels/'
    save_location = '../Data/Pickled_Data/'
    augment_directory = '../Data/Augmented_Data/'
    augment_image_directory = '../Data/Augmented_Image_Data/'
    training_file = '../Data/Training_Files.txt'

    sample_rate = 4800
    number_seconds_to_extract = 10
    seed = 42
    augmentation_probability = 1.0
    augmentation_amount_noise = 2
    augmentation_amount_gibbon = 10

    # 进行切片、数据增广、特征提取
    execute_preprocessing_all_files(training_file, audio_directory,
                                    sample_rate, timestamp_directory,
                                    number_seconds_to_extract, save_location,
                                    augmentation_amount_noise, augmentation_probability,
                                    augmentation_amount_gibbon, seed, augment_directory, augment_image_directory)