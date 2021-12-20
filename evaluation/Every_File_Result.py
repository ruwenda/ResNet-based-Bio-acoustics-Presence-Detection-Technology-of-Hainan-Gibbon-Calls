import os
import numpy as np
import pandas as pd
from sklearn import metrics
from .evaluation import process

def getScore(testLabel, predictLabel):
    accuracy = metrics.accuracy_score(testLabel, predictLabel)      # 准确率
    precision = metrics.precision_score(testLabel, predictLabel)    # 精确率
    recall = metrics.recall_score(testLabel, predictLabel)          # 召回率
    F1 = metrics.f1_score(testLabel, predictLabel)                  # F1 score

    return accuracy, precision, recall, F1


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return TP, FP, TN, FN


def my_post_process(label, threshold1=1, threshold2=1):
    postPredict = np.copy(label)

    idx = np.where(label == 0)[0]
    for i in range(1, len(idx) - 1):
        if idx[i] - idx[i - 1] > threshold2 and \
                idx[i + 1] - idx[i] > threshold2:
            postPredict[idx[i], 0] = 1

    idx = np.where(label == 1)[0]
    for i in range(len(idx)):
        if i == 0:
            if idx[i + 1] - idx[i] > threshold1:
                postPredict[idx[i], 0] = 0
        elif i == len(idx) - 1:
            if idx[i] - idx[i - 1] > threshold1:
                postPredict[idx[i], 0] = 0
        else:
            if idx[i] - idx[i - 1] > threshold1 and idx[i + 1] - idx[i] > threshold1:
                postPredict[idx[i], 0] = 0

    return postPredict

def post_process(predictions, threshold):
    values = predictions
    values = values[:, 0] > threshold
    values = values.astype(np.int32)
    values = np.where(values == 1)[0]

    component_prediction = get_components(values)
    predict_components = check(component_prediction)
    predict_components = get_connected_components(predict_components, 0)

    return predict_components

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
        rolled = component - np.roll(component, 1)
        rolled[0] = 0

        if len(component) < 20:
            continue

        if np.average(rolled) < 10:
            # print('add')
            cleaned_components.append(component)

    return cleaned_components


def get_specificity(y_actual, y_hat):
    TP, FP, TN, FN = perf_measure(y_actual, y_hat)
    return TN / (FP + TN)


if __name__ == '__main__':
    process()

    threshold = 0.91
    testDir = "../Data/Test_Labels"
    predictPath = f"../BPDnet_Result/"
    predictDirList = os.listdir(predictPath)
    resultDir = f"../score"
    resultList = []
    resultPerFile = []
    resultPerFilePost = []
    resultPerFilePost_Paper = []
    resultPerFilePost_synthesize = []
    os.makedirs(resultDir, exist_ok=True)

    for predictDir in predictDirList:   # 一级目录
        totalPredictLabel = []
        totalTestLabel = []
        totalPostLabel = []
        totalPostLabel_Paper = []
        totalPostLabel_synthesize = []
        total_mask_list = []
        total_mask_list2 = []
        for fileName in os.listdir(testDir):    # 二级目录
            # 取出对应文件的预测结果
            predictLabel = np.loadtxt(os.path.join(predictPath, predictDir,
                                                   fileName.split(".")[0][:-5] + ".wav_prediction.txt"))
            testDataIndex = pd.read_csv(os.path.join(testDir, fileName),
                                        index_col=False).to_numpy()
            testLabel = np.zeros((predictLabel.shape[0], 1))

            # 构建正确标记的数据序列(输入的特征持续时长（10秒）应包含全部的标签)
            for idx in range(testDataIndex.shape[0]):
                testLabel[testDataIndex[idx][0] - 8:
                          testDataIndex[idx][0] + testDataIndex[idx][1] - 2, 0] = 1

            total_mask_list.extend(predictLabel[:, 0].reshape((-1, 1)) * testLabel.reshape((-1, 1)))
            total_mask_list2.extend(predictLabel[:, 1].reshape((-1, 1)) * np.abs((testLabel.reshape((-1, 1)) - 1)))

            # 根据阈值构建预测结果序列
            predictLabel = np.int32(predictLabel[:, 1] >= threshold)

            # 本人的后处理
            postPredict = my_post_process(predictLabel.reshape((-1, 1)).reshape((-1, 1)))

            # 论文的后处理方法
            post_section = post_process(predictLabel.reshape((-1, 1)), threshold)
            postPredict_Paper = np.zeros(testLabel.shape)
            postPredict_Paper = postPredict_Paper.reshape((-1, 1))
            for section in post_section:
                postPredict_Paper[section[0]: section[1], 0] = testLabel[section[0]: section[1], 0]

            # 本人 + 论文 的后处理方法
            post_synthesize_section = post_process(postPredict.reshape((-1, 1)), threshold)
            postPredict_Synthesize = np.zeros(testLabel.shape)
            postPredict_Synthesize = postPredict_Synthesize.reshape((-1, 1))
            for section in post_synthesize_section:
                postPredict_Synthesize[section[0]: section[1], 0] = testLabel[section[0]: section[1], 0]

            # 计算本模型本文件的预测结果
            # 预测结果：
            accuracy, precision, recall, F1 = getScore(testLabel.reshape([-1, 1]), predictLabel.reshape([-1, 1]))
            specificity = get_specificity(testLabel.reshape([-1, 1]), predictLabel.reshape([-1, 1]))
            resultPerFile.append({"model": predictDir, "file:": fileName.split(".")[0], "accuracy": accuracy,
                  "precision": precision, "recall": recall, "specificity": specificity, "F1_score": F1})

            # 本人的后处理结果：
            accuracy, precision, recall, F1 = getScore(testLabel.reshape([-1, 1]), postPredict.reshape([-1, 1]))
            specificity = get_specificity(testLabel.reshape([-1, 1]), postPredict.reshape([-1, 1]))
            resultPerFilePost.append({"model": predictDir + "_post", "file:": fileName.split(".")[0], "accuracy": accuracy,
                  "precision": precision, "recall": recall, "specificity": specificity, "F1_score": F1})

            # 论文的后处理结果：
            accuracy, precision, recall, F1 = getScore(testLabel.reshape([-1, 1]), postPredict_Paper.reshape([-1, 1]))
            specificity = get_specificity(testLabel.reshape([-1, 1]), postPredict_Paper.reshape([-1, 1]))
            resultPerFilePost_Paper.append({"model": predictDir + "_post_paper", "file:": fileName.split(".")[0], "accuracy": accuracy,
                  "precision": precision, "recall": recall, "specificity": specificity, "F1_score": F1, "post_section": post_section} )

            # 本人 + 论文 后处理结果：
            accuracy, precision, recall, F1 = getScore(testLabel.reshape([-1, 1]), postPredict_Synthesize.reshape([-1, 1]))
            specificity = get_specificity(testLabel.reshape([-1, 1]), postPredict_Synthesize.reshape([-1, 1]))
            resultPerFilePost_synthesize.append({"model": predictDir + "_post_synthesize", "file:": fileName.split(".")[0], "accuracy": accuracy,
                  "precision": precision, "recall": recall, "specificity": specificity, "F1_score": F1})

            # 将本模型本文件的预测结果保存
            totalPredictLabel.extend(predictLabel)
            totalTestLabel.extend(testLabel)
            totalPostLabel.extend(postPredict)
            totalPostLabel_Paper.extend(postPredict_Paper)
            totalPostLabel_synthesize.extend(postPredict_Synthesize)

        # list --> numpy
        totalTestLabel = np.array(totalTestLabel)
        totalPredictLabel = np.array(totalPredictLabel)
        totalPostLabel = np.array(totalPostLabel)
        totalPostLabel_Paper = np.array(totalPostLabel_Paper)
        totalPostLabel_synthesize = np.array(totalPostLabel_synthesize)

        # 本模型的综合结果
        accuracy, precision, recall, F1 = getScore(totalTestLabel.reshape([-1, 1]), totalPredictLabel.reshape([-1, 1]))
        specificity = get_specificity(totalTestLabel.reshape([-1, 1]), totalPredictLabel.reshape([-1, 1]))
        resultPerFile.append({"model": predictDir, "file:": "all files", "accuracy": accuracy,
                              "precision": precision, "recall": recall, "specificity": specificity, "F1_score": F1})
        print(resultPerFile[-1])

        # 本模型的本人后处理方法后综合结果
        accuracy, precision, recall, F1 = getScore(totalTestLabel.reshape([-1, 1]), totalPostLabel.reshape([-1, 1]))
        specificity = get_specificity(totalTestLabel.reshape([-1, 1]), totalPostLabel.reshape([-1, 1]))
        resultPerFilePost.append({"model": predictDir + "_post", "file:": "all files", "accuracy": accuracy,
                                  "precision": precision, "recall": recall, "specificity": specificity, "F1_score": F1})
        print(resultPerFilePost[-1])

        # 本模型的论文后处理方法后综合结果
        accuracy, precision, recall, F1 = getScore(totalTestLabel.reshape([-1, 1]), totalPostLabel_Paper.reshape([-1, 1]))
        specificity = get_specificity(totalTestLabel.reshape([-1, 1]), totalPostLabel_Paper.reshape([-1, 1]))
        resultPerFilePost_Paper.append({"model": predictDir + "_post_paper", "file:": "all files", "accuracy": accuracy,
                                  "precision": precision, "recall": recall, "specificity": specificity, "F1_score": F1})
        print(resultPerFilePost_Paper[-1])

        # 本模型的 本人 + 论文 后处理方法综合结果
        accuracy, precision, recall, F1 = getScore(totalTestLabel.reshape([-1, 1]), totalPostLabel_synthesize.reshape([-1, 1]))
        specificity = get_specificity(totalTestLabel.reshape([-1, 1]), totalPostLabel_synthesize.reshape([-1, 1]))
        resultPerFilePost_synthesize.append({"model": predictDir + "_post_synthesize", "file:": "all files", "accuracy": accuracy,
                                  "precision": precision, "recall": recall, "specificity": specificity, "F1_score": F1})
        print(resultPerFilePost_synthesize[-1])

        print("#------------------------------------------------------"
              "------------------------------------------------#\n")

        # 保存结果----------------------------------------------------------------------------------------------------------
        # if not os.path.exists("Data/compare_label"):
        #     os.mkdir("Data/compare_label")
        # np.savetxt(os.path.join("Data/compare_label", predictDir + "mask_predict.txt"),
        #            np.array(total_mask_list))
        # np.savetxt(os.path.join("Data/compare_label", predictDir + "mask_predict2.txt"),
        #            np.array(total_mask_list2))

        resultPerFile.append({})
        resultPerFile.append({})
        resultPerFilePost.append({})
        resultPerFilePost.append({})
        resultPerFilePost_Paper.append({})
        resultPerFilePost_Paper.append({})
        resultPerFilePost_synthesize.append({})
        resultPerFilePost_synthesize.append({})

    # 保存每个模型对于每个测试文件的预测结果：
    path = os.path.join(resultDir, "Per_File_Result")
    if not os.path.exists(path):
        os.mkdir(path)

    # 模型结果
    dataResult = pd.DataFrame(resultPerFile)
    dataResult.to_excel(os.path.join(path, "Per_File_Result.xlsx"))

    # 本人后处理结果
    dataResult = pd.DataFrame(resultPerFilePost)
    dataResult.to_excel(os.path.join(path, "Per_File_Result_Post.xlsx"))

    # 论文后处理结果
    dataResult = pd.DataFrame(resultPerFilePost_Paper)
    dataResult.to_excel(os.path.join(path, "Per_File_Result_Post_Paper.xlsx"))

    # 综合后处理结果
    dataResult = pd.DataFrame(resultPerFilePost_synthesize)
    dataResult.to_excel(os.path.join(path, "Per_File_Result_Post_Synthesize.xlsx"))





