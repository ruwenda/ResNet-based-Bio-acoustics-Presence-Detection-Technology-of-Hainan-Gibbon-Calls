import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, dataDir: str, isGibbon: bool):
        self.dataDir = dataDir
        self.isGibbon = isGibbon
        self.dataList = []
        self.label = np.array([1, 0], dtype=np.float32) if self.isGibbon else np.array([0, 1], dtype=np.float32)
        if isGibbon:
            for fileName in os.listdir(dataDir):
                if fileName[0] == "g":
                    self.dataList.append(fileName)
        else:
            for fileName in os.listdir(dataDir):
                if fileName[0] == "n":
                    self.dataList.append(fileName)

    def __getitem__(self, idx):
        data = pickle.load(open(os.path.join(self.dataDir, self.dataList[idx]), "rb"))
        data = torch.tensor(data, dtype=torch.float32)
        data = data.permute(2, 0, 1)
        return data, self.label

    def __len__(self):
        return len(self.dataList)

class TestDataset(Dataset):
    def __init__(self, x):
        self.data = torch.tensor(x, dtype=torch.float32)
        self.data = self.data.permute(0, 3, 1, 2)

    def __getitem__(self, item):
        return self.data[item], torch.ones((1, 2))

    def __len__(self):
        return self.data.shape[0]

def getDataset(dataDir):
    dataset = MyDataset(dataDir, isGibbon=True)\
              + MyDataset(dataDir, isGibbon=False)
    return dataset

def getDataLoader(dataDir, batchSize):
    dataset = getDataset(dataDir)
    dataLoader = DataLoader(dataset=dataset, batch_size=batchSize,
                            shuffle=True, num_workers=1)
    return dataLoader

if __name__ == '__main__':
    dataset = MyDataset("../Data/Augmented_Image_Data", False)
    print(dataset.__len__())
    data, target = dataset[0]
    print(data.shape)