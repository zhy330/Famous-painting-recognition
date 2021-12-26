import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import time, datetime
import pdb, traceback
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from resnest.torch import resnest101

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter

dddd = {0: "Frida Kahlo",
        1: "Edgar Degas",
        2: "Pieter Bruegel",
        3: "Vincent van Gogh",
        4: "Rembrandt",
        5: "Henri Rousseau",
        6: "Henri Matisse",
        7: "Joan Miro",
        8: "Titian",
        9: "Paul Gauguin",
        10: "Pierre-Auguste Renoir",
        11: "Marc Chagall",
        12: "Raphael",
        13: "Leonardo da Vinci",
        14: "Amedeo Modigliani",
        15: "Sandro Botticelli",
        16: "Pablo Picasso",
        17: "Rene Magritte",
        18: "Vasiliy Kandinskiy",
        19: "Salvador Dali",
        20: "Michelangelo",
        21: "Mikhail Vrubel",
        22: "Paul Klee",
        23: "Camille Pissarro",
        24: "Giotto di Bondone",
        25: "Gustave Courbet",
        26: "Gustav Klimt",
        27: "Henri de Toulouse-Lautrec",
        28: "Francisco Goya",
        29: "Jan van Eyck",
        30: "Andrei Rublev",
        31: "Andy Warhol",
        32: "Alfred Sisley",
        33: "Paul Cezanne",
        34: "Diego Velazquez",
        35: "Edouard Manet",
        36: "Peter Paul Rubens",
        37: "Claude Monet",
        38: "Kazimir Malevich",
        39: "Hieronymus Bosch",
        40: "Caravaggio",
        41: "Piet Mondrian",
        42: "Diego Rivera",
        43: "El Greco",
        44: "William Turner",
        45: "Georges Seurat",
        46: "Jackson Pollock",
        47: "Edvard Munch",
        48: "Eugene Delacroix"
        }


class MyDataset(Dataset):
    def __init__(self, test_jpg):
        self.test_jpg = test_jpg

        self.transforms = transforms.Compose([
            transforms.Resize(660),
            transforms.RandomCrop(600),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #
        ])

    def __getitem__(self, index):
        img = Image.open(self.test_jpg[index]).convert('RGB')
        img = self.transforms(img)

        return img, torch.from_numpy(np.array(int('H' in self.test_jpg[index])))

    def __len__(self):
        return len(self.test_jpg)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.base = EfficientNet.from_pretrained('efficientnet-b3')
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._max_pooling = nn.AdaptiveMaxPool2d(1)
        num_ftrs = self.base._fc.in_features
        self.reduce_layer = nn.Conv2d(num_ftrs * 2, 512, 1)  # b3 num_ftrs=1536
        self._dropout = nn.Dropout(0.3)
        self._fc = nn.Linear(512, 49)


def forward(self, x):
    x = self.base.extract_features(x)
    x1 = self._avg_pooling(x)
    x2 = self._max_pooling(x)
    x = torch.cat([x1, x2], dim=1)
    x = self.reduce_layer(x)
    x = x.flatten(start_dim=1)
    x = self._dropout(x)
    x = self._fc(x)
    return x


# In[5]:


def predict(test_loader, model, tta=9):
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for i, (inputs, target) in enumerate(test_loader):
                inputs = inputs.cuda()  # cuda
                target = target.cuda()  # cuda

                output = model(inputs)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
        # print(test_pred_tta)

    return test_pred_tta


def test(s=r'C:\Users\86175\Desktop\795.jpg'):
    test_jpg = [s]
    # test_jpg = [r'D:\BaiduNetdiskDownload\IDA\pycharm\PyCharm 2021.2.3\se-flask\static\img\01.jpg']
    test_jpg = np.array(test_jpg)

    test_data = MyDataset(test_jpg)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    model = EfficientNet.from_pretrained('efficientnet-b3')
    num_trs = model._fc.in_features
    model._fc = nn.Linear(num_trs, 49)
    model = model.cuda()  # cuda

    test_pred = 0.
    pth = r'D:\Dos\aa.pth'
    model.load_state_dict(torch.load(pth))
    test_pred = predict(test_loader, model, tta=9)

    test_csv = pd.DataFrame()
    test_csv['num'] = list(range(800))
    ans = np.argmax(test_pred, 1)  #
    print("successfule load!")
    ans=ans[0]
    print(dddd[ans])
    return dddd[ans]


# In[11]:
if __name__ == '__main__':
    print(test())
