import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import codecs
import pickle
from tqdm import tqdm

dict_size = 2884
batch_size = 32
num_train_data = 128
num_test_data = 32

with open("worddict.pickle", "rb") as fp:
    word_dict = pickle.load(fp)
inv_word_dict = dict(zip(word_dict.values(), word_dict.keys()))
inv_word_dict[0] = ""

def editDistance(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)]
              for i in range(len(str1) + 1)]
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if(str1[i-1] == str2[j-1]):
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i]
                               [j-1]+1, matrix[i-1][j-1]+d)
    return matrix[len(str1)][len(str2)]


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        dataset_path = "./datas/data_thchs30"
        self.mfcc_mat = np.load(os.path.join(
            dataset_path, "mfcc_vec_680x26.npy"))
        mfcc_mat_min = self.mfcc_mat.min()
        mfcc_mat_max = self.mfcc_mat.max()
        self.mfcc_mat = (self.mfcc_mat - mfcc_mat_min) / \
            (mfcc_mat_max - mfcc_mat_min)
        print("Normalized", self.mfcc_mat.min(), self.mfcc_mat.max())
        with codecs.open(os.path.join(dataset_path, "all_texts.txt"), encoding="utf-8") as file_read:
            text_lines = file_read.readlines()
#         token_set = set(list(''.join(text_lines).replace("\n", "")))
#         token_map = dict((j, i+1) for i, j in enumerate(token_set))
        seq_lines = [list(map(lambda x: word_dict[x], text_line.replace(
            "\n", ""))) for text_line in text_lines]
        self.pad_lines = [(seq_line + [0]*48)[:48] for seq_line in seq_lines]
        self.pad_lines = torch.tensor(self.pad_lines).unsqueeze(-1)
        # 小数据测试
        self.mfcc_mat = self.mfcc_mat[:num_train_data+num_test_data]
        self.pad_lines = self.pad_lines[:num_train_data+num_test_data]

    def __len__(self):
        return len(self.mfcc_mat)

    def __getitem__(self, idx):
        return self.mfcc_mat[idx], self.pad_lines[idx]


class ResBlock(torch.nn.Module):
    def __init__(self, size, dila, dim):
        super(ResBlock, self).__init__()
        self.f1 = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim, kernel_size=size,
                            stride=1, dilation=dila, padding=(size-1)*dila//2),
            torch.nn.BatchNorm1d(dim),
            torch.nn.Tanh()
        )
        self.f2 = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim, kernel_size=size,
                            stride=1, dilation=dila, padding=(size-1)*dila//2),
            torch.nn.BatchNorm1d(dim),
            torch.nn.Sigmoid()
        )
        self.f = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim, kernel_size=1, stride=1,
                            dilation=dila, padding=0),
            torch.nn.BatchNorm1d(dim),
            torch.nn.Tanh()
        )
        self.shortcut = torch.nn.Sequential()

    def forward(self, x):
        x1 = self.f1(x)
        x2 = self.f2(x)
        x3 = x1 * x2
        out = self.f(x3)
        out = out + self.shortcut(x)
        return out


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.pre = torch.nn.Sequential(
            torch.nn.Conv1d(26, 192, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm1d(192),
            torch.nn.Tanh(),
        )

        self.b11 = ResBlock(7, 1, 192)
        self.b12 = ResBlock(7, 2, 192)
        self.b13 = ResBlock(7, 4, 192)
        self.b14 = ResBlock(7, 8, 192)
        self.b15 = ResBlock(7, 16, 192)

        self.b21 = ResBlock(7, 1, 192)
        self.b22 = ResBlock(7, 2, 192)
        self.b23 = ResBlock(7, 4, 192)
        self.b24 = ResBlock(7, 8, 192)
        self.b25 = ResBlock(7, 16, 192)

        self.b31 = ResBlock(7, 1, 192)
        self.b32 = ResBlock(7, 2, 192)
        self.b33 = ResBlock(7, 4, 192)
        self.b34 = ResBlock(7, 8, 192)
        self.b35 = ResBlock(7, 16, 192)

        self.b41 = ResBlock(7, 1, 192)
        self.b42 = ResBlock(7, 2, 192)
        self.b43 = ResBlock(7, 4, 192)
        self.b44 = ResBlock(7, 8, 192)
        self.b45 = ResBlock(7, 16, 192)

        self.post = torch.nn.Sequential(
            torch.nn.Conv1d(192, 192, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm1d(192),
            torch.nn.Tanh(),
            torch.nn.Conv1d(192, dict_size, kernel_size=1,
                            stride=1, padding=0),
        )

    def forward(self, x):
        x = self.pre(x)
        skip = x

        x = self.b11(x)
        skip = skip + x
        x = self.b12(x)
        skip = skip + x
        x = self.b13(x)
        skip = skip + x
        x = self.b14(x)
        skip = skip + x
        x = self.b15(x)
        skip = skip + x

        x = self.b21(x)
        skip = skip + x
        x = self.b22(x)
        skip = skip + x
        x = self.b23(x)
        skip = skip + x
        x = self.b24(x)
        skip = skip + x
        x = self.b25(x)
        skip = skip + x

        x = self.b31(x)
        skip = skip + x
        x = self.b32(x)
        skip = skip + x
        x = self.b33(x)
        skip = skip + x
        x = self.b34(x)
        skip = skip + x
        x = self.b35(x)
        skip = skip + x

        x = self.b41(x)
        skip = skip + x
        x = self.b42(x)
        skip = skip + x
        x = self.b43(x)
        skip = skip + x
        x = self.b44(x)
        skip = skip + x
        x = self.b45(x)
        skip = skip + x

        res = self.post(skip)
        return res




def printSeq(seq):
    print([inv_word_dict[i] for i in seq])

import os
import math
import time

os.system("rm model")
tm = time.time()

while True:
    torch.cuda.empty_cache()
    my_dataset = MyDataset()

    # data_train, data_test = torch.utils.data.random_split(my_dataset, [13000, 388])
    data_train, data_test = torch.utils.data.random_split(my_dataset, [
                                                          num_train_data, num_test_data])

    model = MyModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=1, verbose=True)
    dataloader_train = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=False)

    if os.path.exists("model"):
        checkpoint = torch.load("model")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    best_cer_train = 1e9
    best_cer_test = 1e9

    for epoch in range(1000):
        print("> epoch", epoch, num_train_data, math.floor(time.time()-tm))
        sum_loss_train = 0
        sum_error_train = 0
        for batch_idx, (x, y_true) in enumerate(tqdm(dataloader_train)):
            x = torch.transpose(x, 1, 2)
            x = x.type(torch.FloatTensor)
            y_true = y_true.squeeze(-1)
            x, y_true = x.cuda(), y_true.cuda()
            logits = model(x).log_softmax(1)
            ctc_loss = torch.nn.CTCLoss().cuda()
            log_probs = torch.transpose(torch.transpose(
                logits, 0, 2), 1, 2).requires_grad_()
            targets = y_true
            input_lengths = torch.full((batch_size,), 680, dtype=torch.long)
            target_lengths = torch.tensor(
                [sum([1 for j in i if j > 0]) for i in y_true], dtype=torch.long)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss_train += loss.item()

            if batch_idx +1 < len(dataloader_train):
                continue
            y_pred = torch.argmax(logits, -2).cpu().numpy()
            y_pred = [[j for j in i if j > 0] for i in y_pred]
            y_true = y_true.cpu().numpy()
            y_true = [[j for j in i if j > 0] for i in y_true]
#             sum_error_train += np.average(list(editDistance(yp, yt) /
#                                                max(len(yp), len(yt)) for (yp, yt) in zip(y_pred, y_true)))

        loss_train = sum_loss_train / len(dataloader_train)
#         cer_train = sum_error_train / len(dataloader_train)

        sum_loss_test = 0
        sum_error_test = 0
        for batch_idx, (x, y_true) in enumerate(dataloader_test):
            x = torch.transpose(x, 1, 2)
            x = x.type(torch.FloatTensor)
            y_true = y_true.squeeze(-1)
            x, y_true = x.cuda(), y_true.cuda()
            logits = model(x).log_softmax(1)
            ctc_loss = torch.nn.CTCLoss().cuda()
            log_probs = torch.transpose(torch.transpose(
                logits, 0, 2), 1, 2).requires_grad_()
            targets = y_true
            input_lengths = torch.full((batch_size,), 680, dtype=torch.long)
            target_lengths = torch.tensor(
                [sum([1 for j in i if j > 0]) for i in y_true], dtype=torch.long)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            sum_loss_test += loss.item()

            y_pred = torch.argmax(logits, -2).cpu().numpy()
            y_pred = [[j for j in i if j > 0] for i in y_pred]
            y_true = y_true.cpu().numpy()
            y_true = [[j for j in i if j > 0] for i in y_true]
            sum_error_test += np.average(list(editDistance(yp, yt) /
                                         max(len(yp), len(yt)) for (yp, yt) in zip(y_pred, y_true)))

        loss_test = sum_loss_test / len(dataloader_test)
        cer_test = sum_error_test / len(dataloader_test)

        best_cer_test = min(best_cer_test, cer_test)

        print("train: ", loss_train)
        print("test:  ", loss_test, cer_test, "best", best_cer_test)

        if(cer_test < 0.5):
            printSeq(y_pred[0])
            printSeq(y_true[0])

        scheduler.step(loss_train)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, "model")
        
        if loss_test > loss_train * 1.1 and num_train_data < 12800:
            num_train_data = min(2*num_train_data, 12800)
            break

# max val rate: 63%
