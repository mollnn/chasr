import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import codecs
import pickle
from tqdm import tqdm

dict_size = 2884

batch_size = 16


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
        dataset_path = "./datasets/data_thchs30"
        self.mfcc_mat = np.load(os.path.join(
            dataset_path, "mfcc_vec_680x26.npy"))
        with codecs.open(os.path.join(dataset_path, "all_texts.txt"), encoding="utf-8") as file_read:
            text_lines = file_read.readlines()
        token_set = set(list(''.join(text_lines).replace("\n", "")))
        token_map = dict((j, i+1) for i, j in enumerate(token_set))
        seq_lines = [list(map(lambda x: token_map[x], text_line.replace(
            "\n", ""))) for text_line in text_lines]
        self.pad_lines = [(seq_line + [0]*48)[:48] for seq_line in seq_lines]
        self.pad_lines = torch.tensor(self.pad_lines).unsqueeze(-1)
        # 小数据测试
        self.mfcc_mat = self.mfcc_mat[:512]
        self.pad_lines = self.pad_lines[:512]

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


class ResSum(torch.nn.Module):
    def __init__(self, size, dim):
        super(ResSum, self).__init__()
        self.p1 = ResBlock(size, 1, dim)
        self.p2 = ResBlock(size, 2, dim)
        self.p3 = ResBlock(size, 4, dim)
        self.p4 = ResBlock(size, 8, dim)
        self.p5 = ResBlock(size, 16, dim)

    def forward(self, x):
        res = self.p1(x)
        res = res + self.p2(x)
        res = res + self.p3(x)
        res = res + self.p4(x)
        res = res + self.p5(x)
        return res


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.pre = torch.nn.Sequential(
            torch.nn.Conv1d(26, 192, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm1d(192),
            torch.nn.Tanh(),
        )
        self.m1 = ResSum(7, 192)
        self.m2 = ResSum(7, 192)
        self.m3 = ResSum(7, 192)
        self.m4 = ResSum(7, 192)
        self.m5 = ResSum(7, 192)
        self.post = torch.nn.Sequential(
            torch.nn.Conv1d(192, 192, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm1d(192),
            torch.nn.Tanh(),
            torch.nn.Conv1d(192, dict_size, kernel_size=1,
                            stride=1, padding=0),
        )

    def forward(self, x):
        x = self.pre(x)
        m = self.m1(x)
        m += self.m2(x)
        m += self.m3(x)
        m += self.m4(x)
        m += self.m5(x)
        m = self.post(m)
        return m


torch.cuda.empty_cache()
my_dataset = MyDataset()

# data_train, data_test = torch.utils.data.random_split(my_dataset, [13000, 388])
data_train, data_test = torch.utils.data.random_split(my_dataset, [
                                                      480, 512-480])

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.8, patience=1, verbose=True)
dataloader_train = torch.utils.data.DataLoader(
    data_train, batch_size=batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(
    data_test, batch_size=batch_size, shuffle=False)

for epoch in range(1000):
    print("> epoch", epoch)
    sum_loss_train = 0
    sum_error_train = 0
    for batch_idx, (x, y_true) in enumerate(tqdm(dataloader_train)):
        x = torch.transpose(x, 1, 2)
        x = x.type(torch.FloatTensor)
        y_true = y_true.squeeze(-1)
        x, y_true = x.cuda(), y_true.cuda()
        logits = model(x).log_softmax(2)
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

        if batch_idx < 1:
            y_pred = torch.argmax(logits, -2).cpu().numpy()
            y_pred = [[j for j in i if j > 0] for i in y_pred]
            y_true = y_true.cpu().numpy()
            y_true = [[j for j in i if j > 0] for i in y_true]
            sum_error_train += np.average(list(editDistance(yp, yt) /
                                               max(len(yp), len(yt)) for (yp, yt) in zip(y_pred, y_true)))
    loss_train = sum_loss_train / len(dataloader_train)
    cer_train = sum_error_train / min(1, len(dataloader_train))

    sum_loss_test = 0
    sum_error_test = 0
    for batch_idx, (x, y_true) in enumerate(dataloader_test):
        x = torch.transpose(x, 1, 2)
        x = x.type(torch.FloatTensor)
        y_true = y_true.squeeze(-1)
        x, y_true = x.cuda(), y_true.cuda()
        logits = model(x).log_softmax(2)
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
    print("train: ", loss_train, 1 - cer_train)
    print("test:  ", loss_test, 1 - cer_test)
    
    scheduler.step(loss_train)
