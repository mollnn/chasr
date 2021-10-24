import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import codecs
import pickle
from tqdm import tqdm

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        dataset_path = "./dataset/data_thchs30"
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

    def __len__(self):
        return len(self.mfcc_mat)

    def __getitem__(self, idx):
        return self.mfcc_mat[idx], self.pad_lines[idx]


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.stage = torch.nn.Sequential(
            torch.nn.Conv1d(26, 50, kernel_size=5, stride=1, padding=2),
            torch.nn.Tanh(),
            torch.nn.Conv1d(50, 2884, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.stage(x)
        return x


my_dataset = MyDataset()

data_train, data_test = torch.utils.data.random_split(my_dataset, [13000, 388])

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
batch_size = 4
dataloader_train = torch.utils.data.DataLoader(
    data_train, batch_size=batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(
    data_test, batch_size=batch_size, shuffle=False)

for epoch in range(10):
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
        loss.backward()
    print(loss.item())
