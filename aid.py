# coding: utf-8
import argparse
import glob
import librosa
from tensorflow.keras.layers import BatchNormalization, Multiply, Add
from tensorflow.keras.layers import Conv1D
from tensorflow. keras.utils import multi_gpu_model
import tensorflow.keras.callbacks
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.layers import Reshape, Lambda
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.backend as K
import datetime
import re
import itertools
import platform
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from sklearn.metrics import accuracy_score
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import os
import codecs
import numpy as np
from os.path import join
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tensorflow.compat.v1.keras.backend.set_session(sess)
tensorflow.compat.v1.keras.backend.clear_session()  # 清理session

np.random.seed(2018)


def get_mfcc(wav_file, max_mfcc_len):

    y, sr = librosa.load(wav_file, mono=True)  # sr=22050,
    mfcc = librosa.feature.mfcc(y, sr)
    if max_mfcc_len > mfcc.shape[1]:
        mfcc_feature = np.pad(
            mfcc, ((0, 0), (0, max_mfcc_len-mfcc.shape[1])), 'constant')
    else:
        mfcc_feature = mfcc[:, :max_mfcc_len]
    return mfcc_feature


def create_mfcc_mat(wav_files, path='', save_name='mfcc_vec_all', max_mfcc_len=640):

    mfcc_mat = []
    for wav_file in tqdm(wav_files):
        mfcc_vec = get_mfcc(wav_file, max_mfcc_len)
        mfcc_mat.append(mfcc_vec)
    mfcc_mat = np.array(mfcc_mat).transpose(0, 2, 1)
    np.save(join(path, save_name), mfcc_mat)


def get_mfcc_mat(path='', save_name='mfcc_vec_all'):

    mfcc_mat = np.load(join(path, save_name+'.npy'))

    return mfcc_mat


def get_text(text_files):
    lines = []
    for text_file in tqdm(text_files):
        with codecs.open(text_file, encoding='utf-8') as f_read:
            line = f_read.readline()
            lines.append(line.strip().replace(" ", ""))
    return lines


def get_pad_seq(textlines, maxlen=48):

    # saving
    tok_path = join(path_base, 'tokenizer.pickle')
    if not os.path.exists(tok_path):
        tok = Tokenizer(char_level=True)
        tok.fit_on_texts(text_lines)
        with open(tok_path, 'wb') as handle:
            pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('create tok')
    # loading
    else:
        with open(tok_path, 'rb') as handle:
            tok = pickle.load(handle)
            print('load tok')

    seq_lines = tok.texts_to_sequences(text_lines[:])
    print('num of words,', len(tok.word_index.keys()))

    len_lines = pd.Series(map(lambda x: len(x), seq_lines))
    print('max_len', len_lines.max())

    def new_pad_seq(line, maxlen):
        return pad_sequences(line, maxlen=maxlen, padding='post', truncating='pre')

    lines = seq_lines[:]
    pad_lines = new_pad_seq(lines, maxlen)
    return pad_lines, tok

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 　选择使用的GPU
    path_base = './dataset/data_thchs30'
    path_data = join(path_base, 'data')

    K.set_learning_phase(1)  # set learning phase

    # step 1：创建mfcc特征矩阵，如果已经存在，则直接读取，否则重新创建
    if not os.path.exists(join(path_base, 'mfcc_vec_680x26'+'.npy')):
        wav_files = glob.glob(join(path_data, '*.wav'))
        wav_files.sort()
        print('num of wav files', len(wav_files), 'ready to create mfcc mat')
        create_mfcc_mat(wav_files[:], path=path_base)  # 第一次创建使用
    else:
        mfcc_mat = get_mfcc_mat(path=path_base, save_name='mfcc_vec_680x26')
        print('load from npy', mfcc_mat.shape)

    # step 2: 读取语音对应的文本，如果已经存在，则直接读取，否则重新创建
    text_path = join(path_base, 'all_texts.txt')
    if not os.path.exists(text_path):
        text_files = glob.glob(join(path_data, '*.wav.trn'))
        text_files.sort()
        print('num of trn files', len(text_files), 'ready to create text file')
        text_lines = get_text(text_files[:])
        with codecs.open(text_path, mode='w', encoding='utf-8') as f_write:
            for line in text_lines:
                f_write.write(line + '\n')
    else:
        text_lines = [] 
        with codecs.open(text_path, encoding='utf-8') as f_read:
            lines = f_read.readlines()
            for line in lines:
                text_lines.append(line.strip().replace(" ", ""))
        print('load from text file', len(text_lines))

    # step 3: 将文本转成与数字对应的映射关系，如果已经存在，则直接读取，否则重新创建
    # 不建议每次重新生成，避免字符与数字的对应关系前后不符
    pad_lines, tok = get_pad_seq(text_lines, maxlen=48)