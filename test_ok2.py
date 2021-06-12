import pandas as pd
import numpy as np
import os
import random
import sys
import glob 
import json
import keras
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import seaborn as sns
import scipy.io.wavfile
import tensorflow as tf
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import  History, ReduceLROnPlateau, CSVLogger
from keras.models import Model, Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.fftpack import fft # 離散傅立葉變換 - 返回實數或複數序列
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import load_model
data_test = pd.DataFrame(columns=['feature'])
input_duration=3
dir_list_test = os.listdir('voicedata/')
dir_list= os.listdir('data/')
print (dir_list)

#读取原始数据
data_df = pd.DataFrame(columns=['path', 'source', 'actor', 'gender',
                                'intensity', 'statement', 'repetition', 'emotion'])
data_df_test = pd.DataFrame(columns=['path','emotion'])   
path="voicedata/test.mp3"  
emotion=5  
data_df_test.loc[0] = [path, emotion]              
count = 0
for i in dir_list:
    file_list = os.listdir('data/' + i)
    for f in file_list:
        nm = f.split('.')[0].split('-')
        path = 'data/' + i + '/' + f
        src = int(nm[1])
        actor = int(nm[-1])
        emotion = int(nm[2])
        
        if int(actor)%2 == 0:
            gender = "female"
        else:
            gender = "male"
        
        if nm[3] == '01':
            intensity = 0
        else:
            intensity = 1
        
        if nm[4] == '01':
            statement = 0
        else:
            statement = 1
        
        if nm[5] == '01':
            repeat = 0
        else:
            repeat = 1
            
        data_df.loc[count] = [path, src, actor, gender, intensity, statement, repeat, emotion]
        count += 1
label2_list = []
for i in range(len(data_df)):
    if data_df.emotion[i] == 2: # Calm
        lb = "_positive"
    elif data_df.emotion[i] == 3: # Happy
        lb = "_positive"
    elif data_df.emotion[i] == 4: # Sad
        lb = "_negative"
    elif data_df.emotion[i] == 5: # Angry
        lb = "_negative"
    elif data_df.emotion[i] == 6: # Fearful
        lb = "_negative"
    else:
        lb = "_none" 
    # Add gender to the label    
    label2_list.append(data_df.gender[i] + lb)
    
data_df['label'] = label2_list
data2_df = data_df.copy()
data2_df = data2_df[data2_df.label != "male_none"]
data2_df = data2_df[data2_df.label != "female_none"].reset_index(drop=True)
data2_df = data2_df[data2_df.label != "female_neutral"]
data2_df = data2_df[data2_df.label != "female_happy"]
data2_df = data2_df[data2_df.label != "female_angry"]
data2_df = data2_df[data2_df.label != "female_sad"]
data2_df = data2_df[data2_df.label != "female_fearful"]
data2_df = data2_df[data2_df.label != "female_calm"]
data2_df = data2_df[data2_df.label != "female_positive"]
data2_df = data2_df[data2_df.label != "female_negative"].reset_index(drop=True)

tmp1 = data2_df[data2_df.actor == 21]

tmp3 = data2_df[data2_df.actor == 23]

data3_df = pd.concat([tmp1, tmp3],ignore_index=True).reset_index(drop=True)
"""
for i in tqdm(range(len(data3_df))):
    X, sample_rate = librosa.load(data3_df.path[i], res_type='kaiser_fast',duration=input_duration,sr=22050*2,offset=0.5)
#   X = X[10000:90000]
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    feature = mfccs
    data_test.loc[i] = [feature]
"""
#使用librosa提取语音信息，保存到 data_test中
X, sample_rate = librosa.load(data_df_test.path[0], res_type='kaiser_fast',duration=input_duration,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
feature = mfccs
data_test.loc[0] = [feature]
 
 
 
# evaluate 根據測試數據 test data 評估 加載的模型   loaded_model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# 加載 weights 到新模型中 

loaded_model.load_weights("model/voiceclass.h5")
print("Loaded model from disk")
 
# evaluate 根據測試數據 test data 評估 加載的模型  
opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  
test_valid = pd.DataFrame(data_test['feature'].values.tolist())
test_valid = np.array(test_valid)

lb = LabelEncoder()

test_valid = np.expand_dims(test_valid, axis=2)
print(test_valid)
preds = loaded_model.predict(test_valid, 
                         batch_size=16, 
                         verbose=1)
print(preds)
preds1=preds.argmax(axis=1)
print(preds1)                   