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
data_df_test = pd.DataFrame(columns=['path'])   
path="voicedata/output10.wav"  
data_df_test.loc[0] = [path]              

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