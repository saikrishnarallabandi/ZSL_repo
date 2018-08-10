#from utils import *
import numpy as np
import random
from keras.utils import to_categorical

window = 5
num_classes = 3
input_dim = 66
hidden = 16


def get_max_len(arr):
   '''
   This takes a list of lists as input and returns the maximum length
   '''
   max_len = 0
   for a in arr:
     if len(a) > max_len:
          max_len = len(a)
   return max_len


# Process labels
labels_file = '/home2/srallaba/challenges/compare2018/data/ComParE2018_SelfAssessedAffect/lab/ComParE2018_SelfAssessedAffect.tsv'
labels = {}
ids = ['l','m','h']
f = open(labels_file)
cnt = 0 
for line in f:
  if cnt == 0:
    cnt+= 1
  else:
    line = line.split('\n')[0].split()
    fname = line[0].split('.')[0]
    lbl = ids.index(line[1])
    labels[fname] = lbl
    

# Process the dev
print("Processing Dev")
f = open('files.devel')
devel_input_array = []
devel_output_array = []
test_input_array = []
test_output_array = []
for line in f:
    line = line.split('\n')[0]
    input_file = '../feats/compare_selfassessed_world/' + line + '.ccoeffs_ascii'
    inp = np.loadtxt(input_file)
    if labels[line] == 1:
        test_input_array.append(inp)
        test_output_array.append(1)
    else:
        devel_input_array.append(inp)
        devel_output_array.append(labels[line])


x_dev = np.zeros( (len(devel_input_array), 401, input_dim), 'float32')
y_dev = np.zeros( (len(devel_input_array), num_classes ), 'float32')

for i, (x,y) in enumerate(zip(devel_input_array, devel_output_array)):
   x_dev[i] = x
   y_dev[i] = to_categorical(y,num_classes)   

x_test = np.zeros( (len(test_input_array), 401, input_dim), 'float32')
y_test = np.zeros( (len(test_input_array), num_classes ), 'float32')

for i, (x,y) in enumerate(zip(test_input_array, test_output_array)):
   x_test[i] = x
   y_test[i] = to_categorical(y,num_classes)   

print "The shape of x_test is ", x_test.shape

# Process the train
print("Processing Train")
f = open('files.train')
train_input_array = []
train_output_array = []
for line in f:
    line = line.split('\n')[0]
    input_file = '../feats/compare_selfassessed_world/' + line + '.ccoeffs_ascii'
    inp = np.loadtxt(input_file)
    train_input_array.append(inp)
    train_output_array.append(labels[line])

x_train = np.zeros( (len(train_input_array), 401, input_dim), 'float32')
y_train = np.zeros( (len(train_input_array), num_classes ), 'float32')

for i, (x,y) in enumerate(zip(train_input_array, train_output_array)):
   x_train[i] = x
   y_train[i] = to_categorical(y,num_classes)




import keras
from sklearn import preprocessing
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, AlphaDropout
from keras.callbacks import *
import pickle, logging
from keras import regularizers
import time, random
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.callbacks import *
import pickle, logging
from sklearn.metrics import confusion_matrix

dim_w2v = 50

def custom_kernel_init(shape):
    vectors             = np.loadtxt('t', dtype=np.float)
    vectors             = vectors.T
    return vectors

global model
model = Sequential()
model.add(LSTM(hidden, return_sequences=True, input_shape=(401, input_dim)))
model.add(LSTM(hidden, return_sequences=True))
model.add(LSTM(hidden))
model.add(Dense(dim_w2v, activation='relu')) 
model.add(Dense(num_classes, activation='softmax', trainable=False,kernel_initializer=custom_kernel_init))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=4, epochs=10, shuffle=True, validation_data=(x_dev,y_dev))

print "Predicting for mid"
y_hat = model.predict(x_test)
y_true = []
y_pred = []
for (a,b) in zip(y_test,y_hat):
     label_test  = np.argmax(a)
     label_pred = np.argmax(b)
     print label_test, label_pred
     y_true.append(label_test)
     y_pred.append(label_pred)
print confusion_matrix(y_true, y_pred)   

