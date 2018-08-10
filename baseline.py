import os, sys
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import soundfile as sf
#from logger import Logger
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time
from keras.utils import to_categorical


batch_size = 4
num_classes = 3
print_flag = 0

# Process labels
labels_file = '/home1/srallaba/projects/tts/zeroshot_learning/data/compare2018/ComParE2018_SelfAssessedAffect/lab/ComParE2018_SelfAssessedAffect.tsv'
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
        test_output_array.append(to_categorical(1,num_classes) )
        #test_output_array.append(1)
    else:
        devel_input_array.append(inp)
        devel_output_array.append(to_categorical(labels[line], num_classes))
        #devel_output_array.append(labels[line])

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
    train_output_array.append(to_categorical(labels[line], num_classes))
    #train_output_array.append(labels[line])


class compare_dataset(Dataset):
      def __init__(self, ccoeffs_array, labels_array):
        self.ccoeffs_array = ccoeffs_array
        self.labels_array = labels_array

      def __getitem__(self, index):
           return self.ccoeffs_array[index], self.labels_array[index]


      def __len__(self):
           return len(self.ccoeffs_array) 

wks_train = compare_dataset(train_input_array, train_output_array)
train_loader = DataLoader(wks_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4
                         )
wks_devel = compare_dataset(devel_input_array, devel_output_array)
devel_loader = DataLoader(wks_devel,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4
                         )
wks_test = compare_dataset(test_input_array, test_output_array)
test_loader = DataLoader(wks_test,
                          batch_size=1,
                          shuffle=False,
                          num_workers=4
                         )


class Model_LstmFc(torch.nn.Module):

    def __init__(self):
        super(Model_LstmFc, self).__init__()
        self.rnn = nn.LSTM(66, 256, batch_first=True, bidirectional=True) # input_dim, hidden_dim
        self.fc1 = nn.Linear(512, 50) # https://stackoverflow.com/questions/45022734/understanding-a-simple-lstm-pytorch
        self.fc2 = nn.Linear(50, 3)
        self.softmax = nn.LogSoftmax()

    def set_weights(custom_weights):
        with torch.no_grad():
            self.fc2.weight = custom_weights 

    def forward(self, x):
        if print_flag:
          print("Shape input to RNN is ", x.shape) # [1,401,66]
        out, (h, cell) = self.rnn(x) # (batch_size, seq, input_size)
        h_hat = h.view( h.shape[1], h.shape[0]*h.shape[2])
        if print_flag :
          print("Shape output from  RNN is ", x.shape) # [1, 401, 512]
          print("Shape of hidden from RNN is ", h.shape)
          print("Shape of RNN cell: ", cell.shape)
          print("Modified shape of RNN hidden: ", h_hat.shape)
        x = self.fc2(F.relu(self.fc1(h_hat)))
        return F.log_softmax(x, dim=-1)



net = Model_LstmFc()
net.double()
if torch.cuda.is_available():
   net.cuda()
loss_func = F.nll_loss
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

def train():
  for epoch in range(10):
    net.train()
    l = 0
    for i, (a,b) in enumerate(train_loader):
       if torch.cuda.is_available():
          A = Variable(a).cuda()
          B = Variable(b).cuda()
       else:
          A = Variable(a)
          B = Variable(b)
       B_hat = net(A) 

       loss = loss_func(B_hat, torch.max(B, 1)[1] )
       l += loss.item()

       optimizer.zero_grad()
       loss.backward()       
       optimizer.step()  

       if i % 20 == 1:
          print ("   Loss after " , i, " batches is ", l/(i + 1))
    print ( "Loss after " , epoch, " epochs: ", l/(i + 1)) 
    devel(epoch)


def devel(epoch):
  net.eval()
  l = 0
  for i, (a,b) in enumerate(devel_loader):
      if torch.cuda.is_available():
        A = Variable(a).cuda()
        B = Variable(b).cuda()
      else:
        A = Variable(a)
        B = Variable(b)
      B_hat = net(A) 

      loss = loss_func(B_hat, torch.max(B, 1)[1] )
      l += loss.item()

  print ( "Dev Loss after " , epoch, " epochs: ", l/(i + 1)) 
  print ( '\n')
train()
