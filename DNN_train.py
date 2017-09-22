
#Restrict to one gpu
import imp
try:
	imp.find_module('setGPU')
	import setGPU
except ImportError:
	found = False
#/////////////////////

import matplotlib.pyplot as plt
import keras.backend as K
import pylab as P
import pandas as pd
import numpy as np
import keras.callbacks
import glob

from sklearn.metrics import roc_auc_score

#/////////////////TO BE MOVED INTO SEPARATE FILE FOR CLARITY
#ROC value to be printed out after epochs. Does not affect training
class ROC_value(keras.callbacks.Callback):
	def on_epoch_end(self, batch,logs={}):
		print ' - roc auc: ',round(roc_auc_score(y_test,self.model.predict(x_test)),3)

#Save losses etc. to a separate text file for plotting later
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
	file=open("Losses_"+loss_+".txt","w+")
	file.close()

    def on_epoch_end(self, batch, logs={}):

	string = str(round(logs.get('loss'),4))+" "+str(round(logs.get('val_loss'),4))+" "+str(round(logs.get('acc'),4))+" "+str(round(logs.get('val_acc'),4))+" "+str(round(roc_auc_score(y_test,self.model.predict(x_test)),3))+"\n"

	with open("Losses_"+loss_+".txt","a") as myfile:
		myfile.write(string)


from root_pandas import read_root

#Used loss defined here
loss_ = 'mean_squared_error' #'binary_crossentropy'

#Read in list of root files to be used in the training
listOfFiles=glob.glob("/work/hajohajo/DNNTrackReconstruction/trackingNtuples_TRAIN/*.root")

#List of features not to be used in the training
ignores=['trk_algoMask','vtx*','bsb*','simvtx*','simpv*','event','lumi','run','trk_mva','trk_isHP','trk_algo','trk_qualityMask','trk_nValid','trk_q','trk_vtxIdx','__array_index']

data=read_root(listOfFiles,ignore=ignores,flatten=True)
targets=data.trk_isTrue
#data_tr
train=data.sample(frac=0.95,random_state=200).drop(['trk_isTrue'],1)
#data_te
test=data.drop(train.index).drop(['trk_isTrue'],1)

#train = data_tr.drop(['trk_isTrue'],1)
#test = data_te.drop(['trk_isTrue'],1)

#print train.columns.values.tolist()

train_targets=targets[train.index]
test_targets=targets.drop(train.index)

#print train.shape
#print train_targets.shape
#print test.shape
#print test_targets.shape

from keras.models import Model
from keras.layers import Input,Dense,Convolution1D,Flatten,Dropout,Activation
import keras.backend as K
from sklearn.utils import class_weight

x_train=np.array(train.iloc[:])
y_train=np.array(train_targets.iloc[:])
x_test=np.array(test.iloc[:])
y_test=np.array(test_targets.iloc[:])

#Defining the network topology
dropoutRate=0.02
a_inp = Input(shape=(x_train.shape[1],))
a = Dense(300,activation='relu', kernel_initializer='normal')(a_inp)
a = Dropout(dropoutRate)(a)
a = Dense(150,activation='relu', kernel_initializer='normal')(a)
a = Dropout(dropoutRate)(a)
a = Dense(20,activation='relu', kernel_initializer='normal')(a)
a = Dropout(dropoutRate)(a)
a = Dense(100,activation='relu', kernel_initializer='normal')(a)
a = Dropout(dropoutRate)(a)
a = Dense(10,activation='relu', kernel_initializer='normal')(a)
a_out = Dense(1, activation='sigmoid', kernel_initializer='normal')(a)

model=Model(a_inp,a_out)
model.compile(loss=loss_,optimizer='Adam',metrics=['acc'])


cb=ROC_value()
loss=LossHistory()
check=keras.callbacks.ModelCheckpoint('KERAS_best_model_'+loss_+'.h5',monitor='val_loss',save_best_only=True)
class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train[:]),y_train[:])
Nepoch=80
batchS=512
model.fit(x_train,y_train,
	epochs=Nepoch,
	batch_size=batchS,
	class_weight=class_weight,
	callbacks=[cb,loss,check],
	validation_split=0.05)

model.save('my_model_'+loss_+'_Adam.h5')

