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
from root_pandas import read_root

from keras.models import Model,load_model
from keras.layers import Input,Dense,Convolution1D,Flatten,Dropout,Activation

listOfFiles=glob.glob("/work/hajohajo/DNNTrackReconstruction/trackingNtuples_TEST/*.root")
print listOfFiles


data_te=read_root(listOfFiles,'trackingNtuple/tree', ignore=['trk_algoMask','vtx*','bsb*','simvtx*','simpv*'],flatten=True)
test_targets=data_te.trk_isTrue

test = data_te.drop(['event','lumi','run','trk_mva','trk_isHP','trk_isTrue','trk_algo','trk_qualityMask','trk_nValid','trk_q','trk_vtxIdx','__array_index'],1)

print(test.columns.values.tolist())

x_test=np.array(test.iloc[:])
y_test=np.array(test_targets.iloc[:])


model=load_model('KERAS_best_model_mean_squared_error.h5') #binary_crossentropy.h5') #,custom_objects={'loss1':loss1})

pred=model.predict(x_test)

data_te['trk_mva_DNN'] = pred[:]
data_te.to_root('trackingNtuple_forPlotting.root',key='tree')
