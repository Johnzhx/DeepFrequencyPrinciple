import tensorflow as tf
import pickle
import sys
import keras
import os
import numpy as np
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils.np_utils import *
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"]='0'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True 
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

np.random.seed(1)
allset = np.random.permutation(50000)
allset_train = allset[10000:50000]
allset_validation = allset[0:10000]

def mkdir(fn): 
    if not os.path.isdir(fn):
        os.mkdir(fn)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images0, test_images0 = train_images / 255.0, test_images / 255.0
train_labels0=keras.utils.to_categorical(train_labels, num_classes = 10)

train_images=train_images0[allset_train,:,:,:]
train_labels=train_labels0[allset_train,:]

x = train_images 

out_Univ_tmp = []

model=load_model('my_model_Resnet18-2_80.h5')
layer_model = Model(inputs=model.input, outputs=model.layers[45].output)
feature=layer_model.predict(x)
feature = feature.reshape(40000,1024)
out_Univ_tmp.append(feature)

model=load_model('my_model_Resnet18-2_80.h5')
layer_model = Model(inputs=model.input, outputs=model.layers[46].output)
feature=layer_model.predict(x)
feature = feature.reshape(40000,1024)
out_Univ_tmp.append(feature)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images=train_images.reshape(50000,32*32*3)
test_images=test_images.reshape(10000,32*32*3)
train_images0, test_images0 = train_images / 255.0, test_images / 255.0
train_labels0=keras.utils.to_categorical(train_labels, num_classes = 10)
test_labels0=keras.utils.to_categorical(test_labels, num_classes = 10)

train_images=train_images0[allset_train,:]
train_labels=train_labels0[allset_train,:]

R={}
R['train_inputs'] = train_images
R['y_true_train'] = train_labels

for i in [0.02,0.03,0.04,0.05,0.06,0.08,0.09,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,22.0,24.0,26.0,28.0,30.0,35.0,40.0,45.0,50.0]:
    R['s_filter_wid'] = [i]
        
    FolderName = r'my_model_Resnet18-2_80'+str(R['s_filter_wid'])+'/'
    R['FolderName'] = FolderName
    mkdir(FolderName)
    
    def savefile():
        with open('%s/objs.pkl'%(FolderName), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(R, f, protocol=4)
        text_file = open("%s/Output.txt"%(FolderName), "w")
        for para in R:
            if np.size(R[para])>20:
                continue
            text_file.write('%s: %s\n'%(para,R[para]))
        
        for para in sys.argv: 
            text_file.write('%s  '%(para))
        text_file.close()
    
    def normalization_input(out_Univ_f): 
        out_Univ_g=[]
        for i in out_Univ_f:
            num=np.mean(i,axis=0,keepdims=True)
            j = i - num
            ji = abs(j)
            #print(ji.shape)
            maxx = np.max(ji,axis=0,keepdims=True)
            #print(maxx)
            ii = j/(maxx+pow(10.0,-9))
            #print(ii)
            out_Univ_g.append(ii)
        return(out_Univ_g)
    
    def compute_distances_no_loops(Y, X): 
        dists = -2 * np.dot(X, Y.T) + np.sum(Y**2,axis=1) + np.sum(X**2, axis=1)[:, np.newaxis] 
        return dists
       
    def normal_kernel(diff_x2,filter_wid): 
        gau_x2=np.exp(-diff_x2/2/filter_wid)
        n_con=np.sum(gau_x2,axis=1,keepdims=True)
        n_gau_x2=gau_x2/n_con 
        return n_gau_x2
    
    def gauss_filter_normalize2(f_orig,n_gau_x2): 
        f_new=np.matmul(n_gau_x2,f_orig)
        return f_new
    
    def get_f_high_low(yy,xx,s_filter_wid,diff_x2=[]): 
        #t01=time.time()
        if len(diff_x2)==0: 
            diff_x2=compute_distances_no_loops(xx,xx)
        n_gau_x2_all=[]
        for filter_wid in s_filter_wid:
            n_gau_x2=normal_kernel(diff_x2,filter_wid) 
            n_gau_x2_all.append(n_gau_x2)
        
        f_low=[] 
        f_high=[] 
        for filter_wid_ind in range(len(s_filter_wid)):
            f_new_norm=gauss_filter_normalize2(yy,n_gau_x2_all[filter_wid_ind])
            f_low.append(f_new_norm)
            f_high_tmp=yy-f_new_norm 
            f_high.append(f_high_tmp)
        return f_low, f_high 
    
    def low_fre_ratio_one(xx,yy,s_filter_wid,diff_x2=[]):
        f_low, f_high=get_f_high_low(yy,xx,s_filter_wid,diff_x2) 
        syy=np.sum(np.square(yy)) 
        ratio=[]
        for f_ in f_low: 
            sf=np.sum(np.square(f_))/syy 
            ratio.append(sf) 
        return ratio 
        
    def low_fre_ratio(output_all,y):
         ratio_all=[]
         ratio=low_fre_ratio_one(R['train_inputs'],R['y_true_train'],R['s_filter_wid'],diff_x2=dist_input) 
         ratio_all.append(ratio) 
         for out in output_all: 
             ratio=low_fre_ratio_one(out,R['y_true_train'],R['s_filter_wid'],diff_x2=[])
             ratio_all.append(ratio)
         return ratio_all 
    
    dist_input=compute_distances_no_loops(R['train_inputs'],R['train_inputs']) 
    
    out_Univ_tmp=normalization_input(out_Univ_tmp)
    
    ratio_tmp=low_fre_ratio(out_Univ_tmp,R['y_true_train']) 
    
    R['ratio_last'] = ratio_tmp
      
    R['train_inputs'] = 0
    R['y_true_train'] = 0
    savefile()
    R['train_inputs'] = train_images
    R['y_true_train'] = train_labels
