###import the models###
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

###Specify a graphics card###
os.environ["CUDA_VISIBLE_DEVICES"]='0'

###To make sure that others could also use the Server.###
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True 
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

###fix the random seed, and insure it is the same as the one in 'Resnet181Train.py'###
np.random.seed(1)
allset = np.random.permutation(50000)
allset_train = allset[10000:50000]
allset_validation = allset[0:10000]

###to make a direction.###
def mkdir(fn): 
    if not os.path.isdir(fn):
        os.mkdir(fn)

###import the dataset cifar10###
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

###Normalize pixel values to be between 0 and 1###
train_images0, test_images0 = train_images / 255.0, test_images / 255.0
train_labels0=keras.utils.to_categorical(train_labels, num_classes = 10)

###reshape and make the dataset prepared###
train_images=train_images0[allset_train,:,:,:]
train_labels=train_labels0[allset_train,:]

###x is the input###
x = train_images

###this list is used for saving the -1 and -2 hidden layer###
out_Univ_tmp = []

###load the trained model, the model here is also the whole-training-process one.###
model=load_model('my_model_Resnet18-1_80.h5')

###reconstruct the model, and appoint the -1 and -2 hidden layers as output layer respectly###
layer_model = Model(inputs=model.input, outputs=model.layers[65].output)

###feature is the output###
feature=layer_model.predict(x)

###reshape feature into needed shape, for computing LFR###
feature = feature.reshape(40000,1024)

###save feature into out_Univ_tmp###
out_Univ_tmp.append(feature)

###the same above, and this is for -2 hidden layer.##
model=load_model('my_model_Resnet18-1_80.h5')
layer_model = Model(inputs=model.input, outputs=model.layers[66].output)
feature=layer_model.predict(x)
feature = feature.reshape(40000,1024)
out_Univ_tmp.append(feature)

###reshape and make the dataset prepared for computint LFR###
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images=train_images.reshape(50000,32*32*3)
test_images=test_images.reshape(10000,32*32*3)
train_images0, test_images0 = train_images / 255.0, test_images / 255.0
train_labels0=keras.utils.to_categorical(train_labels, num_classes = 10)
train_images=train_images0[allset_train,:]
train_labels=train_labels0[allset_train,:]

###create a ditionary to save data and variates.###
R={}

###the following lines are all about computing LFR###

R['train_inputs'] = train_images
R['y_true_train'] = train_labels

###i represents for differen delta###
for i in [0.02,0.03,0.04,0.05,0.06,0.08,0.09,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,22.0,24.0,26.0,28.0,30.0,35.0,40.0,45.0,50.0]:
    ###the delta###
    R['s_filter_wid'] = [i]
    
    ###the folder name, and also 80 stands for the training epochs###
    FolderName = r'my_model_Resnet18-1_80'+str(R['s_filter_wid'])+'/'
    R['FolderName'] = FolderName
    mkdir(FolderName)
    
    ###to save the parameters in the model.###
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
    
    ###to normalize every column of the -1 and -2 hidden layers.###
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
    
    ###from here start, the functions below are all for computing LFR###
    ###compute the distance between two matrixs###
    def compute_distances_no_loops(Y, X): 
        dists = -2 * np.dot(X, Y.T) + np.sum(Y**2,axis=1) + np.sum(X**2, axis=1)[:, np.newaxis] 
       
    ###to compute concrete x_i and x_j, based on the compute_distances_no_loops###
    def normal_kernel(diff_x2,filter_wid): 
        gau_x2=np.exp(-diff_x2/2/filter_wid) #compute x_i and x_j, and the G(delta).
        n_con=np.sum(gau_x2,axis=1,keepdims=True)
        n_gau_x2=gau_x2/n_con #normalization
        return n_gau_x2
    
    ###to ccompute the true y^{low,delta}###
    def gauss_filter_normalize2(f_orig,n_gau_x2): 
        f_new=np.matmul(n_gau_x2,f_orig) 
        return f_new
    
    ###to obtain the high frequency component and low frequency component. s_filter_wid is the filter width, which is the size of delta###
    def get_f_high_low(yy,xx,s_filter_wid,diff_x2=[]): 
        #t01=time.time()
        if len(diff_x2)==0: #if deff_x2 is empty, then xx=out.
            diff_x2=compute_distances_no_loops(xx,xx) 
        n_gau_x2_all=[]
        for filter_wid in s_filter_wid:
            n_gau_x2=normal_kernel(diff_x2,filter_wid) 
            n_gau_x2_all.append(n_gau_x2) 
        
        f_low=[] #the low frequency component
        f_high=[] #the high frequency componet
        for filter_wid_ind in range(len(s_filter_wid)):
            #f_new_norm=np.reshape(gauss_filter_normalize2(yy,n_gau_x2_all[filter_wid_ind]),[-1,10])
            f_new_norm=gauss_filter_normalize2(yy,n_gau_x2_all[filter_wid_ind]) #gain the low frequency component
            f_low.append(f_new_norm)
            f_high_tmp=yy-f_new_norm #gain the high frequency component
            f_high.append(f_high_tmp)
        
        return f_low, f_high
    
    ###compute one low frequency ratio###
    def low_fre_ratio_one(xx,yy,s_filter_wid,diff_x2=[]):
        #print(type(diff_x2))
        f_low, f_high=get_f_high_low(yy,xx,s_filter_wid,diff_x2) #
        syy=np.sum(np.square(yy)) 
        ratio=[]
        for f_ in f_low: #if we have several delta,then we will get several f_low
            sf=np.sum(np.square(f_))/syy #compute fo low frequency ratio(LFR)
            ratio.append(sf) 
        #print(np.shape(ratio))
        return ratio 
       
    ###compute all the LFR###
    def low_fre_ratio(output_all,y):
         ratio_all=[]
         ratio=low_fre_ratio_one(R['train_inputs'],R['y_true_train'],R['s_filter_wid'],diff_x2=dist_input) 
         ratio_all.append(ratio) 
         for out in output_all: 
             ratio=low_fre_ratio_one(out,R['y_true_train'],R['s_filter_wid'],diff_x2=[])
             ratio_all.append(ratio)
         return ratio_all 
     
    ###To here end, all above are for computing LFR###
    
    ###compute the distance of x and itself, for computing LFR###
    dist_input=compute_distances_no_loops(R['train_inputs'],R['train_inputs']) 
    
    ###to normalize the output###
    out_Univ_tmp=normalization_input(out_Univ_tmp)
    
    ###compute the LFR###
    ratio_tmp=low_fre_ratio(out_Univ_tmp,R['y_true_train']) 
    R['ratio_last'] = ratio_tmp
    print(ratio_tmp)
    
    ###save the results and relative parameters, and 'R['train_inputs'] = 0' is for saving space.###
    R['train_inputs'] = 0
    R['y_true_train'] = 0
    savefile()
    R['train_inputs'] = train_images
    R['y_true_train'] = train_labels
