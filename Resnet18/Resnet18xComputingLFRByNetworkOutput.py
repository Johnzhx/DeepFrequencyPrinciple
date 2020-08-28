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

R={}
os.environ["CUDA_VISIBLE_DEVICES"]='0'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True 
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

np.random.seed(1)
allset = np.random.permutation(50000)
allset_train = allset[10000:50000]
allset_validation = allset[0:10000]
print(allset)


def mkdir(fn): #熟悉，做目录
    if not os.path.isdir(fn):
        os.mkdir(fn)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images0, test_images0 = train_images / 255.0, test_images / 255.0
train_labels0=keras.utils.to_categorical(train_labels, num_classes = 10)

train_images=train_images0[allset_train,:,:,:]
print(train_images.shape)
train_labels=train_labels0[allset_train,:]
print(train_labels.shape)

# 第一步：准备输入数据
x = train_images  #[1,28,28,1] 的形状

print(x[-1])

out_Univ_tmp = []
# 第二步：加载已经训练的模型

model=load_model('my_model_Resnet18-2_80.h5')
# 第三步：将模型作为一个层，输出第7层的输出
layer_model = Model(inputs=model.input, outputs=model.layers[45].output)
# 第四步：调用新建的“曾模型”的predict方法，得到模型的输出
feature=layer_model.predict(x)
print(np.shape(feature))
feature = feature.reshape(40000,1024)
#print(feature)
print(np.shape(feature))
out_Univ_tmp.append(feature)

model=load_model('my_model_Resnet18-2_80.h5')
# 第三步：将模型作为一个层，输出第7层的输出
layer_model = Model(inputs=model.input, outputs=model.layers[46].output)
# 第四步：调用新建的“曾模型”的predict方法，得到模型的输出
feature=layer_model.predict(x)
print(np.shape(feature))
feature = feature.reshape(40000,1024)
#print(feature)
print(np.shape(feature))
out_Univ_tmp.append(feature)

model=load_model('my_model_Resnet18-2_80.h5')
# 第三步：将模型作为一个层，输出第7层的输出
layer_model = Model(inputs=model.input, outputs=model.layers[47].output)
# 第四步：调用新建的“曾模型”的predict方法，得到模型的输出
feature=layer_model.predict(x)
print(np.shape(feature))
feature = feature.reshape(40000,10)
#print(feature)
print(np.shape(feature))
R['y_net_train'] = feature

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images=train_images.reshape(50000,32*32*3)
test_images=test_images.reshape(10000,32*32*3)
# Normalize pixel values to be between 0 and 1
train_images0, test_images0 = train_images / 255.0, test_images / 255.0
train_labels0=keras.utils.to_categorical(train_labels, num_classes = 10)


train_images=train_images0[allset_train,:]
train_labels=train_labels0[allset_train,:]


print(train_images.shape)
print(train_labels.shape)
    
print(train_images[-1])

R['train_inputs'] = train_images
R['y_true_train'] = R['y_net_train']

for i in [0.02,0.03,0.04,0.05,0.06,0.08,0.09,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,22.0,24.0,26.0,28.0,30.0,35.0,40.0,45.0,50.0]:
    R['s_filter_wid'] = [i]
    
    lenarg=np.shape(sys.argv)[0] #Sys.argv[ ]其实就是一个列表，里边的项为用户输入的参数，关键就是要明白这参数是从程序外部输入的，而非代码本身的什么地方，要想看到它的效果就应该将程序保存了，从外部来运行程序并给出参数。
    if lenarg>1:
        ilen=1
        while ilen<lenarg:
            if sys.argv[ilen]=='-m':
                R['num_unit']=np.int32(sys.argv[ilen+1])
            if sys.argv[ilen]=='-g':
                R['gpu']=[np.int32(sys.argv[ilen+1])] 
            if sys.argv[ilen]=='-lr':
                R['learning_rate']=np.float32(sys.argv[ilen+1])  
            if sys.argv[ilen]=='-seed':
                R['seed']=np.int32(sys.argv[ilen+1])
            if sys.argv[ilen]=='-step':
                R['Total_Step']=np.int32(sys.argv[ilen+1])
            if sys.argv[ilen]=='-inputd':
                R['input_dim']=np.int32(sys.argv[ilen+1])
            if sys.argv[ilen]=='-delta':
                R['s_filter_wid']=[np.float32(sys.argv[ilen+1])]
            if sys.argv[ilen]=='-tol':
                R['tol']=np.float32(sys.argv[ilen+1])
            if sys.argv[ilen]=='-act':
                R['ActFuc']=np.int32(sys.argv[ilen+1])
            if sys.argv[ilen]=='-layer':   
                R['hidden_units']=[num_unit]*np.int32(sys.argv[ilen+1])
            if sys.argv[ilen]=='-dir':
                sBaseDir=sys.argv[ilen+1]
            if sys.argv[ilen]=='-subfolder':
                R['subfolder']=sys.argv[ilen+1]
            ilen=ilen+2
    
    FolderName = r'my_model_Resnet18-2_80'+str(R['s_filter_wid'])+'/'
    R['FolderName'] = FolderName
    mkdir(FolderName)
    def savefile(): #保存模型参数的函数
        with open('%s/objs.pkl'%(FolderName), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(R, f, protocol=4)
        #序列化对象，将对象obj保存到文件file中去
        text_file = open("%s/Output.txt"%(FolderName), "w")
        for para in R:
            if np.size(R[para])>20:
                continue
            text_file.write('%s: %s\n'%(para,R[para]))
        
        for para in sys.argv: 
            text_file.write('%s  '%(para))
        text_file.close()
        #写到txt方便看
    
    def normalization_input(out_Univ_f): #（对每一列归一化）
        out_Univ_g=[]
        for i in out_Univ_f:
            num=np.mean(i,axis=0,keepdims=True)
            j = i - num
            ji = abs(j)
            #print(ji.shape)
            maxx = np.max(ji,axis=0,keepdims=True)
            #print(maxx.shape)
            ii = j/maxx
            #print(ii)
            out_Univ_g.append(ii)
        return(out_Univ_g)
    
    def normalization_input(out_Univ_f): #（对每一列归一化）
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
    
    def compute_distances_no_loops(Y, X): #求两个矩阵之间的距离，其中X,Y皆为矩阵，axis是对行求和，矩阵**2作用到每个元素上面。这里，X与Y的size是(train size,input dim)。即，每一行是一个input。
        dists = -2 * np.dot(X, Y.T) + np.sum(Y**2,axis=1) + np.sum(X**2, axis=1)[:, np.newaxis] #后两者最后变成了n行1列，每行都是一个input的二范数，而前者也是对input做内积，[:, np.newaxis]是用来增加维数的,这里使用过后，np.sum(X**2, axis=1)[:, np.newaxis]是按照列加到每一列的（可以当做，做了转至，然后广播加上去的），试验用文件有范例。
        return dists
        #即，对每个dists[i][j]=X的i行做内积+Y的第j行做内积-2X的第i行和Y的第j行做内积。！！！重要的！：就是（X的第i行-Y的第j行）^2，与G(delta)里的e的指数上的(xi-xj)^2相对应,也是矩阵（但每个位置已经做了初次变换了）
    def normal_kernel(diff_x2,filter_wid): #这个就是在compute_distances_no_loops的基础上，具体地算出xi,j
        gau_x2=np.exp(-diff_x2/2/filter_wid) #算出Xi,j，以及真正的G(delta)，算出来的就是G(delta)
        n_con=np.sum(gau_x2,axis=1,keepdims=True)#对行求和,但是维数不变（sum一般是会降一维的）
        n_gau_x2=gau_x2/n_con #这步就是归一化，就差乘以\vec{y}了;注意，在我的note里面y^{low,delta}是行向量，和老师的差一个转至
        return n_gau_x2
    
    def gauss_filter_normalize2(f_orig,n_gau_x2): #这个是真的算出y^{low,delta}的了
        f_new=np.matmul(n_gau_x2,f_orig) #算出y^{low,delta}
        return f_new
    
    
    def get_f_high_low(yy,xx,s_filter_wid,diff_x2=[]): #得到高频成分和低频成分；s_filter_wid是滤波宽度，就是note中的delta大小；diff_x2注意，在用的时候有的是有的——就是diff_x2=dist_input=compute_distances_no_loops(R['train_inputs'],R['train_inputs'])，没有的时候就是空的列表；
        #t01=time.time()
        if len(diff_x2)==0: #空的时候，就是xx=out，就是G(delta)里的e的指数上的(xi-xj)^2,xi.xj均是向量
            diff_x2=compute_distances_no_loops(xx,xx) #用于计算的就是以第倒数某层为input的h(x)的低频成分;得到的是整个未做变形前的G(delta)
        n_gau_x2_all=[]
        for filter_wid in s_filter_wid:
            n_gau_x2=normal_kernel(diff_x2,filter_wid) #得到归一化后的G(delta)l
            n_gau_x2_all.append(n_gau_x2) #记录下来
        
        f_low=[] #按照定义算出来的低频成分
        f_high=[] #按照定义算出来的高频成分，低频成分与高频成分此时均是矩阵
        for filter_wid_ind in range(len(s_filter_wid)):
            #f_new_norm=np.reshape(gauss_filter_normalize2(yy,n_gau_x2_all[filter_wid_ind]),[-1,10])
            f_new_norm=gauss_filter_normalize2(yy,n_gau_x2_all[filter_wid_ind]) #得到低频成分
            f_low.append(f_new_norm)
            f_high_tmp=yy-f_new_norm #原始的减去低频的，自然就是高频的
            f_high.append(f_high_tmp)
        
        return f_low, f_high #将低频和高频的返回，都是list，每个位置只是对应不同的delta罢了
    
    def low_fre_ratio_one(xx,yy,s_filter_wid,diff_x2=[]):
        #print(type(diff_x2))
        f_low, f_high=get_f_high_low(yy,xx,s_filter_wid,diff_x2) #获得低频和高频的成分
        syy=np.sum(np.square(yy)) #计算真实的y的矩阵范数，这是不变的，有意思的是
        ratio=[]
        for f_ in f_low: #如果有多个delta,就会得到多个f_low
            sf=np.sum(np.square(f_))/syy #计算低频比，即低频成分的矩阵范数除以真实的y的矩阵范数
            ratio.append(sf) 
        #print(np.shape(ratio))
        return ratio #返回的是一个list，但这个list可能又不止一个元素，因为有不止一个f_low
        
    def low_fre_ratio(output_all,y):
         ratio_all=[]
         ratio=low_fre_ratio_one(R['train_inputs'],R['y_true_train'],R['s_filter_wid'],diff_x2=dist_input) #这个是不变的，以真实的input和output计算出的
         ratio_all.append(ratio) 
         for out in output_all: #以此计算以每个隐藏层的神经元作为input，真实的y作为output的低频比
             ratio=low_fre_ratio_one(out,R['y_true_train'],R['s_filter_wid'],diff_x2=[])
             ratio_all.append(ratio)
         return ratio_all #如果有多个delta，那么应该是形如:[[1,2],[3,4]...[n,n+1]]，其中每组依次对应[delta1,delta2]；且依次对应，输入层，第一层，第二层...第n层
      
    dist_input=compute_distances_no_loops(R['train_inputs'],R['train_inputs']) #算x的‘距离’，进而算低频成分，这个是不会随着训练改变的
    
    out_Univ_tmp=normalization_input(out_Univ_tmp)
    
    #print(max(out_Univ_tmp[0][201]))
    
    ratio_tmp=low_fre_ratio(out_Univ_tmp,R['y_true_train']) #低频比，这个其实在note中比较清楚，计算以每个隐藏层作为输入的，此时的低频比是多少；注意的是，以真实值作为output是不会变的；那么，其实就会慢慢地收敛到真实的情况
    R['ratio_last'] = ratio_tmp
    print(ratio_tmp)
    
    R['train_inputs'] = 0
    R['y_true_train'] = 0
    
    savefile()
    
    R['train_inputs'] = train_images
    R['y_true_train'] = R['y_net_train']
