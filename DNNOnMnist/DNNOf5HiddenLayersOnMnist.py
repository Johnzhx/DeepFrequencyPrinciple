###import the modules###
import keras
import os,sys
import matplotlib
matplotlib.use('Agg')   
import pickle
import time  
import shutil 
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt   
#from BasicFunc import mySaveFig, mkdir
import platform
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
tf.disable_eager_execution()
np.set_printoptions(precision=10)

###to adjust the position of the picture###
Leftp=0.18
Bottomp=0.18
Widthp=0.88-Leftp
Heightp=0.9-Bottomp
pos=[Leftp,Bottomp,Widthp,Heightp]

###import the dataset, and make the data prepared for the experiment###
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data(path='mnist.npz')
x_train=x_train.reshape(60000,28*28)
x_test=x_test.reshape(10000,28*28)
y_train=y_train.reshape(60000,1)
y_test=y_test.reshape(10000,1)
x_train=x_train[0:30000,:]
x_test=x_test[0:10000,:]
y_train=y_train[0:30000,:]
y_test=y_test[0:10000,:]
x_train=x_train / 255
x_test=x_test / 255
y_train=y_train
y_test=y_test
y_train=keras.utils.to_categorical(y_train, num_classes = 10)
y_test=keras.utils.to_categorical(y_test, num_classes = 10)

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

###to make a direction.###
def mkdir(fn): 
    if not os.path.isdir(fn):
        os.mkdir(fn)
        
###to save the figures###        
def mySaveFig(pltm, fntmp,fp=0,ax=0,isax=0,iseps=0,isShowPic=0): 
    if isax==1:
        #pltm.legend(fontsize=18)
        # plt.title(y_name,fontsize=14)
#        ax.set_xlabel('step',fontsize=18)
#        ax.set_ylabel('loss',fontsize=18)
        pltm.rc('xtick',labelsize=18)
        pltm.rc('ytick',labelsize=18)
        ax.set_position(pos, which='both')
    fnm='%s.png'%(fntmp)
    pltm.savefig(fnm)
    if iseps:
        fnm='%s.eps'%(fntmp)
        pltm.savefig(fnm, format='eps', dpi=600)
    if fp!=0:
        fp.savefig("%s.pdf"%(fntmp), bbox_inches='tight')
    if isShowPic==1:
        pltm.show() 
    elif isShowPic==-1:
        return
    if isax==1:
        pltm.close()

###from here start, the functions below are all for computing LFR###
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
###To here end, it is exactly the same as the one in Resnet18###

###create a ditionary to save data and variates.###
R={} 

R['test_inputs']=x_test
R['train_inputs']=x_train
R['y_true_test']=  y_test 
R['y_true_train']= y_train 

###normalize the ###
R['train_inputs_nl']=normalization_input([R['train_inputs']])
###compute the distance of x and itself, for computing LFR###
dist_input=compute_distances_no_loops(R['train_inputs_nl'][0],R['train_inputs_nl'][0])



 ### used for saved all parameters and data 保存所有参数

for i in [10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30]:
    for j in [3,4,5,6,7]:
        R['s_filter_wid']=[i] #delta的取值，这里自然可以取多个
        ### mkdir a folder to save all output
        sBaseDir = 'fitnd1_delta'+str(R['s_filter_wid'])+'/'
        if platform.system()=='Windows':
            device_n="0"
            BaseDir = r'D:\跟着许志钦搞科研\验证低频比问题\deepstudy20200606/%s'%(sBaseDir)
        else:
            device_n="0"
            BaseDir = sBaseDir
            matplotlib.use('Agg')
            
        R['issave']=0 #是否保存模型参数
        #BaseDir = 'fitnd/'
        R['normalization']=1  #是否归一化
        #subFolderName = '%s'%(int(np.absolute(np.random.normal([1])*100000))//int(1)) 
        subFolderName = '%s'%(datetime.now().strftime("%y%m%d%H%M%S")) 
        FolderName = '%s%s/'%(BaseDir,subFolderName)
        mkdir(BaseDir)
        mkdir(FolderName) #至此，文件目录做好了
        if R['issave']: #是否保存model，具体的模型参数
            print('save')
            mkdir('%smodel/'%(FolderName))
        R['FolderName']=FolderName #文件名记录一下
        
        if  True: #not platform.system()=='Windows':
            shutil.copy(__file__,'%s%s'%(FolderName,os.path.basename(__file__)))
            
        R['input_dim']=28*28 #输入维数
        R['output_dim']=10 #按照上面的函数定义，结果是一维的
        R['epsion']=0.1 #整个代码就这里出现了...
        
        R['ActFuc']=1   ###  0: ReLU; 1: Tanh; 3:sin;4: x**5,, 5: sigmoid  6 sigmoid derivate 激活函数
        
        #R['hidden_units']=[1500,1500,1500]
        #R['hidden_units']=[500,500,500]
        num_unit=500
        R['hidden_units']=[num_unit]*j #自然就是隐藏层了
        R['fake_hidden_units']=[num_unit]*20 #虚假的隐藏层？
        #R['hidden_units']=[500,500]
        #R['hidden_units']=[500,1500]
        #R['hidden_units']=[500]
        ### initialization standard deviation
        R['astddev']=np.sqrt(1/num_unit) # for weight
        R['bstddev']=np.sqrt(1/num_unit)# for bias terms2 #用于初始化参数的
        
        R['ASI']=0
        R['learning_rate']=5e-5 #学习速率
        R['learning_rateDecay']=2e-5 #学习速率衰减率
        
        ### setup for activation function
        R['seed']=1 #随机种子
        np.random.seed(R['seed']) #固定随机种子
        R['gpu']='0'
        os.environ["CUDA_VISIBLE_DEVICES"]='%s'%(R['gpu']) 
            
        
        R['plotepoch']=100 #每100次，画一次
        R['train_size']=10000;  ### training size #训练集的大小，就是那个函数的采点数
        R['batch_size']=R['train_size'] # int(np.floor(R['train_size'])) ### batch size
        R['test_size']=5000  ### test size
        R['x_start']=-np.pi/2  #math.pi*3 ### start point of input #函数从-pi/2开始
        R['x_end']=np.pi/2  #6.28/4 #math.pi*3  ### end point of input #到pi/2结束
        R['c_amp']=1 #最后一偏移项的倍数，取0就是没有偏移项
        
        R['tol']=1e-2 #停止时的训练集上的误差
        R['Total_Step']=600000  ### the training step. Set a big number, if it converges, can manually stop training 
        R['isresnet']=0
        R['FolderName']=FolderName   ### folder for save images
        
       
        '''
        R['y_true_test']= R['y_true_test']- np.sum(R['y_true_test'],axis=0)/R['test_size']
        R['y_true_train']=R['y_true_train']- np.sum(R['y_true_train'],axis=0)/R['train_size']
        '''
        
        print(np.sum(R['y_true_train'],axis=0,keepdims=True))
        
        R['gpu']=device_n
        
        lenarg=np.shape(sys.argv)[0] #Sys.argv[ ]其实就是一个列表，里边的项为用户输入的参数，关键就是要明白这参数是从程序外部输入的，而非代码本身的什么地方，要想看到它的效果就应该将程序保存了，从外部来运行程序并给出参数。
        if lenarg>1:
            ilen=1
            while ilen<lenarg:
                if sys.argv[ilen]=='-m':
                    R['num_unit']=np.int32(sys.argv[ilen+1])
                if sys.argv[ilen]=='-g':
                    R['gpu']=np.int32(sys.argv[ilen+1]) 
                if sys.argv[ilen]=='-lr':
                    R['learning_rate']=np.float32(sys.argv[ilen+1])  
                if sys.argv[ilen]=='-seed':
                    R['seed']=np.int32(sys.argv[ilen+1])
                if sys.argv[ilen]=='-step':
                    R['Total_Step']=np.int32(sys.argv[ilen+1])
                if sys.argv[ilen]=='-tol':
                    R['tol']=np.float32(sys.argv[ilen+1])
                if sys.argv[ilen]=='-act':
                    R['ActFuc']=np.int32(sys.argv[ilen+1])
                if sys.argv[ilen]=='-layer':   
                    R['hidden_units']=[num_unit]*np.int32(sys.argv[ilen+1])
                ilen=ilen+2
                
        #以上的东西不影响代码，暂时不关注了        
        os.environ["CUDA_VISIBLE_DEVICES"]='%s'%(R['gpu']) #设置用哪个GPU
        #train_inputs_normalnization=normalization_input([R['train_inputs']])
        
        t0=time.time() 
        
        def add_layer2(x,input_dim = 1,output_dim = 1,isresnet=1,astddev=0.05, #添加一个隐藏层的程序
                       bstddev=0.05,ActFuc=0,seed=0, name_scope='hidden'):
            if not seed==0: #固定随机种子
                tf.set_random_seed(seed)
            
            with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
                ua_w = tf.get_variable(name='ua_w', initializer=astddev)
                ua_b = tf.get_variable(name='ua_b', initializer=bstddev) 
                z=tf.matmul(x, ua_w) + ua_b
                
                
                if ActFuc==1:
                    output_z = tf.nn.tanh(z)
                    print('tanh')
                elif ActFuc==3:
                    output_z = tf.sin(z)
                    print('sin')
                elif ActFuc==0:
                    output_z = tf.nn.relu(z)
                    print('relu')
                elif ActFuc==4:
                    output_z = z**50
                    print('z**50')
                elif ActFuc==5:
                    output_z = tf.nn.sigmoid(z)
                    print('sigmoid')
                    
                L2Wight= tf.nn.l2_loss(ua_w) 
                if isresnet and input_dim==output_dim:
                    output_z=output_z+x
                return output_z,ua_w,ua_b,L2Wight
        
        def getWini(hidden_units=[10,20,40],input_dim = 1,output_dim_final = 1,astddev=0.05,bstddev=0.05): #这个就是生产初始化参数，构建所有的从input到output的参数了，老样子，先把所有参数生成好，再带入到等会的初始化当中
            
            hidden_num = len(hidden_units)
            #print(hidden_num)
            add_hidden = [input_dim] + hidden_units;
            
            w_Univ0=[]
            b_Univ0=[]
            
            for i in range(hidden_num):
                input_dim = add_hidden[i]
                output_dim = add_hidden[i+1]
                ua_w=np.float32(np.random.normal(loc=0.0,scale=(2/(input_dim+output_dim))**0.5,size=[input_dim,output_dim]))
                ua_b=np.float32(np.random.normal(loc=0.0,scale=0,size=[1,output_dim]))
                w_Univ0.append(ua_w)
                b_Univ0.append(ua_b)
            ua_w=np.float32(np.random.normal(loc=0.0,scale=(2/(hidden_units[hidden_num-1]+ output_dim_final))**0.5,size=[hidden_units[hidden_num-1], output_dim_final]))
            ua_b=np.float32(np.random.normal(loc=0.0,scale=0,size=[1,output_dim_final]))
            w_Univ0.append(ua_w)
            b_Univ0.append(ua_b)
            return w_Univ0, b_Univ0
        
        w_Univ0, b_Univ0=getWini(hidden_units=R['fake_hidden_units'], #生产虚假的神经网络初始化参数
                                                 input_dim = R['input_dim'],
                                                 output_dim_final = R['output_dim'])
        print(w_Univ0[1].shape)
        def univAprox2(x0, hidden_units=[10,20,40],input_dim = 1,output_dim_final = 1, #生成神经网络框架
                       isresnet=1,astddev=0.05,bstddev=0.05,
                       ActFuc=0,seed=0,ASI=1):
            if seed==0:
                seed=time.time()
            # The simple case is f: R -> R 
            hidden_num = len(hidden_units)
            #print(hidden_num)
            add_hidden = [input_dim] + hidden_units;
            
            w_Univ=[]
            b_Univ=[] 
            out_Univ=[] 
             
            
        
            output=x0
            
            
            for i in range(hidden_num):
                input_dim = add_hidden[i]
                output_dim = add_hidden[i+1]
                print('input_dim:%s, output_dim:%s'%(input_dim,output_dim))
                name_scope = 'hidden' + np.str(i+1)
                    
                output,ua_w,ua_b,L2Wight_tmp=add_layer2(output,input_dim,output_dim,isresnet=isresnet,
                                                       astddev=w_Univ0[i],bstddev=b_Univ0[i], ActFuc=ActFuc,
                                                       seed=seed, name_scope= name_scope)
                w_Univ.append(ua_w)
                b_Univ.append(ua_b)
                out_Univ.append(output) 
            
            ua_we = tf.get_variable(
                    name='ua_we'
                    #, shape=[hidden_units[hidden_num-1], output_dim_final]
                    , initializer=w_Univ0[-1]
                )
            ua_be = tf.get_variable(
                    name='ua_be'
                    #, shape=[1,output_dim_final]
                    , initializer=b_Univ0[-1]
                )
            
            z1 = tf.matmul(output, ua_we)+ua_be
            w_Univ.append(ua_we)
            b_Univ.append(ua_be)
            
            # you can ignore this trick for now. Consider ASI=False
            if ASI:
                output=x0
                for i in range(hidden_num):
                    input_dim = add_hidden[i]
                    output_dim = add_hidden[i+1]
                    print('input_dim:%s, output_dim:%s'%(input_dim,output_dim))
                    name_scope = 'hidden' + np.str(i+1+hidden_num)
                    output,ua_w,ua_b,L2Wight_tmp=add_layer2(output,input_dim,output_dim,isresnet=isresnet,
                                                       astddev=w_Univ0[i],bstddev=b_Univ0[i], ActFuc=ActFuc,
                                                       seed=seed, name_scope= name_scope)
                ua_we = tf.get_variable(
                        name='ua_wei2'
                        #, shape=[hidden_units[hidden_num-1], output_dim_final]
                        , initializer=-w_Univ0[-1]
                    )
                ua_be = tf.get_variable(
                        name='ua_bei2'
                        #, shape=[1,output_dim_final]
                        , initializer=-b_Univ0[-1]
                    )
                z2 = tf.matmul(output, ua_we)+ua_be
            else:
                z2=0
            z=z1+z2
            return z,w_Univ,b_Univ,out_Univ
        
        tf.reset_default_graph() #开始构建神经网络了
            
        with tf.variable_scope('Graph',reuse=tf.AUTO_REUSE) as scope:
            # Our inputs will be a batch of values taken by our functions
            x = tf.placeholder(tf.float32, shape=[None, R['input_dim']], name="x")
            
            
            y_true = tf.placeholder_with_default(input=[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]], shape=[None, R['output_dim']], name="y")
            in_learning_rate= tf.placeholder_with_default(input=1e-3,shape=[],name='lr')
            y,w_Univ,b_Univ,out_Univ = univAprox2(x, R['hidden_units'],input_dim = R['input_dim'],output_dim_final = R['output_dim'],
                                                    astddev=R['astddev'],bstddev=R['bstddev'],
                                                    ActFuc=R['ActFuc'],
                                                    isresnet=R['isresnet'],
                                                    seed=R['seed'],ASI=R['ASI'])
            
            loss=tf.reduce_mean(tf.square(y_true-y)) #求随损失函数
            # We define our train operation using the Adam optimizer
            adam = tf.train.AdamOptimizer(learning_rate=in_learning_rate)
            train_op = adam.minimize(loss)
        
        
        config = tf.ConfigProto(allow_soft_placement=True) #以下是用来指派设备的
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer()) #初始化参数，没关系下面又初始化了一次 注意，这里仅仅只是初始化了，并没有真正的跑过，如果真的run，是会出问题的；本人改了代码试验过，是不行的，（注意y_true是一维的，因此可以光波到y的维数。）
        saver = tf.train.Saver()   #保存模型的函数
        R['w_dis']=[]
        R['b_dis']=[] 
        R2={}
        R2['w_Univ_ini']=[] #最一开始的W 以下均是测试集上的，但其实跑测试集时并不改变参数
        R2['b_Univ_ini']=[] #最一开始的b
        R2['w_Univ_end']=[] #结束时的w
        R2['b_Univ_end']=[] #结束时的b
        R['loss_epoch']=[]
        class model():
            def __init__(self): 
                R['y_train']=[] #最新的训练集上的跑出来的函数结果y
                R['y_test']=[] #最新的测试集上的跑出来的函数结果y
                R['loss_test']=[] #记录测试集损失函数变化的list，每步都有
                R['loss_train']=[] #记录训练集损失函数变化的list
                R['ratio']=[]
                if R['issave']: #保存下模型以及模型参数
                    nametmp='%smodel/'%(FolderName)
                    mkdir(nametmp)
                    saver.save(sess, "%smodel.ckpt"%(nametmp))
                y_test, loss_test_tmp,w_Univ_tmp,b_Univ_tmp= sess.run([y,loss,w_Univ,b_Univ],feed_dict={x: R['test_inputs'], y_true: R['y_true_test']})
                #看看测试集
                y_train,loss_train_tmp= sess.run([y,loss],feed_dict={x: R['train_inputs'], y_true: R['y_true_train']})
                #再看看训练集的
                R['y_train']=y_train
                R['y_test']=y_test
                R['loss_test'].append(loss_test_tmp)
                R['loss_train'].append(loss_train_tmp)
                R2['w_Univ_ini']=w_Univ_tmp
                R2['b_Univ_ini']=b_Univ_tmp
                R['w_dis']=[]
                R['b_dis']=[]
                for tmp in R2['w_Univ_ini']: #先把dis的list创建好，有几个wi，创建几个
                    R['w_dis'].append([])
                    R['b_dis'].append([])
                self.ploty(name='ini') 
            def run_onestep(self): #跑一步
                y_test, loss_test_tmp,w_Univ_tmp,b_Univ_tmp = sess.run([y,loss,w_Univ,b_Univ],feed_dict={x: R['test_inputs'], y_true: R['y_true_test']})
                R2['w_Univ_end'],R2['b_Univ_end']=w_Univ_tmp,b_Univ_tmp
                y_train,loss_train_tmp= sess.run([y,loss],feed_dict={x: R['train_inputs'], y_true: R['y_true_train']}) #看看训练集的情况
                    
                if R['train_size']>R['batch_size']: #分批次训练，还打乱顺序哟
                    indperm = np.random.permutation(R['train_size'])
                    nrun_epoch=np.int32(R['train_size']/R['batch_size'])
                    
                    for ijn in range(nrun_epoch):
                        ind = indperm[ijn*R['batch_size']:(ijn+1)*R['batch_size']] 
                        _= sess.run(train_op, feed_dict={x: R['train_inputs'][ind], y_true: R['y_true_train'][ind],
                                                          in_learning_rate:R['learning_rate']})
                else: #优化一次
                    _ = sess.run(train_op, feed_dict={x: R['train_inputs'], y_true: R['y_true_train'],
                                                          in_learning_rate:R['learning_rate']})
                R['learning_rate']=R['learning_rate']*(1-R['learning_rateDecay']) #学习速率递减
                    
                R['y_train']=y_train
                R['y_test']=y_test
                R['loss_test'].append(loss_test_tmp)
                R['loss_train'].append(loss_train_tmp)
            def run(self,step_n=1): #开始真正地跑了
                if R['issave']: #把保存的模型，恢复出来，继续训练
                    nametmp='%smodel/model.ckpt'%(FolderName)
                    saver.restore(sess, nametmp)
                for ii in range(step_n):
                    self.run_onestep()
                    if R['loss_train'][-1]<R['tol']: #如果最新的训练误差小于阈值就结束
                        print('model end, error:%s'%(R['loss_train'][-1]))
                        self.plotloss()
                        self.ploty()
                        self.savefile()
                        R['step']=ii
                        if R['issave']: #保存模型
                            nametmp='%smodel/'%(FolderName)
                            shutil.rmtree(nametmp) #shutil.rmtree() 表示递归删除文件夹下的所有子文件夹和子文件
                            saver.save(sess, "%smodel.ckpt"%(nametmp))
                        break
                    if ii==0:
                        print('initial %s'%(np.max(R['y_train'])))
                        
                    if ii%R['plotepoch']==0: #打印一下
                        losss,y_train,out_Univ_tmp= sess.run([loss,y,out_Univ],feed_dict={x: R['train_inputs'], y_true: R['y_true_train']}) #每个隐藏层层和输出层的此时神经元的值
                        #print(out_Univ_tmp[-2][-1][-1])
                        if R['normalization']==1 :
                            out_Univ_tmp=normalization_input(out_Univ_tmp)
                        #print(out_Univ_tmp)
                        #print(out_Univ_tmp[-2][-1][-1])
                        ratio_tmp=self.low_fre_ratio(out_Univ_tmp,y_train) #低频比，这个其实在note中比较清楚，计算以每个隐藏层作为输入的，此时的低频比是多少；注意的是，以真实值作为output是不会变的；那么，其实就会慢慢地收敛到真实的情况
                        R['ratio'].append(np.squeeze(ratio_tmp)) #把ratio_tmp从list变成向量了
                        R['ratio_last']=R['ratio'][-1]
                        R['loss_epoch'].append(losss)
                        R['step']=ii
                        print('len:%s, ii:%s'%(len(R['ratio']),ii)) #打印一下 R['ratio']的长度和现在的步数
                        print('time elapse: %.3f'%(time.time()-t0)) #打印一下时间
                        print('model, epoch: %d, test loss: %f' % (ii,R['loss_test'][-1])) #打印一下测试集的损失函数值
                        print('model, epoch: %d, train loss: %f' % (ii,R['loss_train'][-1])) #打印一下训练集的损失函数值
                        print('ratio:%s'%(ratio_tmp)) #打印一下当前的低频比值
        
                        x_train_label = sess.run(y,feed_dict={x: R['train_inputs'], y_true: R['y_true_train']})
                        x_train_label_location=np.argmax(x_train_label,axis=1)
                        y_train_label_location=np.argmax(y_train,axis=1)
                        iii=0
                        for i in range(500):
                            if x_train_label_location[i] == y_train_label_location[i]:
                                iii = iii + 1
                        print('训练集上的正确率：'+str(iii/500))
                        R['train_accuracy']=iii/500
                        R['train_loss']=sess.run(loss,feed_dict={x: R['train_inputs'], y_true: R['y_true_train']})
                        
                        x_test_label = sess.run(y,feed_dict={x: R['test_inputs'], y_true: R['y_true_test']})
                        x_test_label_location=np.argmax(x_test_label,axis=1)
                        y_test_label_location=np.argmax(y_test,axis=1)
                        iii=0
                        for i in range(500):
                            if x_test_label_location[i] == y_test_label_location[i]:
                                iii = iii + 1
                        print('测试集上的正确率：'+str(iii/500))
                        R['test_accuracy']=iii/500
                        R['test_loss']=sess.run(loss,feed_dict={x: R['test_inputs'], y_true: R['y_true_test']})
                        
                        print(R['test_loss'])
                        self.plotloss()
                        self.ploty(name='%s'%(ii))
                        self.savefile()
                        if R['issave']: 
                            nametmp='%smodel/'%(FolderName)
                            shutil.rmtree(nametmp)
                            saver.save(sess, "%smodel.ckpt"%(nametmp))
                    
            def low_fre_ratio_one(self,xx,yy,s_filter_wid,diff_x2=[]):
                #print(type(diff_x2))
                f_low, f_high=get_f_high_low(yy,xx,s_filter_wid,diff_x2) #获得低频和高频的成分
                syy=np.sum(np.square(yy)) #计算真实的y的矩阵范数，这是不变的，有意思的是
                ratio=[]
                for f_ in f_low: #如果有多个delta,就会得到多个f_low
                    sf=np.sum(np.square(f_))/syy #计算低频比，即低频成分的矩阵范数除以真实的y的矩阵范数
                    ratio.append(sf) 
                #print(np.shape(ratio))
                return ratio #返回的是一个list，但这个list可能又不止一个元素，因为有不止一个f_low
                
            def low_fre_ratio(self,output_all,y):
                 ratio_all=[]
                 ratio=self.low_fre_ratio_one(R['train_inputs_nl'][0],R['y_true_train'],R['s_filter_wid'],diff_x2=dist_input) #这个是不变的，以真实的input和output计算出的
                 ratio_all.append(ratio) 
                 for out in output_all: #以此计算以每个隐藏层的神经元作为input，真实的y作为output的低频比
                     ratio=self.low_fre_ratio_one(out,R['y_true_train'],R['s_filter_wid'],diff_x2=[])
                     ratio_all.append(ratio)
                 return ratio_all #如果有多个delta，那么应该是形如:[[1,2],[3,4]...[n,n+1]]，其中每组依次对应[delta1,delta2]；且依次对应，输入层，第一层，第二层...第n层
            def plotloss(self): 
                plt.figure() #画损失函数
                ax = plt.gca()
                y1 = R['loss_test']
                y2 = R['loss_train']
                plt.plot(y1,'ro',label='Test')
                plt.plot(y2,'g*',label='Train')
                ax.set_xscale('log')
                ax.set_yscale('log')                
                plt.legend(fontsize=18)
                plt.title('loss',fontsize=15)
                fntmp = '%sloss'%(FolderName)
                mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
                
                
                ratio_tmp=np.stack(R['ratio']) #画低频比了，这里将R['ratio']本来是个list，然后黏成了一个大的矩阵了,而且是增加一个维度的黏！
                layern=np.shape(R['ratio'])[1] #即，行数，也就是一共有多少个低频比每次
                # print(np.shape(R['ratio'])) = (10,5,2) 10是第10个plotepoch，5是5个低频比，2是两个delta
                plt.figure()
                ax = plt.gca() 
                for ijk  in range(layern): #不同倒数层最为input 分开画，要是有多个delta，自然就分开画多条曲线
                    plt.plot(ratio_tmp[:,ijk],label='layer:%s'%(ijk))
                    
                plt.xlabel('epoch (*%s)'%(R['plotepoch']),fontsize=18)
                plt.legend(fontsize=18)
                plt.title('low_fre_ratio',fontsize=15)
                fntmp = '%slowfreratio'%(FolderName)
                mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
                
                plt.figure() #这个是重复了
                ax = plt.gca()
                y1 = R['loss_test']
                y2 = R['loss_train']
                plt.plot(y1,'ro',label='Test')
                plt.plot(y2,'g*',label='Train')
                ax.set_xscale('log')
                ax.set_yscale('log')                
                plt.legend(fontsize=18)
                plt.title('loss',fontsize=15)
                fntmp = '%sloss'%(FolderName)
                mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
                
                
                #下面这个图的横坐标坐标尺度没有变，然后损失函数和低频闭画到一张图里了
                t=np.arange(np.shape(R['loss_train'])[0]) #np.arange([start, ]stop, [step, ]dtype=None)用于生成等差数组
                fig, ax1 = plt.subplots()
                
                color = 'tab:red'
                ax1.set_xlabel('epoch',fontsize=18)
                ax1.set_ylabel('loss', color=color,fontsize=18)
                ax1.plot(t, R['loss_train'], color=color) #画
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.set_yscale('log')
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis 
                color = 'tab:blue'
                ax2.set_ylabel('low fre ratio', color=color,fontsize=18)  # we already handled the x-label with ax1
                ax2.plot(t[::R['plotepoch']], ratio_tmp[:,-2], color=color) #后面的-2是指倒数第二层；前面是全部都要，但是每500个取一次，这是合理的，因为你ratio_tmp每R['plotepoch']才算一次
                ax2.tick_params(axis='y', labelcolor=color)
                
                #fig.tight_layout()  # otherwise the right y-label is slightly clipped
                fntmp = '%slossratio'%(FolderName)
                mySaveFig(plt,fntmp,isax=0,iseps=0)
                plt.close()
                
                #这个画的是w与b变化的大小
                for ij in range(np.shape(R2['w_Univ_end'])[0]): #注意一下，这个会随着plotepoch的增多，而增多，因为是append，所以以前的还在
                    tmp=np.sqrt(np.sum(np.square(R2['w_Univ_end'][ij]-R2['w_Univ_ini'][ij])))
                    R['w_dis'][ij].append(tmp)
                    tmp=np.sqrt(np.sum(np.square(R2['b_Univ_end'][ij]-R2['b_Univ_ini'][ij])))
                    R['b_dis'][ij].append(tmp)
                
                plt.figure()
                ax = plt.gca() 
                for ij in range(np.shape(R2['w_Univ_end'])[0]):
                    plt.plot(R['w_dis'][ij], label='%s-layer'%(ij))  
                #ax.set_yscale('log')                
                plt.legend(fontsize=18)
                plt.title('wdis',fontsize=15)
                fntmp = '%swdis'%(FolderName)
                mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
                #上面画的是w，下面画的是b
                plt.figure()
                ax = plt.gca() 
                for ij in range(np.shape(R2['w_Univ_end'])[0]):
                    plt.plot(R['b_dis'][ij], label='%s-layer'%(ij))  
                ax.set_yscale('log')                
                plt.legend(fontsize=18)
                plt.title('bdis',fontsize=15)
                fntmp = '%sbdis'%(FolderName)
                mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
                
                    
            def ploty(self,name=''): #如果是一维或者二维，那么就把图形化出来，就真实函数与，测试集的output
                
                if R['input_dim']==1:
                    plt.figure()
                    ax = plt.gca()
                    y1 = R['y_test']
                    y2 = R['y_train']
                    y3 = R['y_true_test']
                    plt.plot(R['test_inputs'][:,0],y1[:,0],'ro',label='Test')
                    plt.plot(R['train_inputs'][:,0],y2[:,0],'g*',label='Train')
                    plt.plot(R['test_inputs'][:,0],y3[:,0],'b*',label='True')
                    plt.title('y',fontsize=15)        
                    plt.legend(fontsize=18) 
                    fntmp = '%su_m%s'%(FolderName,name)
                    mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
                    
            def savefile(self): #保存模型参数的函数
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
                        
                    
        model1=model()
        model1.run(1000000)
    
    
