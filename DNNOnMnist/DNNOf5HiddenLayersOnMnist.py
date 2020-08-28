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

###normalize the train input###
R['train_inputs_nl']=normalization_input([R['train_inputs']])
###compute the distance of x and itself, for computing LFR###
dist_input=compute_distances_no_loops(R['train_inputs_nl'][0],R['train_inputs_nl'][0])

###i represents for delta###
for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30]:
    ###j represents for the number of the hidden layers. In the paper, j only takes 5. But you could change it as you want###
    for j in [5]:
        R['s_filter_wid']=[i] 
        
        ### mkdir a folder to save all output###
        sBaseDir = 'fitnd1_delta'+str(R['s_filter_wid'])+'/'
        if platform.system()=='Windows':
            device_n="0"
            BaseDir = r'D:\deepstudy20200606/%s'%(sBaseDir)
        else:
            device_n="0"
            BaseDir = sBaseDir
            matplotlib.use('Agg')
            
        R['issave']=0 #whether save the model

        R['normalization']=1  #whether conduct normalization
        
        subFolderName = '%s'%(datetime.now().strftime("%y%m%d%H%M%S")) 
        FolderName = '%s%s/'%(BaseDir,subFolderName)
        mkdir(BaseDir)
        mkdir(FolderName) 
        
        if R['issave']: 
            print('save')
            mkdir('%smodel/'%(FolderName))
            
        R['FolderName']=FolderName 
        
        if  True: #not platform.system()=='Windows':
            shutil.copy(__file__,'%s%s'%(FolderName,os.path.basename(__file__)))
        
        ###followings are some critial parameters###
        R['input_dim']=28*28 
        R['output_dim']=10 
        R['epsion']=0.1 
        R['ActFuc']=1   ###  0: ReLU; 1: Tanh; 3:sin;4: x**5,, 5: sigmoid  6 sigmoid derivate 激活函数
        num_unit=500 #the number of the neures at each hidden layers
        R['hidden_units']=[num_unit]*j #the hidden layers
        R['fake_hidden_units']=[num_unit]*20 
        R['astddev']=np.sqrt(1/num_unit) # for weight
        R['bstddev']=np.sqrt(1/num_unit)# for bias terms
        R['ASI']=0
        R['learning_rate']=5e-5 
        R['learning_rateDecay']=2e-5 
        ### setup for activation function
        R['seed']=1 #random seed
        np.random.seed(R['seed']) #fix the random seed
        R['gpu']='0'
        os.environ["CUDA_VISIBLE_DEVICES"]='%s'%(R['gpu']) 
            
        R['plotepoch']=100 #plot every x epochs
        R['train_size']=30000;  ### training size 
        R['batch_size']=R['train_size'] # int(np.floor(R['train_size'])) ### batch size
        R['test_size']=10000  ### test size
        
        R['tol']=1e-2 #the loss when stopping
        R['Total_Step']=600000  ### the training step. Set a big number, if it converges, can manually stop training 
        R['isresnet']=0
        R['FolderName']=FolderName   ### folder for save images
             
        R['gpu']=device_n
     
        os.environ["CUDA_VISIBLE_DEVICES"]='%s'%(R['gpu']) 
        
        t0=time.time() 
        
        ###the function used for add one hidden layer###
        def add_layer2(x,input_dim = 1,output_dim = 1,isresnet=1,astddev=0.05,
                       bstddev=0.05,ActFuc=0,seed=0, name_scope='hidden'):
            if not seed==0: 
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
            
        ###this function is used for initialize the parameters. And it could keep the initialized parameters fixed.###
        def getWini(hidden_units=[10,20,40],input_dim = 1,output_dim_final = 1,astddev=0.05,bstddev=0.05):
            hidden_num = len(hidden_units)
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
        
        w_Univ0, b_Univ0=getWini(hidden_units=R['fake_hidden_units'], 
                                                 input_dim = R['input_dim'],
                                                 output_dim_final = R['output_dim'])
        
        ###now we are constructing the whole network###
        def univAprox2(x0, hidden_units=[10,20,40],input_dim = 1,output_dim_final = 1, #
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
        
        ###following is the network###
        tf.reset_default_graph() 
            
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
            
            loss=tf.reduce_mean(tf.square(y_true-y)) #the loss function
            # We define our train operation using the Adam optimizer
            adam = tf.train.AdamOptimizer(learning_rate=in_learning_rate)
            train_op = adam.minimize(loss)
        
        
        config = tf.ConfigProto(allow_soft_placement=True) 
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer()) 
        saver = tf.train.Saver()  
        R['w_dis']=[]
        R['b_dis']=[] 
        R2={}
        R2['w_Univ_ini']=[] #the initial w
        R2['b_Univ_ini']=[] #the initial b
        R2['w_Univ_end']=[] #the last w
        R2['b_Univ_end']=[] #the last b
        R['loss_epoch']=[]
        
        ###now, for convenience, we create a class###
        class model():
            def __init__(self): 
                R['y_train']=[] #the lastest y of training set
                R['y_test']=[] #the lastest y of test set
                R['loss_test']=[] #loss of test set for each epoch
                R['loss_train']=[] #loss of training set for each epoch
                R['ratio']=[]
                if R['issave']: #save the model
                    nametmp='%smodel/'%(FolderName)
                    mkdir(nametmp)
                    saver.save(sess, "%smodel.ckpt"%(nametmp))
                
                ###have a look at both training and test set###
                y_test, loss_test_tmp,w_Univ_tmp,b_Univ_tmp= sess.run([y,loss,w_Univ,b_Univ],feed_dict={x: R['test_inputs'], y_true: R['y_true_test']})
                y_train,loss_train_tmp= sess.run([y,loss],feed_dict={x: R['train_inputs'], y_true: R['y_true_train']})
                
                R['y_train']=y_train
                R['y_test']=y_test
                R['loss_test'].append(loss_test_tmp)
                R['loss_train'].append(loss_train_tmp)
                R2['w_Univ_ini']=w_Univ_tmp
                R2['b_Univ_ini']=b_Univ_tmp
                R['w_dis']=[]
                R['b_dis']=[]
                for tmp in R2['w_Univ_ini']: #save w and b
                    R['w_dis'].append([])
                    R['b_dis'].append([])
                self.ploty(name='ini') 
            
            ###run one step###
            def run_onestep(self): 
                ###have a look first###
                y_test, loss_test_tmp,w_Univ_tmp,b_Univ_tmp = sess.run([y,loss,w_Univ,b_Univ],feed_dict={x: R['test_inputs'], y_true: R['y_true_test']})
                R2['w_Univ_end'],R2['b_Univ_end']=w_Univ_tmp,b_Univ_tmp
                y_train,loss_train_tmp= sess.run([y,loss],feed_dict={x: R['train_inputs'], y_true: R['y_true_train']}) 
                    
                ###optimize for one time###
                if R['train_size']>R['batch_size']: #train by the batch size, if needed
                    indperm = np.random.permutation(R['train_size'])
                    nrun_epoch=np.int32(R['train_size']/R['batch_size'])
                    
                    for ijn in range(nrun_epoch):
                        ind = indperm[ijn*R['batch_size']:(ijn+1)*R['batch_size']] 
                        _= sess.run(train_op, feed_dict={x: R['train_inputs'][ind], y_true: R['y_true_train'][ind],
                                                          in_learning_rate:R['learning_rate']})
                else: 
                    _ = sess.run(train_op, feed_dict={x: R['train_inputs'], y_true: R['y_true_train'],
                                                          in_learning_rate:R['learning_rate']})
                R['learning_rate']=R['learning_rate']*(1-R['learning_rateDecay']) #change the learning rate
                ###save the parameters###
                R['y_train']=y_train
                R['y_test']=y_test
                R['loss_test'].append(loss_test_tmp)
                R['loss_train'].append(loss_train_tmp)
                
            def run(self,step_n=1): #run!!!f
                if R['issave']: #save the model if needed
                    nametmp='%smodel/model.ckpt'%(FolderName)
                    saver.restore(sess, nametmp)
                for ii in range(step_n):
                    self.run_onestep()
                    if R['loss_train'][-1]<R['tol']: #to decide whether to stop or not
                        print('model end, error:%s'%(R['loss_train'][-1]))
                        self.plotloss()
                        self.ploty()
                        self.savefile()
                        R['step']=ii
                        if R['issave']: 
                            nametmp='%smodel/'%(FolderName)
                            shutil.rmtree(nametmp) #shutil.rmtree() 
                            saver.save(sess, "%smodel.ckpt"%(nametmp))
                        break
                    if ii==0:
                        print('initial %s'%(np.max(R['y_train'])))
                        
                    if ii%R['plotepoch']==0: 
                        losss,y_train,out_Univ_tmp= sess.run([loss,y,out_Univ],feed_dict={x: R['train_inputs'], y_true: R['y_true_train']}) 
                        
                        if R['normalization']==1 : #normalize if needed
                            out_Univ_tmp=normalization_input(out_Univ_tmp)
                       
                        ratio_tmp=self.low_fre_ratio(out_Univ_tmp,y_train) #compute the LFR
                        R['ratio'].append(np.squeeze(ratio_tmp)) #save LFR
                        R['ratio_last']=R['ratio'][-1]#save the lastest LFR
                        R['loss_epoch'].append(losss)#save the loss
                        R['step']=ii
                        ###print some useful information###
                        print('len:%s, ii:%s'%(len(R['ratio']),ii)) 
                        print('time elapse: %.3f'%(time.time()-t0)) 
                        print('model, epoch: %d, test loss: %f' % (ii,R['loss_test'][-1]))
                        print('model, epoch: %d, train loss: %f' % (ii,R['loss_train'][-1])) 
                        print('ratio:%s'%(ratio_tmp))
                        
                        ###calculate the acc of the training set and test set, and 500 need to be changed, coresponding to the size you use###
                        x_train_label = sess.run(y,feed_dict={x: R['train_inputs'], y_true: R['y_true_train']})
                        x_train_label_location=np.argmax(x_train_label,axis=1)
                        y_train_label_location=np.argmax(y_train,axis=1)
                        iii=0
                        for i in range(500):
                            if x_train_label_location[i] == y_train_label_location[i]:
                                iii = iii + 1
                        print('the acc of training size：'+str(iii/500))
                        R['train_accuracy']=iii/500
                        R['train_loss']=sess.run(loss,feed_dict={x: R['train_inputs'], y_true: R['y_true_train']})
                        
                        x_test_label = sess.run(y,feed_dict={x: R['test_inputs'], y_true: R['y_true_test']})
                        x_test_label_location=np.argmax(x_test_label,axis=1)
                        y_test_label_location=np.argmax(y_test,axis=1)
                        iii=0
                        for i in range(500):
                            if x_test_label_location[i] == y_test_label_location[i]:
                                iii = iii + 1
                        print('the acc of test size：'+str(iii/500))
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
                            
            ###from here start, the functions below are all for computing LFR###
            def low_fre_ratio_one(self,xx,yy,s_filter_wid,diff_x2=[]):
                #print(type(diff_x2))
                f_low, f_high=get_f_high_low(yy,xx,s_filter_wid,diff_x2)
                syy=np.sum(np.square(yy)) 
                ratio=[]
                for f_ in f_low: 
                    sf=np.sum(np.square(f_))/syy 
                    ratio.append(sf) 
                #print(np.shape(ratio))
                return ratio 
                
            def low_fre_ratio(self,output_all,y):
                 ratio_all=[]
                 ratio=self.low_fre_ratio_one(R['train_inputs_nl'][0],R['y_true_train'],R['s_filter_wid'],diff_x2=dist_input) 
                 ratio_all.append(ratio) 
                 for out in output_all: 
                     ratio=self.low_fre_ratio_one(out,R['y_true_train'],R['s_filter_wid'],diff_x2=[])
                     ratio_all.append(ratio)
                 return ratio_all 
            ###To here end, all above are for computing LFR, and they are the same as in Resnet18###
            
            ###plot the loss. In fact, this function is not important at all. So, you could delete it if you wish.###
            def plotloss(self): 
                plt.figure() 
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
                
                ###plot the LFR###
                ratio_tmp=np.stack(R['ratio']) 
                layern=np.shape(R['ratio'])[1]
                plt.figure()
                ax = plt.gca() 
                for ijk  in range(layern): 
                    plt.plot(ratio_tmp[:,ijk],label='layer:%s'%(ijk))
                    
                plt.xlabel('epoch (*%s)'%(R['plotepoch']),fontsize=18)
                plt.legend(fontsize=18)
                plt.title('low_fre_ratio',fontsize=15)
                fntmp = '%slowfreratio'%(FolderName)
                mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
                
                plt.figure() 
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
                
                t=np.arange(np.shape(R['loss_train'])[0])
                fig, ax1 = plt.subplots()
                
                color = 'tab:red'
                ax1.set_xlabel('epoch',fontsize=18)
                ax1.set_ylabel('loss', color=color,fontsize=18)
                ax1.plot(t, R['loss_train'], color=color) 
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.set_yscale('log')
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis 
                color = 'tab:blue'
                ax2.set_ylabel('low fre ratio', color=color,fontsize=18)  # we already handled the x-label with ax1
                ax2.plot(t[::R['plotepoch']], ratio_tmp[:,-2], color=color) 
                ax2.tick_params(axis='y', labelcolor=color)
                
                #fig.tight_layout()  # otherwise the right y-label is slightly clipped
                fntmp = '%slossratio'%(FolderName)
                mySaveFig(plt,fntmp,isax=0,iseps=0)
                plt.close()
                
                #plot the change of w and b
                for ij in range(np.shape(R2['w_Univ_end'])[0]):
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

                plt.figure()
                ax = plt.gca() 
                for ij in range(np.shape(R2['w_Univ_end'])[0]):
                    plt.plot(R['b_dis'][ij], label='%s-layer'%(ij))  
                ax.set_yscale('log')                
                plt.legend(fontsize=18)
                plt.title('bdis',fontsize=15)
                fntmp = '%sbdis'%(FolderName)
                mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
                
                    
            def ploty(self,name=''): #plot the figure if the input dim is one or two.
                
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
                    
            def savefile(self): #save the model and parameters
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
                        
                    
        model1=model()
        model1.run(1000000)
    
    
