###import the modules,this program is very similar to 'PlotCdfAndRdf.py'. So if you have any question, you could refer that one.###
import sys
sys.path.insert(0,'D:/GDrive/jupyterNB/basicfolder')
from BasicFunc import mySaveFig
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

RR={}
Seq=[]

target_folder=r'D:\deepstudy 20200808\Resnet18/' #your target folder!

###get the parameters you need###
all_sub=os.listdir(target_folder)
for i in all_sub:
    if not i[0:4] == 'loss':
        continue
    fd = target_folder+'/'+ i
    all_sub_i = os.listdir(fd)
    print(all_sub_i)
    for k in all_sub_i:
        fdd = fd + '/' + k
        print(fdd[0:48])
        if not k[-4:] == '.pkl':
            continue
        with open(fdd,'rb') as f:
            R = pickle.load(f)
        if fdd[0:48] == 'D:\deepstudy 20200808\Resnet18//loss_and_acc_-1':
            RR['loss_and_-1_hist_val_loss']=R['hist_val_loss']
            RR['loss_and_-1_hist_loss']=R['hist_loss']
            RR['loss_and_-1_hist_acc']=R['hist_acc']
            RR['loss_and_-1_hist_val_acc']=R['hist_val_acc']
            RR['loss_and_-1_hist_test_loss']=R['hist_test_loss']
            RR['loss_and_-1_hist_test_acc']=R['hist_test_acc']
        if fdd[0:48] == 'D:\deepstudy 20200808\Resnet18//loss_and_acc_-2':
            RR['loss_and_-2_hist_val_loss']=R['hist_val_loss']
            RR['loss_and_-2_hist_loss']=R['hist_loss']
            RR['loss_and_-2_hist_acc']=R['hist_acc']
            RR['loss_and_-2_hist_val_acc']=R['hist_val_acc']
            RR['loss_and_-2_hist_test_loss']=R['hist_test_loss']
            RR['loss_and_-2_hist_test_acc']=R['hist_test_acc']
        if fdd[0:48] == 'D:\deepstudy 20200808\Resnet18//loss_and_acc_-3':
            RR['loss_and_-3_hist_val_loss']=R['hist_val_loss']
            RR['loss_and_-3_hist_loss']=R['hist_loss']
            RR['loss_and_-3_hist_acc']=R['hist_acc']
            RR['loss_and_-3_hist_val_acc']=R['hist_val_acc']
            RR['loss_and_-3_hist_test_loss']=R['hist_test_loss']
            RR['loss_and_-3_hist_test_acc']=R['hist_test_acc']
        if fdd[0:48] == 'D:\deepstudy 20200808\Resnet18//loss_and_acc_-4':
            RR['loss_and_-4_hist_val_loss']=R['hist_val_loss']
            RR['loss_and_-4_hist_loss']=R['hist_loss']
            RR['loss_and_-4_hist_acc']=R['hist_acc']
            RR['loss_and_-4_hist_val_acc']=R['hist_val_acc']
            RR['loss_and_-4_hist_test_loss']=R['hist_test_loss']
            RR['loss_and_-4_hist_test_acc']=R['hist_test_acc']

#plot val loss
plt.figure()
ax = plt.gca()
plt.plot(RR['loss_and_-1_hist_val_loss'],'g,-',label='-1')
plt.plot(RR['loss_and_-2_hist_val_loss'],'b,-',label='-2')
plt.plot(RR['loss_and_-3_hist_val_loss'],'c,-',label='-3')
plt.plot(RR['loss_and_-4_hist_val_loss'],'m,-',label='-4')
plt.xlabel(r'epoch',fontsize=22)
plt.ylabel(r'val_loss',fontsize=22)
plt.legend(fontsize=18) 
fntmp = '%s hist_val_loss'%(target_folder)
mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)

#plot train loss
plt.figure()
ax = plt.gca()
plt.plot(RR['loss_and_-1_hist_loss'],'g,-',label='-1')
plt.plot(RR['loss_and_-2_hist_loss'],'b,-',label='-2')
plt.plot(RR['loss_and_-3_hist_loss'],'c,-',label='-3')
plt.plot(RR['loss_and_-4_hist_loss'],'m,-',label='-4')
plt.xlabel(r'epoch',fontsize=22)
plt.ylabel(r'loss',fontsize=22)
plt.legend(fontsize=18) 
fntmp = '%s hist_loss'%(target_folder)
mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)

#plot train acc
plt.figure()
ax = plt.gca()
plt.plot(RR['loss_and_-1_hist_acc'],'g,-',label='-1')
plt.plot(RR['loss_and_-2_hist_acc'],'b,-',label='-2')
plt.plot(RR['loss_and_-3_hist_acc'],'c,-',label='-3')
plt.plot(RR['loss_and_-4_hist_acc'],'m,-',label='-4')
plt.xlabel(r'epoch',fontsize=22)
plt.ylabel(r'acc',fontsize=22)
plt.legend(fontsize=18) 
fntmp = '%s hist_acc'%(target_folder)
mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)

#plot val acc
plt.figure()
ax = plt.gca()
plt.plot(RR['loss_and_-1_hist_val_acc'],'g,-',label='-1')
plt.plot(RR['loss_and_-2_hist_val_acc'],'b,-',label='-2')
plt.plot(RR['loss_and_-3_hist_val_acc'],'c,-',label='-3')
plt.plot(RR['loss_and_-4_hist_val_acc'],'m,-',label='-4')
plt.xlabel(r'epoch',fontsize=22)
plt.ylabel(r'val_acc',fontsize=22)
plt.legend(fontsize=18) 
fntmp = '%s val_acc'%(target_folder)
mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)

loss_and_hist_test_loss = [RR['loss_and_-1_hist_test_loss'],RR['loss_and_-2_hist_test_loss'],RR['loss_and_-3_hist_test_loss'],RR['loss_and_-4_hist_test_loss']]
layers_x = ['Resnet18-1','Resnet18-2','Resnet18-3','Resnet18-4']

#plot tess loss between different structures
plt.figure()
ax = plt.gca()
plt.plot(layers_x,loss_and_hist_test_loss ,'r,-' , label = 'test_acc vs. function')
plt.xlabel(r'function',fontsize=22)
plt.ylabel(r'test loss',fontsize=22)
plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xticks(fontproperties = 'Times New Roman', size = 16)
plt.legend(fontsize=16) 
fntmp = '%s test_loss'%(target_folder)
mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)

loss_and_hist_test_acc = [RR['loss_and_-1_hist_test_acc'],RR['loss_and_-2_hist_test_acc'],RR['loss_and_-3_hist_test_acc'],RR['loss_and_-4_hist_test_acc']]
layers_x = ['Resnet18-1','Resnet18-2','Resnet18-3','Resnet18-4']

#plot tess acc between different structures
plt.figure()
ax = plt.gca()
plt.plot(layers_x,loss_and_hist_test_acc ,'r,-' , label = 'test_acc vs. function')
plt.xlabel(r'function',fontsize=22)
plt.ylabel(r'test acc',fontsize=22)
plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xticks(fontproperties = 'Times New Roman', size = 16)
plt.legend(fontsize=16) 
fntmp = '%s test_acc'%(target_folder)
mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)

