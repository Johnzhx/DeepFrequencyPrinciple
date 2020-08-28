# !python3
# -*- coding: utf-8 -*-
# author: flag
import sys
sys.path.insert(0,'D:/GDrive/jupyterNB/basicfolder')
from BasicFunc import mySaveFig
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
#%%
RR={}
Seq=[]

target_folder=r'D:\跟着许志钦搞科研\验证低频比问题\deepstudy 20200808 Resnet18和mnist 关于神经网络的输出\Resnet18/'
 
all_sub=os.listdir(target_folder)
#print(all_sub)
#print(all_sub)
for i in all_sub:
    if not i[0:4] == 'loss':
        continue
    fd = target_folder+'/'+ i
    all_sub_i = os.listdir(fd)
    print(all_sub_i)
    #print(all_sub_i)
    for k in all_sub_i:
        fdd = fd + '/' + k
        print(fdd[0:69])
        #print(len(fdd))
        #print(fdd[0:67])
        if not k[-4:] == '.pkl':
            continue
        with open(fdd,'rb') as f:
            R = pickle.load(f)
        #print(fdd + '/' + k)
        if fdd[0:69] == 'D:\跟着许志钦搞科研\验证低频比问题\deepstudy 20200617 Resnet18动图完整版//loss_and_acc_-1':
            RR['loss_and_-1_hist_val_loss']=R['hist_val_loss']
            RR['loss_and_-1_hist_loss']=R['hist_loss']
            RR['loss_and_-1_hist_acc']=R['hist_acc']
            RR['loss_and_-1_hist_val_acc']=R['hist_val_acc']
            RR['loss_and_-1_hist_test_loss']=R['hist_test_loss']
            RR['loss_and_-1_hist_test_acc']=R['hist_test_acc']
        if fdd[0:69] == 'D:\跟着许志钦搞科研\验证低频比问题\deepstudy 20200617 Resnet18动图完整版//loss_and_acc_-2':
            RR['loss_and_-2_hist_val_loss']=R['hist_val_loss']
            RR['loss_and_-2_hist_loss']=R['hist_loss']
            RR['loss_and_-2_hist_acc']=R['hist_acc']
            RR['loss_and_-2_hist_val_acc']=R['hist_val_acc']
            RR['loss_and_-2_hist_test_loss']=R['hist_test_loss']
            RR['loss_and_-2_hist_test_acc']=R['hist_test_acc']
        if fdd[0:69] == 'D:\跟着许志钦搞科研\验证低频比问题\deepstudy 20200617 Resnet18动图完整版//loss_and_acc_-3':
            RR['loss_and_-3_hist_val_loss']=R['hist_val_loss']
            RR['loss_and_-3_hist_loss']=R['hist_loss']
            RR['loss_and_-3_hist_acc']=R['hist_acc']
            RR['loss_and_-3_hist_val_acc']=R['hist_val_acc']
            RR['loss_and_-3_hist_test_loss']=R['hist_test_loss']
            RR['loss_and_-3_hist_test_acc']=R['hist_test_acc']
        if fdd[0:69] == 'D:\跟着许志钦搞科研\验证低频比问题\deepstudy 20200617 Resnet18动图完整版//loss_and_acc_-4':
            RR['loss_and_-4_hist_val_loss']=R['hist_val_loss']
            RR['loss_and_-4_hist_loss']=R['hist_loss']
            RR['loss_and_-4_hist_acc']=R['hist_acc']
            RR['loss_and_-4_hist_val_acc']=R['hist_val_acc']
            RR['loss_and_-4_hist_test_loss']=R['hist_test_loss']
            RR['loss_and_-4_hist_test_acc']=R['hist_test_acc']


plt.figure()
ax = plt.gca()
#plt.plot(all_m,all_Q_ini,'k*',label='Q_ini')
#plt.plot(delta_range_plott,LFR_layer7t1,'ro-',label='layer7')
plt.plot(RR['loss_and_-1_hist_val_loss'],'g,-',label='-1')
plt.plot(RR['loss_and_-2_hist_val_loss'],'b,-',label='-2')
plt.plot(RR['loss_and_-3_hist_val_loss'],'c,-',label='-3')
plt.plot(RR['loss_and_-4_hist_val_loss'],'m,-',label='-4')
#plt.title('hist_val_loss',fontsize=15)        
#ax.set_yscale('log')
plt.xlabel(r'epoch',fontsize=22)
plt.ylabel(r'val_loss',fontsize=22)
#ax.set_xscale('log')
#plt.ylim([0.5,1.1])
#plt.ylim([0.31,0.32])
plt.legend(fontsize=18) 
fntmp = '%s hist_val_loss'%(target_folder)
mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)

plt.figure()
ax = plt.gca()
#plt.plot(all_m,all_Q_ini,'k*',label='Q_ini')
#plt.plot(delta_range_plott,LFR_layer7t1,'ro-',label='layer7')
plt.plot(RR['loss_and_-1_hist_loss'],'g,-',label='-1')
plt.plot(RR['loss_and_-2_hist_loss'],'b,-',label='-2')
plt.plot(RR['loss_and_-3_hist_loss'],'c,-',label='-3')
plt.plot(RR['loss_and_-4_hist_loss'],'m,-',label='-4')
#plt.title('hist_loss',fontsize=15)        
#ax.set_yscale('log')
plt.xlabel(r'epoch',fontsize=22)
plt.ylabel(r'loss',fontsize=22)
#ax.set_xscale('log')
#plt.ylim([0.5,1.1])
#plt.ylim([0.31,0.32])
plt.legend(fontsize=18) 
fntmp = '%s hist_loss'%(target_folder)
mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)


plt.figure()
ax = plt.gca()
#plt.plot(all_m,all_Q_ini,'k*',label='Q_ini')
#plt.plot(delta_range_plott,LFR_layer7t1,'ro-',label='layer7')
plt.plot(RR['loss_and_-1_hist_acc'],'g,-',label='-1')
plt.plot(RR['loss_and_-2_hist_acc'],'b,-',label='-2')
plt.plot(RR['loss_and_-3_hist_acc'],'c,-',label='-3')
plt.plot(RR['loss_and_-4_hist_acc'],'m,-',label='-4')
#plt.title('hist_acc',fontsize=15)        
#ax.set_yscale('log')
plt.xlabel(r'epoch',fontsize=22)
plt.ylabel(r'acc',fontsize=22)
#ax.set_xscale('log')
#plt.ylim([0.5,1.1])
#plt.ylim([0.31,0.32])
plt.legend(fontsize=18) 
fntmp = '%s hist_acc'%(target_folder)
mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)

plt.figure()
ax = plt.gca()
#plt.plot(all_m,all_Q_ini,'k*',label='Q_ini')
#plt.plot(delta_range_plott,LFR_layer7t1,'ro-',label='layer7')
plt.plot(RR['loss_and_-1_hist_val_acc'],'g,-',label='-1')
plt.plot(RR['loss_and_-2_hist_val_acc'],'b,-',label='-2')
plt.plot(RR['loss_and_-3_hist_val_acc'],'c,-',label='-3')
plt.plot(RR['loss_and_-4_hist_val_acc'],'m,-',label='-4')
#plt.title('hist_val_acc',fontsize=15)        
#ax.set_yscale('log')
plt.xlabel(r'epoch',fontsize=22)
plt.ylabel(r'val_acc',fontsize=22)
#ax.set_xscale('log')
#plt.ylim([0.5,1.1])
#plt.ylim([0.31,0.32])
plt.legend(fontsize=18) 
fntmp = '%s val_acc'%(target_folder)
mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)

loss_and_hist_test_loss = [RR['loss_and_-1_hist_test_loss'],RR['loss_and_-2_hist_test_loss'],RR['loss_and_-3_hist_test_loss'],RR['loss_and_-4_hist_test_loss']]
layers_x = ['Resnet18-1','Resnet18-2','Resnet18-3','Resnet18-4']

plt.figure()
ax = plt.gca()
#plt.plot(all_m,all_Q_ini,'k*',label='Q_ini')
#plt.plot(delta_range_plott,LFR_layer7t1,'ro-',label='layer7')
plt.plot(layers_x,loss_and_hist_test_loss ,'r,-' , label = 'test_acc vs. function')
#plt.title('Resnet18 test_loss vs. function',fontsize=20)        
#ax.set_yscale('log')
plt.xlabel(r'function',fontsize=22)
plt.ylabel(r'test loss',fontsize=22)
plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xticks(fontproperties = 'Times New Roman', size = 16)
#ax.set_xscale('log')
#plt.ylim([0.5,1.1])
#plt.ylim([0.31,0.32])
plt.legend(fontsize=16) 
fntmp = '%s test_loss'%(target_folder)
mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)



loss_and_hist_test_acc = [RR['loss_and_-1_hist_test_acc'],RR['loss_and_-2_hist_test_acc'],RR['loss_and_-3_hist_test_acc'],RR['loss_and_-4_hist_test_acc']]
layers_x = ['Resnet18-1','Resnet18-2','Resnet18-3','Resnet18-4']

plt.figure()
ax = plt.gca()
#plt.plot(all_m,all_Q_ini,'k*',label='Q_ini')
#plt.plot(delta_range_plott,LFR_layer7t1,'ro-',label='layer7')
plt.plot(layers_x,loss_and_hist_test_acc ,'r,-' , label = 'test_acc vs. function')
#plt.title('Resnet18 test_acc vs. function',fontsize=20)        
#ax.set_yscale('log')
plt.xlabel(r'function',fontsize=22)
plt.ylabel(r'test acc',fontsize=22)
plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xticks(fontproperties = 'Times New Roman', size = 16)
#ax.set_xscale('log')
#plt.ylim([0.5,1.1])
#plt.ylim([0.31,0.32])
plt.legend(fontsize=16) 
fntmp = '%s test_acc'%(target_folder)
mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)

