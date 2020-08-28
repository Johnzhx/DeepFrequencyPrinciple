###import the modules#
import sys
from BasicFunc import mySaveFig #this model is uploaded in the folder
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np

###create a ditionary to save data and variates.###
RR={}
Seq=[]

###this is the folder where you save the results of the program 'Resnet18xComputingLFR.py' and 'Resnet18xTrain.py'###
###r'xxxxxxxxxxxx/',don't forget '/'!###
target_folder= r'D:\deepstudy 20200617 Resnet18/'
 
###to get all the direction in the target folder###
all_sub=os.listdir(target_folder)

###jjj represents the epochs you have trained. As for us, we have trained 0, 1, 2, 3, 4, 5, 10, 15, 20, 40, 50, 60, 70, 80 epochs respectively.###
for jjj in [0,1,2,3,4,5,10,15,20,40,50,60,70,80]:
    for i in all_sub:
        #to make sure other files and folders will not disturb our program
        if i[-1:] != ']':
            continue
        #to sort by the number of epochs
        #print(i[0:22])
        if jjj == 0:
            if i[0:21] != 'my_model_Resnet18-4_0' and i[0:21] != 'my_model_Resnet18-3_0' and i[0:21] != 'my_model_Resnet18-2_0' and i[0:21] != 'my_model_Resnet18-1_0':
                continue
        if jjj == 1:
            if i[0:21] != 'my_model_Resnet18-4_1' and i[0:21] != 'my_model_Resnet18-3_1' and i[0:21] != 'my_model_Resnet18-2_1' and i[0:21] != 'my_model_Resnet18-1_1':
                continue
        if jjj == 2:
            if i[0:21] != 'my_model_Resnet18-4_2' and i[0:21] != 'my_model_Resnet18-3_2' and i[0:21] != 'my_model_Resnet18-2_2' and i[0:21] != 'my_model_Resnet18-1_2':
                continue
        if jjj == 3:
            if i[0:21] != 'my_model_Resnet18-4_3' and i[0:21] != 'my_model_Resnet18-3_3' and i[0:21] != 'my_model_Resnet18-2_3' and i[0:21] != 'my_model_Resnet18-1_3':
                continue
        if jjj == 4:
            if i[0:21] != 'my_model_Resnet18-4_4' and i[0:21] != 'my_model_Resnet18-3_4' and i[0:21] != 'my_model_Resnet18-2_4' and i[0:21] != 'my_model_Resnet18-1_4':
                continue
        if jjj == 5:
            if i[0:21] != 'my_model_Resnet18-4_5' and i[0:21] != 'my_model_Resnet18-3_5' and i[0:21] != 'my_model_Resnet18-2_5' and i[0:21] != 'my_model_Resnet18-1_5':
                continue
        if jjj == 10:
            if i[0:22] != 'my_model_Resnet18-4_10' and i[0:22] != 'my_model_Resnet18-3_10' and i[0:22] != 'my_model_Resnet18-2_10' and i[0:22] != 'my_model_Resnet18-1_10':
                continue
        if jjj == 15:
            if i[0:22] != 'my_model_Resnet18-4_15' and i[0:22] != 'my_model_Resnet18-3_15' and i[0:22] != 'my_model_Resnet18-2_15' and i[0:22] != 'my_model_Resnet18-1_15':
                continue
        if jjj == 20:
            if i[0:22] != 'my_model_Resnet18-4_20' and i[0:22] != 'my_model_Resnet18-3_20' and i[0:22] != 'my_model_Resnet18-2_20' and i[0:22] != 'my_model_Resnet18-1_20':
                continue
        if jjj == 40:
            if i[0:22] != 'my_model_Resnet18-4_40' and i[0:22] != 'my_model_Resnet18-3_40' and i[0:22] != 'my_model_Resnet18-2_40' and i[0:22] != 'my_model_Resnet18-1_40':
                continue
        if jjj == 50:
            if i[0:22] != 'my_model_Resnet18-4_50' and i[0:22] != 'my_model_Resnet18-3_50' and i[0:22] != 'my_model_Resnet18-2_50' and i[0:22] != 'my_model_Resnet18-1_50':
                continue
        if jjj == 60:
            if i[0:22] != 'my_model_Resnet18-4_60' and i[0:22] != 'my_model_Resnet18-3_60' and i[0:22] != 'my_model_Resnet18-2_60' and i[0:22] != 'my_model_Resnet18-1_60':
                continue
        if jjj == 70:
            if i[0:22] != 'my_model_Resnet18-4_70' and i[0:22] != 'my_model_Resnet18-3_70' and i[0:22] != 'my_model_Resnet18-2_70' and i[0:22] != 'my_model_Resnet18-1_70':
                continue
        if jjj == 80:
            if i[0:22] != 'my_model_Resnet18-4_80' and i[0:22] != 'my_model_Resnet18-3_80' and i[0:22] != 'my_model_Resnet18-2_80' and i[0:22] != 'my_model_Resnet18-1_80':
                continue
        
        #continue to get the file we need
        fd = target_folder+'/'+ i
        all_sub_i = os.listdir(fd)
        #print(all_sub_i)
        for k in all_sub_i:
            fdd = fd + '/' + k
            #one should take care that, for different target folder, fdd is different, so the lenth of fragment is different
            print(fdd[0:51])
            #to get the .pkl file
            if not k[-4:] == '.pkl':
                continue
            #now, we get dictionary 'R', it is what we need to compute CDF and RDF
            with open(fdd,'rb') as f:
                R = pickle.load(f)
            #to select a suitable range for delta
            if R['s_filter_wid'][0]<0.016:
                continue
            if R['s_filter_wid'][0]>50.0:
                continue
            #save the delta in Seq
            if R['s_filter_wid'] not in Seq:
                Seq.append(R['s_filter_wid'])
            #put the materials into RR, for computing CDF and RDF
            if ('R[s_filter_wid]'+str(R['s_filter_wid'])) not in RR:
                RR['R[s_filter_wid]'+str(R['s_filter_wid'])]={}
            #put LFR of -1 and -2 hidden layers of different varients of the Resnet18 into relevant position, for convenience to calculate
            if fdd[0:51] == r'D:\deepstudy 20200617 Resnet18//my_model_Resnet18-4':
                RR['R[s_filter_wid]'+str(R['s_filter_wid'])]['ratio_of_3layers_of_1stlast']=R['ratio_last'][-1][0]
                RR['R[s_filter_wid]'+str(R['s_filter_wid'])]['ratio_of_3layers_of_2ndlast']=R['ratio_last'][-2][0]
            if fdd[0:51] == r'D:\deepstudy 20200617 Resnet18//my_model_Resnet18-3':
                RR['R[s_filter_wid]'+str(R['s_filter_wid'])]['ratio_of_4layers_of_1stlast']=R['ratio_last'][-1][0]
                RR['R[s_filter_wid]'+str(R['s_filter_wid'])]['ratio_of_4layers_of_2ndlast']=R['ratio_last'][-2][0]
            if fdd[0:51] == r'D:\deepstudy 20200617 Resnet18//my_model_Resnet18-2':
                RR['R[s_filter_wid]'+str(R['s_filter_wid'])]['ratio_of_5layers_of_1stlast']=R['ratio_last'][-1][0]
                RR['R[s_filter_wid]'+str(R['s_filter_wid'])]['ratio_of_5layers_of_2ndlast']=R['ratio_last'][-2][0]
            if fdd[0:51] == r'D:\deepstudy 20200617 Resnet18//my_model_Resnet18-1':
                RR['R[s_filter_wid]'+str(R['s_filter_wid'])]['ratio_of_6layers_of_1stlast']=R['ratio_last'][-1][0]
                RR['R[s_filter_wid]'+str(R['s_filter_wid'])]['ratio_of_6layers_of_2ndlast']=R['ratio_last'][-2][0]
    
    
    Seq.sort(reverse = True)
    
    delta_range_plott=[]
    delta_range_plot=[]
    LFR_layer3t1=[]
    LFR_layer4t1=[]
    LFR_layer5t1=[]
    LFR_layer6t1=[]
    LFR_layer7t1=[]
    LFR_layer31=[]
    LFR_layer41=[]
    LFR_layer51=[]
    LFR_layer61=[]
    LFR_layer71=[]
    LFR_layer3t2=[]
    LFR_layer4t2=[]
    LFR_layer5t2=[]
    LFR_layer6t2=[]
    LFR_layer7t2=[]
    LFR_layer32=[]
    LFR_layer42=[]
    LFR_layer52=[]
    LFR_layer62=[]
    LFR_layer72=[]
    
    ###compute CDF of -2 hidden layer###
    for i in Seq:
        delta_range_plott.append(1/i[0])
        LFR_layer3t2.append(RR['R[s_filter_wid]'+str(i)]['ratio_of_3layers_of_2ndlast'])
        LFR_layer4t2.append(RR['R[s_filter_wid]'+str(i)]['ratio_of_4layers_of_2ndlast'])
        LFR_layer5t2.append(RR['R[s_filter_wid]'+str(i)]['ratio_of_5layers_of_2ndlast'])
        LFR_layer6t2.append(RR['R[s_filter_wid]'+str(i)]['ratio_of_6layers_of_2ndlast'])
    ###compute RDF of -2 hidden layer###
    for j in range(len(Seq)-1):
        LFR_layer32.append((RR['R[s_filter_wid]'+str(Seq[j])]['ratio_of_3layers_of_2ndlast']-RR['R[s_filter_wid]'+str(Seq[j+1])]['ratio_of_3layers_of_2ndlast'])/(1/Seq[j][0]-1/Seq[j+1][0]))
        LFR_layer42.append((RR['R[s_filter_wid]'+str(Seq[j])]['ratio_of_4layers_of_2ndlast']-RR['R[s_filter_wid]'+str(Seq[j+1])]['ratio_of_4layers_of_2ndlast'])/(1/Seq[j][0]-1/Seq[j+1][0]))
        LFR_layer52.append((RR['R[s_filter_wid]'+str(Seq[j])]['ratio_of_5layers_of_2ndlast']-RR['R[s_filter_wid]'+str(Seq[j+1])]['ratio_of_5layers_of_2ndlast'])/(1/Seq[j][0]-1/Seq[j+1][0]))
        LFR_layer62.append((RR['R[s_filter_wid]'+str(Seq[j])]['ratio_of_6layers_of_2ndlast']-RR['R[s_filter_wid]'+str(Seq[j+1])]['ratio_of_6layers_of_2ndlast'])/(1/Seq[j][0]-1/Seq[j+1][0]))
        delta_range_plot.append((1/Seq[j][0]+1/Seq[j+1][0])/2)
    ###compute CDF of -1 hidden layer###
    for k in Seq:
        LFR_layer3t1.append(RR['R[s_filter_wid]'+str(k)]['ratio_of_3layers_of_1stlast'])
        LFR_layer4t1.append(RR['R[s_filter_wid]'+str(k)]['ratio_of_4layers_of_1stlast'])
        LFR_layer5t1.append(RR['R[s_filter_wid]'+str(k)]['ratio_of_5layers_of_1stlast'])
        LFR_layer6t1.append(RR['R[s_filter_wid]'+str(k)]['ratio_of_6layers_of_1stlast'])
    ###compute RDF of -1 hidden layer###
    for l in range(len(Seq)-1):
        LFR_layer31.append((RR['R[s_filter_wid]'+str(Seq[l])]['ratio_of_3layers_of_1stlast']-RR['R[s_filter_wid]'+str(Seq[l+1])]['ratio_of_3layers_of_1stlast'])/(1/Seq[l][0]-1/Seq[l+1][0]))
        LFR_layer41.append((RR['R[s_filter_wid]'+str(Seq[l])]['ratio_of_4layers_of_1stlast']-RR['R[s_filter_wid]'+str(Seq[l+1])]['ratio_of_4layers_of_1stlast'])/(1/Seq[l][0]-1/Seq[l+1][0]))
        LFR_layer51.append((RR['R[s_filter_wid]'+str(Seq[l])]['ratio_of_5layers_of_1stlast']-RR['R[s_filter_wid]'+str(Seq[l+1])]['ratio_of_5layers_of_1stlast'])/(1/Seq[l][0]-1/Seq[l+1][0]))
        LFR_layer61.append((RR['R[s_filter_wid]'+str(Seq[l])]['ratio_of_6layers_of_1stlast']-RR['R[s_filter_wid]'+str(Seq[l+1])]['ratio_of_6layers_of_1stlast'])/(1/Seq[l][0]-1/Seq[l+1][0]))

    #plot CDF of -1 layer
    plt.figure()
    ax = plt.gca()
    plt.plot(delta_range_plott,LFR_layer6t1,'go-',label='Resnet18-1')
    plt.plot(delta_range_plott,LFR_layer5t1,'bo-',label='Resnet18-2')
    plt.plot(delta_range_plott,LFR_layer4t1,'co-',label='Resnet18-3')
    plt.plot(delta_range_plott,LFR_layer3t1,'mo-',label='Resnet18-4')
    plt.xlabel(r'$1/\delta$',fontsize=20)
    plt.ylabel(r'LFR',fontsize=20)
    plt.ylim([0.060,1.05])
    plt.legend(fontsize=18) 
    fntmp = '%s%s -1 count backwards of delta v.s. LFR CDF'%(target_folder,jjj)
    mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
    
    #plot RDF of -1 layer
    plt.figure()
    ax = plt.gca()
    plt.plot(delta_range_plot,LFR_layer61,'go-',label='Resnet18-1')
    plt.plot(delta_range_plot,LFR_layer51,'bo-',label='Resnet18-2')
    plt.plot(delta_range_plot,LFR_layer41,'co-',label='Resnet18-3')
    plt.plot(delta_range_plot,LFR_layer31,'mo-',label='Resnet18-4')
    plt.title('%s -1layer 1/delta_v.s._LFR density function'%jjj,fontsize=15)        
    plt.xlabel(r'1/delta',fontsize=18)
    plt.ylabel(r'LFR',fontsize=18)
    plt.ylim([-0.02,2.0])
    plt.legend(fontsize=18) 
    fntmp = '%s%s -1 count backwards of delta_v.s._LFR RDF'%(target_folder,jjj)
    mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)

    #plot CDF of -2 layer
    plt.figure()
    ax = plt.gca()
    plt.plot(delta_range_plott,LFR_layer6t2,'go-',label='Resnet18-1')
    plt.plot(delta_range_plott,LFR_layer5t2,'bo-',label='Resnet18-2')
    plt.plot(delta_range_plott,LFR_layer4t2,'co-',label='Resnet18-3')
    plt.plot(delta_range_plott,LFR_layer3t2,'mo-',label='Resnet18-4')
    plt.title('%s -2layer 1/delta_v.s._LFR distribution function'%jjj,fontsize=15)        
    plt.xlabel(r'1/delta',fontsize=18)
    plt.ylabel(r'LFR',fontsize=18)
    plt.ylim([-0.00,1.05])
    plt.legend(fontsize=18) 
    fntmp = '%s%s -2 count backwards of delta v.s. LFR CDF'%(target_folder,jjj)
    mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
    
    #plot RDF of -2 layer
    plt.figure()
    ax = plt.gca()
    plt.plot(delta_range_plot,LFR_layer62,'g.-',label='Resnet18-1')
    plt.plot(delta_range_plot,LFR_layer52,'b.-',label='Resnet18-2')
    plt.plot(delta_range_plot,LFR_layer42,'c.-',label='Resnet18-3')
    plt.plot(delta_range_plot,LFR_layer32,'m.-',label='Resnet18-4')
    plt.xlabel(r'$1/\delta$',fontsize=20)
    plt.ylabel(r'RDF',fontsize=20)
    plt.ylim([-0.01,0.11])
    plt.legend(fontsize=18) 
    fntmp = '%s%s -2 count backwards of delta_v.s._LFR RDF'%(target_folder,jjj)
    mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
    

