###this program is similar to the one in Resnet18###
###import the module###
import sys
sys.path.insert(0,'D:/GDrive/jupyterNB/basicfolder')
from BasicFunc import mySaveFig
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

###iii*100 represents for the epochs###
for iii in range(15):
    RR={}
    Seq=[]
    
    target_folder=r'D:\deepstudy20200606 mnist/' #this is your target folder, and is changed with different location.
    
    ###find the LFR we have saved, and put them into dictionary RR, for computing CDF and RDF###
    all_sub=os.listdir(target_folder)
    print(all_sub)
    for i in all_sub:
        if i[-1:] != ']':
            continue
        fd = target_folder+'/'+ i
        all_sub_i = os.listdir(fd)
        print(all_sub_i)
        for j in all_sub_i:
            fdd = fd +'/'+ j
            all_sub_ij = os.listdir(fdd)
            print(all_sub_ij)
            for k in all_sub_ij:
                if not k[-4:] == '.pkl':
                    continue
                with open(fdd + '/' + k,'rb') as f:
                    R = pickle.load(f)
                if R['s_filter_wid'][0]<1:
                    continue
                ###we use the 5-hidden-layers network, so we have to make sure that we use the right one###
                if np.shape(R['hidden_units'])[0] != 5: 
                    continue
                if R['s_filter_wid'] not in Seq:
                    Seq.append(R['s_filter_wid'])
                if ('R[s_filter_wid]'+str(R['s_filter_wid'])) not in RR:
                    RR['R[s_filter_wid]'+str(R['s_filter_wid'])]={}
                print(np.shape(R['ratio'][-2]))
                if np.shape(R['hidden_units'])[0]==5:
                    RR['R[s_filter_wid]'+str(R['s_filter_wid'])]['ratio_of_5layers_of_1stlast']=R['ratio'][iii][-1]
                    RR['R[s_filter_wid]'+str(R['s_filter_wid'])]['ratio_of_5layers_of_2ndlast']=R['ratio'][iii][-2]
                    RR['R[s_filter_wid]'+str(R['s_filter_wid'])]['ratio_of_5layers_of_3rdlast']=R['ratio'][iii][-3]
                    RR['R[s_filter_wid]'+str(R['s_filter_wid'])]['ratio_of_5layers_of_4thlast']=R['ratio'][iii][-4]
                    RR['R[s_filter_wid]'+str(R['s_filter_wid'])]['ratio_of_5layers_of_5thlast']=R['ratio'][iii][-5]
            
    Seq.sort(reverse = True)
    print(RR)
    print(Seq)
    
    
    delta_range_plott=[]
    delta_range_plot=[]
    LFR_layer5t1=[]
    LFR_layer51=[]
    LFR_layer5t2=[]
    LFR_layer52=[]
    LFR_layer53=[]
    LFR_layer5t3=[]
    LFR_layer54=[]
    LFR_layer5t4=[]
    LFR_layer55=[]
    LFR_layer5t5=[]
    
    ###compute for the CDF and RDF###
    for i in Seq:
        delta_range_plott.append(1/i[0])
        LFR_layer5t2.append(RR['R[s_filter_wid]'+str(i)]['ratio_of_5layers_of_2ndlast'])
    for j in range(len(Seq)-1):
        LFR_layer52.append((RR['R[s_filter_wid]'+str(Seq[j])]['ratio_of_5layers_of_2ndlast']-RR['R[s_filter_wid]'+str(Seq[j+1])]['ratio_of_5layers_of_2ndlast'])/(1/Seq[j][0]-1/Seq[j+1][0]))
        delta_range_plot.append((1/Seq[j][0]+1/Seq[j+1][0])/2)
    
    for k in Seq:
        LFR_layer5t1.append(RR['R[s_filter_wid]'+str(k)]['ratio_of_5layers_of_1stlast'])
    for l in range(len(Seq)-1):
        LFR_layer51.append((RR['R[s_filter_wid]'+str(Seq[l])]['ratio_of_5layers_of_1stlast']-RR['R[s_filter_wid]'+str(Seq[l+1])]['ratio_of_5layers_of_1stlast'])/(1/Seq[l][0]-1/Seq[l+1][0]))
    
    for k in Seq:
        LFR_layer5t3.append(RR['R[s_filter_wid]'+str(k)]['ratio_of_5layers_of_3rdlast'])
    for l in range(len(Seq)-1):
        LFR_layer53.append((RR['R[s_filter_wid]'+str(Seq[l])]['ratio_of_5layers_of_3rdlast']-RR['R[s_filter_wid]'+str(Seq[l+1])]['ratio_of_5layers_of_3rdlast'])/(1/Seq[l][0]-1/Seq[l+1][0]))
    
    for k in Seq:
        LFR_layer5t4.append(RR['R[s_filter_wid]'+str(k)]['ratio_of_5layers_of_4thlast'])
    for l in range(len(Seq)-1):
        LFR_layer54.append((RR['R[s_filter_wid]'+str(Seq[l])]['ratio_of_5layers_of_4thlast']-RR['R[s_filter_wid]'+str(Seq[l+1])]['ratio_of_5layers_of_4thlast'])/(1/Seq[l][0]-1/Seq[l+1][0]))
    
    for k in Seq:
        LFR_layer5t5.append(RR['R[s_filter_wid]'+str(k)]['ratio_of_5layers_of_5thlast'])
    for l in range(len(Seq)-1):
        LFR_layer55.append((RR['R[s_filter_wid]'+str(Seq[l])]['ratio_of_5layers_of_5thlast']-RR['R[s_filter_wid]'+str(Seq[l+1])]['ratio_of_5layers_of_5thlast'])/(1/Seq[l][0]-1/Seq[l+1][0]))
    
    ###plot the CDF###
    plt.figure()
    ax = plt.gca()
    plt.plot(delta_range_plott,LFR_layer5t1,'r.-',label='layer5',linewidth=0.8)
    plt.plot(delta_range_plott,LFR_layer5t2,'g.-',label='layer4',linewidth=0.8)
    plt.plot(delta_range_plott,LFR_layer5t3,'b.-',label='layer3',linewidth=0.8)
    plt.plot(delta_range_plott,LFR_layer5t4,'c.-',label='layer2',linewidth=0.8)
    plt.plot(delta_range_plott,LFR_layer5t5,'m.-',label='layer1',linewidth=0.8)
    plt.text(0.65,0.6,'epoches = %s00'%(iii),size=12)
    plt.xlabel(r'$1/\delta$',fontsize=22)
    plt.ylabel(r'LFR',fontsize=22)
    plt.legend(fontsize=12) 
    fntmp = '%s 5 layers DNN Different layers LFR distribution function'%(target_folder) + str(iii)
    mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
    
    ###plot the RDF###
    plt.figure()
    ax = plt.gca()
    plt.plot(delta_range_plot,LFR_layer51,'r.-',label='layer5',linewidth=0.8)
    plt.plot(delta_range_plot,LFR_layer52,'g.-',label='layer4',linewidth=0.8)
    plt.plot(delta_range_plot,LFR_layer53,'b.-',label='layer3',linewidth=0.8)
    plt.plot(delta_range_plot,LFR_layer54,'c.-',label='layer2',linewidth=0.8)
    plt.plot(delta_range_plot,LFR_layer55,'m.-',label='layer1',linewidth=0.8)      
    plt.text(0.59,1.15,'epoches = %s00'%(iii),size=12)    #ax.set_yscale('log')
    plt.xlabel(r'$1/\delta$',fontsize=22)
    plt.ylabel(r'RDF',fontsize=22)
    plt.legend(fontsize=12) 
    fntmp = '%s 5 layers DNN  Different layers LFR density function'%(target_folder) + str(iii)
    mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)

