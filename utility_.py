# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 17:57:07 2021

@author: Elham
"""
import pandas as pd
import numpy as np
from optimal_sequence_clustering import optimal_sequence_clustering as osc

def clustering_rep(data,window_size,k,data_type, algorithm):

    results=[]
    all_center=[]
    delta_x= []
    data_list = data['value'].tolist()
    data_list= [0]+ data_list
    data_len= window_size*(len(data)//window_size)    # this an exact coefficient of data len to remove uncomplete last window
    windows_counter=0
    delta_frame=pd.DataFrame() # computing delta x
    if data_type=='deltax':
        for i in range(0,len(data_list)-1):
            delta_x+=[data_list[i+1]-data_list[i]]
        delta_frame['value']=delta_x
    
    for  start in range(0, data_len,window_size): 
        if algorithm== 'osc':
            if data_type=='deltax':
                X2=delta_frame.loc[start:start+window_size-1]
            elif data_type=='x':    
                X2=data.loc[start:start+window_size-1]
            X2= X2['value']
            X2=np.array(X2)
            X_array=X2
            centers, sizes, SSE = osc(X_array, k)
            all_center+=[centers]
            start=0
            label=0
            yhat=[None]*len(X_array)
            for j in sizes:
                end = start+j
                yhat[start:end]=[label for k in range(start,end)]
                start=end
                label+=1
        description= model_description(yhat,X2,algorithm,centers, windows_counter)
        results+=[description]
        windows_counter+=1
    return results


def performance_indexes (anomaly_scores,labels, winsize):
    
    min_anomaly_windows = 1   # minimum number of anomaly windows
    max_anomaly_windows = int(len(anomaly_scores)*0.01)
    num_windows=len(anomaly_scores)
    len_data=len(labels)
    ano_index=[]
    anomaly_points = [0]*len_data
    anomaly_percent= min(num_windows, max(min_anomaly_windows, max_anomaly_windows))
    ano_index= np.argsort(anomaly_scores)[::-1][:anomaly_percent]
    for i in ano_index:
        if i!=1:
            for j in range(i*winsize,min(len_data, (i+1)*winsize)):
            
                anomaly_points[j]=1

    
    TP=0
    TN=0
    FP=0
    FN=0        
    for i in range(0,len_data):
        
        if labels[i]==1 and anomaly_points[i]==1:
            TP+=1
        elif labels[i]==0 and anomaly_points[i]==1:
            FP+=1
        elif labels[i]==1 and anomaly_points[i]==0:
            FN+=1
        elif labels[i]==0 and anomaly_points[i]==0:
            TN+=1
            
    total=  TP+TN+FP+FN 
             
    precision = (TP)/(TP+FP+0.001)
    recall = (TP)/(TP+FN+0.001)
    f_score= 2*(precision*recall)/(precision+recall+ 0.001)
    accuracy= (TP+TN)/total
    
    return accuracy, precision, recall, f_score


def model_description(yhat, data, algorithm,centers, windows_counter=1):
    
    clusters = np.unique(yhat)
    num_clusters= len(clusters)
    total_records=len(yhat)
    members=[]
    description_info=pd.DataFrame(columns=['number_of_clussters','windows_counter',
                                           'cluster','number_of_members','center',
                                           'density', 'max_distance', 'min_distance'])
    for i in range(0,len(clusters)):
        members.append([])
        
    for cluster,j in zip(clusters, range(0, len(clusters))):
        for i in range(0, len(yhat)):
            if yhat[i]==cluster:
                 members[j]+= [i]
   
    for mem,i in zip(members, range(0,num_clusters)):
         description_info.loc[i,'number_of_members']= len(mem)
         description_info.loc[i,'density']=len(mem)/total_records
         description_info.loc[i,'cluster']= i
         description_info.loc[i,'windows_counter']= windows_counter
         description_info.loc[i,'number_of_clussters']= num_clusters
         if algorithm=='osc':
             description_info.loc[i,'center']=centers[i]
             center=centers[i]
         dist=[]
         max_dist=float('-inf')
         min_dist=float('inf')
         if algorithm=='osc':
             for idx in mem:
                 dist=abs(data[idx]-center)
                 if max_dist< dist:
                     max_dist= dist
                     
                 if min_dist > dist:
                     min_dist=dist
         description_info.loc[i,'min_distance'] = min_dist
         description_info.loc[i,'max_distance']= max_dist  
        
    return description_info

def Anomaly_score_clustering (clustering_inf_TS):
    
    num_windows=len(clustering_inf_TS)
    max_num_cluster=0
    # this loop find maximum number of clusters between all sliding windows of TS
    for win in clustering_inf_TS:
        curent_win_num_clusters = len(np.unique(win['cluster']))
        if curent_win_num_clusters > max_num_cluster:
            max_num_cluster = curent_win_num_clusters
    
    # compute difference between centers of clusters in all slid windows
    anomaly_socres=[]
    dis_type='2prev'
    if dis_type== '2prev':  # very bad results
        for i in range(0,num_windows):
            cur_win =  clustering_inf_TS[i]
            delta_center= 0
            if i>1 and i< num_windows:
                prev2_win = clustering_inf_TS[i-2]
                prev_win =  clustering_inf_TS[i-1] 
                for center_of_cur_win, center_of_prev_win, center_of_prev2_win in zip(cur_win['center'], prev_win['center'], prev2_win['center']):
                     delta_center += (abs(center_of_cur_win-center_of_prev_win)+ abs(center_of_cur_win-center_of_prev2_win))/2
        
            elif i==0:
                
                delta_center =0
                     
            elif i== 1:        
                prev_win =  clustering_inf_TS[i-1] 
                for center_of_cur_win,  center_of_prev_win in zip(cur_win['center'],  prev_win['center']):
                     delta_center +=  abs(center_of_cur_win-center_of_prev_win)
            
            anomaly_socres+= [delta_center] 
    return anomaly_socres 




def normalization (input_list):
    
    max_list= max( input_list)
    min_list = min(input_list)
    output_list=[]
    for i in range(0, len(input_list)):
        output_list+=[(input_list[i]-min_list)/(max_list - min_list)]
    
    return output_list    

    