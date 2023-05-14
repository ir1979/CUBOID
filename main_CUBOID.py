# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import utility_ as ru
import matplotlib.pyplot as plt

output_path= os.getcwd()+'/'
input_path = os.getcwd()+'/' 
files_path = glob.glob(input_path + "*.csv")
files_path.sort()

    
lower_bound= 60
upper_bound= 61

for file_path in files_path:
    
    results= pd.DataFrame()  
    # load data------
    data= pd.read_csv(file_path)
    names= file_path.split('/')
    dataset_name=os.path.basename(names[-1])
    if dataset_name.endswith('.csv'):
        dataset_name = dataset_name[:-4] 
    total_len= len(data) 
    labels= data['anomaly']
    ano_ = data.index[data['anomaly']==1].tolist()
    data_type='deltax'
    algorithm='osc'
    k=3 # number of clusters
    # CUBOID implementation
    for window_size in range(lower_bound,upper_bound):
        
        print(f'\n dataset: {dataset_name}    window size:{window_size}....\n')
        clustering_inf_TS= ru.clustering_rep(data,window_size,k,data_type,algorithm)
        anomaly_scores_c= ru.Anomaly_score_clustering(clustering_inf_TS,)
        accuracy_c, precision_c, recall_c, f_score_c,ano_index_c= ru.performance_indexes(anomaly_scores_c,
                                                                      labels, window_size)
        
        print('clustering method results:-----------\n')
        print('accuracy= ', accuracy_c, '  precision= ',precision_c)
        print('recall=', recall_c, '   f_score=', f_score_c)
        
        current_result_c= {'dataset_name':dataset_name,'lenght':total_len,'method_name':'clustering',
                             'window_size':window_size, 'accuracy':accuracy_c,
                             'precision':precision_c, 'recall':recall_c,
                             'f_measure':f_score_c}
        #results= results.append(current_result_c, ignore_index=True)
    
    
        fig, ax =plt.subplots(2,1,figsize=(2,5)) # ,
        ax[0].plot(data['value'])
        series=data['value'].copy()
        for x in ano_:
            ax[0].plot(x,series[x],'rx')
        
        hax=[] 
        x_labels=[] 
        wid=[]
    
       
        hax+=[0]
        x_labels+=['0']
        i=0
        for j in range(1, len(anomaly_scores_c)):
                val=j*window_size 
                hax+=[val] 
                x_labels+=[str(val)]     
        ax[1].bar(hax,anomaly_scores_c, width=window_size*9/10, align='center') 
        series=anomaly_scores_c.copy()
        for x in ano_index_c:
            ax[1].bar(hax[x],anomaly_scores_c[x],width=window_size*9/10, align='center')
            ax[1].bar(hax[x],anomaly_scores_c[x],width=window_size*9/10, align='center', fill=False, hatch='////')
        alpha=['(a) Original time series ', '(b) CUBOID']
   
        for i in range(0,2):
            ax[i].set_xlim([0,len(data)])
            ax[i].set_xlabel(alpha[i],loc='center',labelpad=1.0,size=8)
            # ax[i].set_title(alpha[i], size=10,pad=-11)
            ax[i].set_ylabel('Anomaly Score',labelpad=15.0,size=8)
        ax[0].set_ylabel('Value',labelpad=5.0,size=8)
        plt.setp(ax[0].get_yticklabels(), fontsize=8)
    
    
        fig.tight_layout()
        fig.set_size_inches(10, 6)
        fig.savefig('Sin.png')
        
        plt.show()
        
        
        