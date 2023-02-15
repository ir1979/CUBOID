# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import utility_ as ru

output_path= f'D:/PHDThesis/anomalyWithTimeSeries/projects/copy of osc 14000816/osc-master/output/'
folders=[ 'A1Benchmark','A2Benchmark', 'A3Benchmark', 'A4Benchmark'] 
input_path = os.getcwd() + '/yahooData/'+folders[0]
files_path = glob.glob(input_path + "*.csv")
files_path.sort()

if not os.path.exists('output'):
    os.makedirs('output')
    
lower_bound= 10
upper_bound= 100

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
    data_type='deltax'
    algorithm='osc'
    k=3 # number of clusters
    # CUBOID implementation
    for window_size in range(lower_bound,upper_bound):
        
        print(f'\n dataset: {dataset_name}    window size:{window_size}....\n')
        clustering_inf_TS= ru.clustering_rep(data,window_size,k,data_type,algorithm)
        animaly_scores_c= ru.Anomaly_score_clustering(clustering_inf_TS,)
        accuracy_c, precision_c, recall_c, f_score_c= ru.performance_indexes(animaly_scores_c,
                                                                      labels, window_size)
        
        print('clustering method results:-----------\n')
        print('accuracy= ', accuracy_c, '  precision= ',precision_c)
        print('recall=', recall_c, '   f_score=', f_score_c)
        
        current_result_c= {'dataset_name':dataset_name,'lenght':total_len,'method_name':'clustering',
                             'window_size':window_size, 'accuracy':accuracy_c,
                             'precision':precision_c, 'recall':recall_c,
                             'f_measure':f_score_c}
        results= results.append(current_result_c, ignore_index=True)
        
        
             
    results.to_csv('output/'+ dataset_name +'.csv')
    print('1 file is saved')
        
        
        
        
        
        
        