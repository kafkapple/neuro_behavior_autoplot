# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:53:30 2018

@author: 2014_Joon_IBS
"""
import pandas as pd
import numpy as np
import glob

def data_vec(data):  # convert Pandas dataframe into vectors
    n_category = len(data.keys())
    d_list = [0 for i in range(n_category)]
    legend=[]
    for i, i_key in enumerate(data.keys()):
        d_list[i] = np.array(data[i_key])
        d_list[i] = d_list[i][np.isnan(d_list[i])==False]  # remove NaN
        print('n = {}'.format(len(d_list[i])))
        legend.append(i_key + ' \n(n={})'.format(len(d_list[i])))
    return d_list, legend

def data_ofl_read(path):
    csv_file = glob.glob(path+'*.csv')[0]
    print(csv_file)

    # OFL time bin 만큼 추출
    time_bin = 9
    df = pd.read_csv(csv_file, header=None, skiprows=3).iloc[:,0:time_bin+1]
    df.rename(columns={0:'subject'}, inplace=True)
    key_subject = df.keys()[0] # pandas 로 csv read 시, 기본 key 탐색 (첫번째 row)
    
    df = df.melt(id_vars='subject',var_name='timebin',value_name='freezing')
    
    obs = df[df[key_subject].str.contains("dem|exclude")==False]  # dem, exclude 중 하나라도 포함되어 있으면 제외 -> obs 데이터만 추출 가능
    dem = df[df[key_subject].str.contains("dem")==True]  # dem data
    
    dem = dem.assign(obsdem='dem')
    obs = obs.assign(obsdem='obs')

    total = [obs,dem]
    total = pd.concat(total).sort_values(['subject', 'obsdem']).reset_index(drop=True)
    total.to_csv('pandas_total.csv', index=False)
    print(total.head())

    # 쉽게 보기 위한 pivot table
    easy_total = total.pivot_table(index='subject', columns = ['timebin'], values='freezing')
    easy_total.to_csv('normal_total.csv')
    print(easy_total.head())
    
    return total

def main():
    print('Data process')
     

if __name__ == '__main__':
    main()