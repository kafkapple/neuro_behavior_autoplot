# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:53:30 2018

@author: 2014_Joon_IBS
"""
import pandas as pd
import numpy as np
import glob
import os

def search_file(path_data='./'):
    #path = 'E:\Behavior data\FreezeFrame\MeA' # becareful 
    #os.getcwd() # current directory
    #os.listdir() #current lists
    
    path_list=[]
    file_list=[]
    
    count=np.array(0)
    ofl=np.array(0)#pd.DataFrame()#[]
    ret=np.array(0)#pd.DataFrame()#[]
    #dem=np.array(0)#pd.DataFrame()#[]
    
    for (path, _, files) in os.walk(path_data): # walk: 계속 깊이 들어갈 수 있음. glob 도 가능은 함
        path_list.append(path)
        file_list.append(files)
        
        
        #print(len(glob.glob(path+'/*.csv')))
        
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            
            if ext == '.csv' and 'freeze' in filename: # freeze frame csv 확장자 output 만 탐색
                id_frz = filename.find('freeze_')
                id_batch = filename.find('batch')
                batch_exp = filename[id_batch + 5]
                date_exp = filename[id_frz + 7:id_frz + 15]
                print('Batch no: {} / Data: {} / File name: {}'.format(batch_exp, date_exp, filename))
                count+=1 
                final_path = os.path.join(path_data, path, filename)
                total = data_ofl_read(final_path)
                
                if total.shape[1]>10: # OFL 일 때는
                    if len(ofl.shape) <1: #or len(dem.shape)<1:
                        ofl = total
                        #dem = total[1]
                    else:
                        ofl = ofl.append(total)
                        #dem.append(total[1])
                        #ofl = pd.merge(ofl, total[0], how='outer')
                        #dem = pd.merge(dem, total[1], how='outer')
                    #ofl.append(total[0])
                    #dem.append(total[1])f
                else:
                    if len(ret.shape) < 1:
                        ret = total
                    else:
                        #ret = pd.merge(ret, total)
                        ret = ret.append(total)
                    
    print('csv 파일 수는 {}개\n'.format(count))
                
    idx=0
    
    print(path_list[idx])
    print(file_list[idx])
    
    #dem.index = ofl.index
    ret.index = ofl.index
    
    ofl_ret = pd.concat([ofl, ret], axis=1)
    
    return ofl_ret

def data_vec(data):  # convert Pandas dataframe into vectors
    #if len(data.shape) == 1: # vector 하나인 경우.        
    n_category = len(data.keys())
    d_list = [0 for i in range(n_category)]
    legend=[]
    for i, i_key in enumerate(data.keys()):
        d_list[i] = np.array(data[i_key])
        d_list[i] = d_list[i][np.isnan(d_list[i])==False]  # remove NaN
        print('n = {}'.format(len(d_list[i])))
        legend.append(i_key + ' \n(n={})'.format(len(d_list[i])))
    return d_list, legend

def data_ofl_read(csv_file):

    # OFL time bin 만큼 추출
    df = pd.read_csv(csv_file, header=None, skiprows=3)
    if 'ret' in csv_file: # ret 인 경우 timebin 4
        print('ret data')
        time_bin = 4
        df = df.iloc[:,0:time_bin+1]  # 실제 데이터 부분만 취함.
        #ret_m = np.mean(df.iloc[:,1:5], axis=1)
        ret_m = np.mean(df, axis=1)
        ret_std = np.std(df,axis=1, ddof=1)
        df['ret_avg']=ret_m
        #df['ret_std']=ret_std # for CV during the period.
        df.rename(columns={0:'subject'}, inplace=True)
        df = df.sort_values(['subject'])
        total = df
        #total.to_csv('ret_total.csv')
    else:
        print('ofl data')
        time_bin = 9
        baseline_bin = 5  
        df = df.iloc[:,0:time_bin+1]  # 실제 데이터 부분만 취함.
        base_m = np.mean(df.iloc[:,1:baseline_bin+1], axis=1)
        ofl_m = np.mean(df.iloc[:,baseline_bin+1:], axis=1)
        df['b_avg']=base_m
        df['ofl_avg']=ofl_m 
        
        df.rename(columns={0:'subject'}, inplace=True)
        key_subject = df.keys()[0] # pandas 로 csv read 시, 기본 key 탐색 (첫번째 row)
        
        obs = df[df[key_subject].str.contains("dem|exclude") == False]  # dem, exclude 중 하나라도 포함되어 있으면 제외 -> obs 데이터만 추출 가능
        dem = df[df[key_subject].str.contains("dem")==True]  # dem data
        
        dem.index = obs.index
        #total = pd.concat([obs,dem],axis=1)
        
        ######임시로 끔.
        obs = obs.melt(id_vars='subject',var_name='timebin',value_name='freezing')
        dem = dem.melt(id_vars='subject',var_name='timebin',value_name='freezing')
        
        #dem = dem.assign(obsdem='dem')
        #obs = obs.assign(obsdem='obs')
    
        
        obs = obs.sort_values(['subject'])
        #dem.sort_values(['subject'])
        #total = pd.concat(total).sort_values(['subject', 'obsdem']).reset_index(drop=True)
        
        ########### 임시로 끔. 
        obs = obs.pivot_table(index='subject', columns = ['timebin'], values='freezing')
        dem = dem.pivot_table(index='subject', columns = ['timebin'], values='freezing')
        total = obs
        #total = total.reset_index(drop=True)
        
        
        
    #total.to_csv('pandas_total.csv', index=False)


    # 쉽게 보기 위한 pivot table
    #easy_total = total.pivot_table(index='subject', columns = ['timebin'], values='freezing')
        
    #print(easy_total.head())
    #total.to_pickle('pandas_result.pkl')
    
    return total

# cage 별 계산
def per_cage(df):
    n_total = df.values.shape[0]
    n_mice = 5 # 일단 하드코딩. 상황에 맞게 변화 가능.
    n_cage = int(n_total/n_mice) # cage 수
    
    total_coef_var=[]
    # CV (coefficient of variance) calculation per cage)
    for i in range(n_cage):
        target_sample = df.iloc[i*n_mice:i*n_mice+n_mice,-1]
        
        #print(target_sample)
        mm = np.mean(target_sample)
        ss = np.std(target_sample, ddof=1) # ddof =1 to apply sample std
        min_cage = np.min(target_sample)
        max_cage = np.max(target_sample)
        coef_var = ss/mm
        for j in range(n_mice):
            n_mice*j
        #print(mm, ss,coef_var)
        total_coef_var.append(coef_var)
    


def gen_num(i_batch=3, n_cage=4, n_mice=5):
    idx=[]
    for i in range(n_cage):
        for j in range(n_mice):
            idx.append('{}{}{}'.format(i_batch,i+1,j+1))
            
    d = pd.DataFrame(idx)
    d.to_csv('idx.csv')
    return np.array(idx)

def main():
    import csv
    path = r'E:\0Temp_data\Tube\tube_OFL'
    os.chdir(path)
    # put r for window directory
    out = search_file()
    out.to_csv('result.csv')
    print('Data process')
    #ofl = pd.DataFrame()
    #coef = []
    
#    path = '/Data/tube/'
#    os.chdir(path)
#    file = glob.glob('*.csv')
#    for f in file:    
#        obs, coef_var = data_ofl_read(f)
#        coef.append(coef_var)
#        ofl = ofl.append(obs)
#    ofl.to_csv('ofl.csv')
#    #coef.to_csv('coef_csv')
#    with open("output.csv", "wb") as f:
#        writer = csv.writer(f)
#        writer.writerows(coef)
#    
    #a=gen_num()
     

if __name__ == '__main__':
    main()