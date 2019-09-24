# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:53:30 2018

@author: 2014_Joon_IBS
"""
import pandas as pd
import numpy as np
import glob
import os
import re
import csv


n_mice = 5 # 일단 하드코딩. 상황에 맞게 변화 가능.
col_to_nan = ['b_avg','ofl_avg', 'ofl_norm' ] #'ofl_early', 'ofl_late', 'ofl_e_norm', 'ofl_l_norm'
path =r'E:\0Temp_data\Tube' #tube_OFL'# put r for window directory


def gen_num(i_batch=3, n_cage=4, n_mice=5):
    idx=[]
    for i in range(n_cage):
        for j in range(n_mice):
            idx.append('{}{}{}'.format(i_batch,i+1,j+1))
            
    d = pd.DataFrame(idx)
    d.to_csv('idx.csv')
    return np.array(idx)

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

def search_file(path_data, exp_type, exp_id):
    batch_exp = 0 # batch 정보 없으면, 1부터 시작해 하나씩 증가하도록.
    #flag_batch = False # 기본은 false 
    
    path_list=[]
    file_list=[]
    
    count=np.array(0)
    ofl=np.array(0)#pd.DataFrame()#[]
    ret=np.array(0)#pd.DataFrame()#[]

    for (path, _, files) in os.walk(path_data): # walk: 계속 깊이 들어갈 수 있음. glob 도 가능은 함
        path_list.append(path)
        file_list.append(files)
        files= [f for f in files if os.path.splitext(f)[-1] =='.csv'] # csv data only
    
        for filename in files:
            if int(batch_exp)>4:
                print('debug')
                
            #ext = os.path.splitext(filename)[-1]
            
            if 'freeze' in filename: # freeze frame csv 확장자 output 만 탐색
                id_frz = filename.find('freeze_')
                date_exp = filename[id_frz + 7:id_frz + 15] # date 
                
                if 'batch' in filename: # batch number
                    id_batch = filename.find('batch')
                    batch_exp = filename[id_batch + 5]
                else:
                    #flag_batch = True
                    batch_exp += 1
                    
                print('Exp type:{} Batch no: {} / Data: {} / File name: {}'.format(exp_type, batch_exp, date_exp, filename))
                count+=1 
                
                final_path = os.path.join(path_data, path, filename)
                total = data_ofl_read(final_path, batch_exp)
                
                if total.shape[1]>10: # OFL 일 때는
                    total['doe'] =int(date_exp) # 날짜 추가. ret 은 생략. 바로 다음날이니.
                    if len(ofl.shape) <1: #or len(dem.shape)<1:
                        ofl = total             
                    else:
                        ofl = ofl.append(total)
                else:
                    if len(ret.shape) < 1: # ret data 없는경우?
                        ret = total
                    else:
                        ret = ret.append(total)
                    
    print('csv 파일 수는 {}개\n'.format(count))
         
    try:
        ret.index = ofl.index
        ofl_ret = pd.concat([ofl, ret], axis=1)
        ofl_ret = ofl_ret.reset_index(drop=True)
        print('ofl / ret analysis done')
    except:
        dum_idx_ret = pd.DataFrame(ofl['subject'].iloc[len(ret):].rename('subject_ret'))#.index[len(ret):]
        ret = ret.append(dum_idx_ret, sort=True)
        ret.index = ofl.index
        ofl_ret = pd.concat([ofl, ret], axis=1)
        ofl_ret = ofl_ret.reset_index(drop=True)
        #ofl_ret['ret_avg']=0
        #ofl_ret['subject_ret']=ofl_ret['subject']
        print('\nofl only. no ret. add dummy ret data.\n')
        
    
    ### normalize
    ofl_ret['ofl_norm']=ofl_ret['ofl_avg']-ofl_ret['b_avg']
    if 'ofl_early' in ofl_ret.columns: # in case of early / late two  phase
        ofl_ret['ofl_e_norm']=ofl_ret['ofl_early']-ofl_ret['b_avg']
        ofl_ret['ofl_l_norm']=ofl_ret['ofl_late']-ofl_ret['b_avg']
    ofl_ret['ret_norm']=ofl_ret['ret_avg']-ofl_ret['b_avg']

    #### new idx
    
    ofl_ret.insert(0,column='exp_type', value = exp_type)  
    ofl_ret['exp_id']=exp_id
    ofl_ret['exp_batch_cage'] = (str(exp_id) + ofl_ret['batch_cage'].astype(str)).astype(int)
    
    list_id_move = ['subject','id_sub','id_original','subject_ret', 'doe']
    id_move = ofl_ret[list_id_move]
    
    ofl_ret = ofl_ret.drop(list_id_move, axis=1)
    ofl_ret = pd.concat([ofl_ret, id_move], axis=1)
    
    # fam 인 경우, n5 면 5라 보고 빈subject추가. tubeOFL 은 그냥 일단 빈 경우 추가. (4기준.)
    # fam 인 경우, tubeOFL 에서 obsdem 인 경우는 dem id 확보. 
    
    return ofl_ret


def data_ofl_read(csv_file, batch_n):
    # OFL time bin 만큼 추출
    n_cage_num = 4 # default 4 mice per cage
    df = pd.read_csv(csv_file, header=None, skiprows=3)
    flag_2phase = False
   ## id       
    n_id = 0
    if 'ret' in csv_file: # ret 인 경우 timebin 4
        print('ret data')
        time_bin = 4
        df = df.iloc[:,0:time_bin+1+n_id]  # 실제 데이터 부분만 취함.
        df.rename(columns={0:'subject_ret'}, inplace=True)
        df_d=df.iloc[:,1:time_bin+1] 
        ret_m = np.mean(df_d, axis=1)
        #ret_std = np.std(df_d,axis=1, ddof=1)
        
#        df = df.melt(id_vars='subject_ret',var_name='timebin',value_name='freezing')
#        df = df.sort_values(['subject_ret', 'timebin'])
#        df = df.pivot_table(index='subject_ret', columns = ['timebin'], values='freezing')

        df = df.reset_index(drop=True)
        df['ret_avg']=ret_m
        if ('fam' in csv_file) or ('obsdem' in csv_file):
            nans = np.where(np.empty_like(df.values), np.nan, np.nan)
            data = np.hstack([df.values, nans]).reshape(-1, df.shape[1])
            df = pd.DataFrame(data, columns=df.columns)
        
    else:
        print('ofl data')
        time_bin = 9 
        baseline_bin = 5  
        df = df.iloc[:,0:time_bin+1+n_id]  # 실제 데이터 부분만 취함.
        df.rename(columns={0:'subject'}, inplace=True)    
        
        #df.rename(columns={0:'subject'}, inplace=True)
        key_subject = df.keys()[0] # pandas 로 csv read 시, 기본 key 탐색 (첫번째 row)
        
        
        
        if ('fam' in csv_file) or ('obsdem' in csv_file):
            if 'n5' in csv_file:
                n_cage_num = 5
            ## Fam, obsdem 의 경우, dem id 확보
            #df_dem = df[df[key_subject].str.contains("dem") == True]['subject'].to_frame()
            #df = df[df[key_subject].str.contains("dem|exclude") == False]  # dem, exclude 중 하나라도 포함되어 있으면 제외 -> obs 데이터만 추출 가능
            
        else: # 일반적인 상황에선 dem 제외
            print('-0-------------')
            df = df[df[key_subject].str.contains("dem") == False]  # dem, exclude 중 하나라도 포함되어 있으면 제외 -> obs 데이터만 추출 가능
            
        base_m = np.mean(df.iloc[:,1:baseline_bin+1], axis=1)
        ofl_m = np.mean(df.iloc[:,baseline_bin+1:time_bin+1], axis=1)
        
        
                
            #df = pd.concat([df, df_dem], axis=0, sort=False) # dem 정보도 빈칸으로 추가. 
            
        df.insert(len(df.columns),column='id_original', value = df['subject'])   # backup
        if flag_2phase:
            
            ofl_e = np.mean(df.iloc[:,baseline_bin+1:time_bin-1], axis=1)
            ofl_l = np.mean(df.iloc[:,time_bin-1:time_bin+1], axis=1)
            df['ofl_early']=ofl_e 
            df['ofl_late']=ofl_l
        
        df['b_avg']=base_m
        df['ofl_avg']=ofl_m 
        
    
        try:
            chk_list =df['subject'].str.contains('-')
        except:
            print('\n-----------------ret case')
            chk_list =df['subject_ret'].str.contains('-')
        if ~chk_list.any(): # subject 이름이 그냥 숫자인 경우. 1,2,3,4,...
            n_cage_num = 5
            print('No dashes. - Default setting.')
            n_sub = df.shape[0]
            no_cage = int(n_sub/n_cage_num)
    
            id_cage_subject = []
            for i in range(no_cage):
                id_cage_subject += [str(i+1)+'-'+str(r) for r in list(range(1,n_cage_num+1))]
                
            df.loc[:,'subject'] = id_cage_subject 
            
        else: # subject 이름이 cage no-sub no 인경우. 1-1,1-2,... 보통 obs1-3, or obs1-3dom obs1-4re, etc...
            
            df['subject'] = df['subject'].str.findall(r'\d').apply(lambda x: '-'.join(map(str, x))[:3])
            print('n')
            # dem 특이 경우 일단 idx 그대로 두기.
            
        try:
            df['id_sub']=df['subject'].str[-1].astype(int) # subject id int 로 저장.
        except:
            print('here')
            
        df.insert(0,column='batch_cage', value = (str(batch_n) + df['subject'].str.findall(r'\d').apply(lambda x:x[0])).astype(int))  
    
    return df

def save_excel(df, save_name='result'):
    
    ## col selection
    writer = pd.ExcelWriter(save_name+'.xlsx', engine = 'xlsxwriter')
    df_total = df # backup total
    n_sub = df.shape[0]
    min_val=0
    min_int = 1
    max_int = 5
    
    if 'rank' in save_name:
        ref_col = '1'
        
        list_range = ['C2:F'+str(n_sub+1), 'H2:J'+str(n_sub+1)]
        list_range_max = [5,5]
        
    elif 'total' in save_name:
        ref_col = 'w_rank_norm'
        list_range = ['AT2:BA'+str(n_sub+1) ]#, 'H2:J'+str(n_sub+1)]
        list_range_max = [5,5]
        
    else:
        list_col = df.columns.tolist()
        cols=[i_col for i_col in list_col if not type(i_col)==int] # calculated value only. exclude timebin data.
        df = df[cols]
        df_total.to_excel(writer, sheet_name='raw')
        worksheet_raw = writer.sheets['raw']
        ref_col='ofl_norm'
        
        list_range = ['C2:G'+str(n_sub+1), 'I2:I{}'.format(n_sub+1)]
        max_val = np.max(df[ref_col].values)
        list_range_max = [max_val,5]
        
        color_range_raw = 'D2:E'+str(n_sub+1)
        
            
        worksheet_raw.conditional_format(color_range_raw, {'type': '3_color_scale',
                                            'min_value': min_val,
                                            'max_value': max_val, #'mid_type': "num",
                                         'min_color':'#a8e6cf',
                                         'max_color':'#fe4a49'})


    df.to_excel(writer, sheet_name='main')
    worksheet = writer.sheets['main']
    
    for i,_ in enumerate(list_range):
        worksheet.conditional_format(list_range[i], {'type': '3_color_scale',
                                            'min_value': min_val,
                                            'max_value': list_range_max[i], #'mid_type': "num",
                                         'min_color':'#a8e6cf',
                                         'max_color':'#fe4a49'})
#    
#    
#    workbook = writer.book
#    
#    format_min = workbook.add_format({'bg_color':   '#a8e6cf'})
#                               #'font_color': '#9C0006'}
#    format_max = workbook.add_format({'bg_color':   '#fe4a49'})                         
##    
#    
#    
#    
#    color_range_id = 'H2:H{}'+str(n_sub+1)
#    color_range_val = 'C2:G'+str(n_sub+1)
#    color_range_category = 'I2:I{}'.format(n_sub+1) #subject batch cage
#    
    
 
    
#    worksheet.conditional_format(color_range_id, {'type': '3_color_scale',
#                                            'min_value': 100,
#                                            'max_value': 999, #'mid_type': "num",
#                                         'min_color':'#a8e6cf',
##                                         'max_color':'#fe4a49'})
#    
#    print('aa')
#    # OFL freezing
#    worksheet.conditional_format(color_range_val, {'type': '3_color_scale',
#                                            'min_value': min_val,
#                                            'max_value': max_val, #'mid_type': "num",
#                                         'min_color':'#a8e6cf',
#                                         'max_color':'#fe4a49'})
#    print('aaa')
#    # rank                                     
#    worksheet.conditional_format(color_range_category, {'type': '3_color_scale',
#                                            'min_value': min_int,
#                                            'max_value': max_int, #'mid_type': "num",
#                                         'min_color':'#a8e6cf',
#                                         'max_color':'#fe4a49'})
#                                         
     # raw data                               
    
#    
    ## batch_cage 
    #worksheet.conditional_format('B2:B{}'.format(n_sub+1), {'type':'3_color_scale'})
#    
#    ## Rank : OFL / ret
#    worksheet.conditional_format('U2:V{}'.format(n_sub+1), {'type':     'cell',
#                                    'criteria': '>',
#                                    'value':     3,
#                                    'format':   format_max})
#    worksheet.conditional_format('U2:V{}'.format(n_sub+1), {'type':     'cell',
#                                    'criteria': '<',
#                                    'value':     3,
#                                    'format':   format_min})
    try:
        writer.save()
    except:
        print('file is open')

### exp_id 는 윈도우 폴더 순서. 대소문자 구분 x
# exp sort 는 구분 되는듯. 주의.
def main():
    
    #os.chdir(path)
    
    #### target dir list
    list_dir = next(os.walk(path))[1] # next means 1st output from walk generator. 3-tuple (dirpath, dirnames, filenames)-> dirname idx 1
    list_dir = [i for i in list_dir if '_' not in i] # include only group dir. not '_'
    
    
    total_df = pd.DataFrame() # null frame
    
    for i_d, i_dir in enumerate(list_dir):
        cur_dir = os.path.join(path, i_dir)
        os.chdir(cur_dir)
        # output from each exp type
        out = search_file(cur_dir, i_dir, i_d+1)
        print(out.shape)
        print(total_df.shape)
        
        total_df = pd.concat([total_df, out], sort=False, axis=0)
        ################################################################## GOOD
        total_df.loc[total_df['id_original'].str.contains('dem'),col_to_nan] = np.nan
        

    #id_frz = filename.find('freeze_')
     #           date_exp = filename[id_frz + 7:id_frz + 15] # date 
     
    # confirm / raw : id_original / subject_ret
    # drop : exp_id, batch_cage, subject
    # necessary: exp_batch_cage, id_sub
    list_drop_col = ['exp_id','batch_cage', 'subject']
    os.chdir(path)
    total_df = total_df.sort_values(by=['exp_type', 'batch_cage', 'subject'])
    new_group = total_df['id_original'].str.extract(r'exp_(\w+)') 
    total_df.loc[new_group.notnull()[0], 'exp_type'] = new_group.loc[new_group.notnull()[0]][0] # 더 단순화할수는?
    
    total_df.drop(list_drop_col, axis=1, inplace=True)
    try:
        save_excel(total_df,'result')
    except:
        print('no')
        save_excel(total_df,'result2')

    print('Data process')

     

if __name__ == '__main__':
   
    main()