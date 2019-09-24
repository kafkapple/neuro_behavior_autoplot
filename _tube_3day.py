# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 21:41:04 2019

@author: 2014_Joon_IBS
"""

import math
import pandas as pd
import numpy as np
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import matplotlib.pyplot as plt

import cal_stat as cs
import data_process as ds
#---

import os
import pandas as pd
import numpy as np
#path = r'E:\0Temp_data\Tube'
path=r'E:\0Temp_data\Tube' 
#fam\batch7'
#name='20190214_tube_fam_batch7.xlsx'
name = '_tube_3day_total.xlsx'

os.chdir(path)

flag_plot=False
w_decay = 0.8

list_file = glob.glob('*.xlsx')
list_file = [k for k in list_file if 'tube' in k]

name = list_file[0]# 'test_tube.xlsx'

flag_append = True
# cage / no 정보로부터 id 확보
n_trial = 3 # trial per day
rank_day='w_avg'#4 # rank 정하는 기준이 되는 날
flag_plot =False
fig_size = (20,17)
    
pastel_rainbow = ['#a8e6cf','#dcedc1','#ffd3b6','#ffaaa5', '#ff8b94']


def cat_plot(data, ax, idx_ax):
    if idx_ax ==0:
        flag_legend=True
    else:
        print('False')
        flag_legend=False
        
    sns.set_style('ticks',rc = {"lines.linewidth":0.1,"xtick.major.size": 0.1, "ytick.major.size": 1})
    sns.set_context("talk")
    sns.catplot(
                #palette={"male": "g", "female": "m"},
                x='day', y='rank', hue='subject',
                #markers=["^",'o'], linestyles=["-",'--'],linewidth=0.1,
                kind="point", data=data, ci=68, aspect=1, scale=1, legend=flag_legend, ax=ax)
    
     
def tube_rank(flag_append=flag_append):
    df = pd.read_excel(name, skiprows=3)
    d_col = [i for i in df.columns if type(i)!=str] # int col 만 선택.
    if 'total' in name: # total data 의 경우 따로 처리
        print('total')
        flag_append = False
        list_type = df['exp'].unique()
        df['id'] = df['exp']+df["batch"].map(str) + df["cage"].map(str)
        df_n_cage = df[['id','exp','n_cage']]
    else:    
        df['id'] = df['cage_trial'].fillna(method='ffill') #.astype(int)
        df_n_cage = df.groupby('id')['L'].transform(lambda x: x.unique().size).rename('n_cage') # 각 cage 별 subject 숫자 
        df_n_cage = pd.concat([df['id'], df_n_cage], axis=1)
        info_weight = df['weight'].dropna(how='all', axis=0)
        
        flag_append = True #개별 파일이므로, total 에 append 할 것.

    df = df[d_col] # rank result only. numeric values
    n_day = int(len(d_col)/n_trial) # total max day  -> 각기 처리할 필요?
    
    # 데이터 있는 day 만 선택
    df = df.dropna(how='all', axis=1)
    df = df.dropna(how='all', axis=0)
    #df = df.dropna(thresh = df.shape[1], axis=0) # 각 day별로 최소 2회의 승부가 있어야 함. 그렇지 않고 NA 값 더 많으면 drop
    df = df.fillna(0).astype(int)
    
    # 다시 info 합쳐주기
    df = pd.concat([df,df_n_cage[:df.shape[0]]], axis=1, sort=False)

    result_win = pd.DataFrame()
    
    ### group by 2중으로 개선하기. 가능하면.
    for i in range(n_day):
        # i~ n_trial
        i_winner = pd.Series()#np.array([])
        df_g = df.groupby('id')
        temp_win = pd.DataFrame()
        
        for g in df_g:
            i_n_cage = int(g[1].n_cage.mean()) # n =4 or 5 ... 약간은 편법.
            df_i= g[1].iloc[:,i*n_trial:(i+1)*n_trial] # 123, 456 ...
            i_count_win = df_i.apply(pd.value_counts, sort=False, axis=1).drop([0],axis=1) # 0 제외.  #.fillna(0).astype(int) + 1  #  #drop([0])
            if i_count_win.shape[1]==0: # data 없음.dummy 생성
                i_count_win[0]=0
                
            g_winner = i_count_win.idxmax(axis=1)
            i_winner = i_winner.append(g_winner)
            
            g_win_count = g_winner.value_counts(sort=False)
            
            if i_n_cage != len(g_win_count): # 한번도 못이긴 쥐 있는 경우, 찾아서 0 추가해주기.
                to_be_added = [gi for gi  in range(1,i_n_cage+1) if gi not in g_win_count.index] 
                for gg in to_be_added:
                    g_win_count.loc[gg]=0
                    
            g_win_count = pd.DataFrame({i+1 : g_win_count.sort_index()})
            g_win_count['id'] = g[0]
            
            
            temp_win = pd.concat([temp_win, g_win_count], sort=False, axis=0) #####
                    
        temp_win = temp_win[temp_win.index != 0] # 0 subject 제
        
 
        temp_win_count = temp_win[[t_col for t_col in temp_win.columns if t_col!='id']]
        result_win = pd.concat([result_win, temp_win_count], sort=False, axis=1) # 0 빼고, count 없는 subject 더해줬다 빼주기.
        
    
    result_win.insert(0, column='id', value = temp_win['id'])
    result_win['exp'] = result_win['id'].str.extract(r'(\D+)') 
    
    #if flag_append:
        #result_win.insert(1,column='weight', value = info_weight) ##############
    #df = pd.concat([df, df_winner], axis=1) # concat winner info
    df_subject = result_win.index
    result_win['subject']=  df_subject
    
    ############################## 4일 tube rank 결과
    col_total_days = [i for i in result_win.columns if type(i)==int]
    
    #### !!!
    n_dum = 50
    dummy=pd.DataFrame(data=9*np.ones((n_dum,1)), columns=['rank'])
    dummy['id']='OFLtube'
    
    ## average rank
    list_w_decay = [w_decay**i for i in range(n_day)][::-1]
    result_win['w_avg']=result_win[col_total_days].dot(list_w_decay) 
    
    result_win['rank']=result_win.groupby('id')[rank_day].rank(axis=1, method='min', ascending=False)
    
    grouped = result_win.groupby('id')
    
    ################## ㅜㅐ그
    result_win['w_avg_norm'] = grouped['w_avg'].apply(lambda x: (x-x.min())/(x.max()-x.min()))
    #result_win[['id','w_avg']].groupby('id').max()
    ###
    result_win = pd.concat([result_win, dummy], axis=0, sort=True)
    result_win = result_win.sort_values(by=['exp','id','subject'])
    
    import data_process as dp
    dp.save_excel(result_win, 'result_rank_3')
    
    #result_win.to_csv('result_rank.csv')
    
    if flag_plot:
        sns.set_style('ticks',rc = {"lines.linewidth":0.1,"xtick.major.size": 1, "ytick.major.size": 1})
        sns.set_context("talk")
        for c_d in col_total_days:
            result_win[c_d]=result_win.groupby('id')[c_d].rank(axis=1, method='min', ascending=False)
        
        result_win = pd.concat([result_win[['id','subject']], result_win[col_total_days]], axis=1)   
########################################## ###### 아주 중요!!
        result_win = result_win.melt(id_vars=['id','subject'],var_name='day',value_name='rank') ####
        result_win['chk']=result_win.groupby(['id','day'])['rank'].transform(sum) # 임시 
        result_win = result_win[result_win['chk']!=4] # 없는 데이터 제외. n4 일경우로 매우 제한적. 일화하기.
        n_plot =len(result_win.id.unique()) # plot no
        g= sns.catplot(
                palette=pastel_rainbow,#{"male": "g", "female": "m"},
            x='day', y='rank', hue='subject', col = 'id', data = result_win, col_wrap=int(math.sqrt(n_plot)), # 
                #markers=["^",'o'], linestyles=["-",'--'],linewidth=0.1,
            kind="point", ci=68, aspect=1.5, scale=1)
        #g.invert_yaxis()
        from matplotlib.ticker import MaxNLocator
        for i_ax in g.fig.axes:
            i_ax = i_ax.invert_yaxis() 
            #labels = [item.get_text() for item in i_ax.get_yticklabels()]
            #new_labels = [ "%d" % int(float(l)) if '.5' not in l else '' for l in labels]
            #i_ax.set_yticklabels(new_labels)
            # cause FacetGrid. just use inver_yaxis in normal case
            #i_ax.set(title='Cage: {}'.format(1))
#            ax[kk].axis('scaled')
            #ax[kk].autoscale(enable=True)
            #i_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        #g.fig.axes[0].set_yticklabels(g.fig.axes[0].get_yticklabels(), rotation = 0, fontsize = 8)
        #g = sns.FacetGrid(result_win, col="id",col_wrap=2)
        
        
        #g.map(sns.catplot, x='day',y= "rank", hue='subject')#, data = result_win)
        #g.map_dataframe(test_plot, 'day', 'rank')
        #g.add_legend();
# 
#            
#            
#           
    #ax[0].legend(loc='best', edgecolor =None, fontsize=3, bbox_to_anchor=(0.8,0.7)) # upper right
    #fig.tight_layout()
    #fig.show()
    #result_win.groupby('id').apply(plot_rank)
    #plt.show(fig)
    plt.savefig("image.png",bbox_inches='tight',dpi=100)
    
if __name__ == '__main__':
    #main()
    plt.close('all')
    tube_rank()
    
    
    #n_class = len(df['type'].unique())
    #print(n_class)
    #n_row = df.shape[0]
    #n_col = df.shape[1]
    #print(n_row, n_col)
    
