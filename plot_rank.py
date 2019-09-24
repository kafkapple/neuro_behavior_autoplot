# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 17:27:35 2018

@author: 2014_Joon_IBS
"""
import sys
import os
import glob

import pandas as pd
import numpy as np

from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from datetime import datetime

from matplotlib.ticker import MaxNLocator
from sklearn import preprocessing
sys.path.insert(0, 'F:\\github/dataplot/')
import data_process as ds

    

b_thr = 10
n_base = 5
n_total=9
path = r'E:\0Temp_data\Tube'
name = 'on_total_test.xlsx'

name_sheet='Sheet1'
target_group =  ['OFLtube']#['OFLtube']
target_hue = 'weight'#'comb_rank'#fix_obx'#'rel_rank' #'obsdem_rank'#'obsdem_rank'
target_x =  ['ofl_norm']#['b_avg','ofl_avg','ofl_norm']#, 'ret_avg', 'ret_norm']
val_max = 40

n_g = 2
# 4 bin OFL (on going)

###### normal
w_decay=0.7


flag_color_auto=False
now_target = 'rank_diff_std'#  #'win_std'
flag_pair = True #False
flag_project = 'tube2'
flag_group = False #True#False #True
flag_special = '---crowd_age'#'immature_n'#'crowd_age'#'n5'#'crowd_age'
####
tt='n_mice'#'age'#'rank'#'type'#'age'
#tt='age'
tt='rank'

plt.close('all')
if flag_project !='tube':
    name = 'total.xlsx'#'total_ofl_2.xlsx'
    target_group = [tt]#['age','gender',
    if tt=='tubeOFL':
        tt='n_mice'    
    elif 'rank' in tt:
        target_group = ['exp_type_obs','exp_type_dem']
        name_sheet='rank'
        
    target_hue = tt#'weight'#tt#'comb_rank'#'pr

#os.chdir('F:\\')
#ata = '/python/data/1115_tube_ofl_2.csv' #ofl_rank2.csv'
###### customize color palette
#title = 'OFL vs body weight'
title = 'Social Rank / OFL '
#ylabel = 'Weight (g)'
ylabel = 'avg freezing (%)'
ylim=[-2,val_max]
size = 12
fig_size =7
fig_ratio = 0.5
bar_width = 0.4
marker_size =100

params = {
       'axes.labelsize': size,
        "axes.titlesize":size,
      'font.size': size,
       'legend.fontsize': size*0.8,
       'legend.frameon':False,
       'xtick.labelsize': size,
       'ytick.labelsize': size,
       'text.usetex': False,
       
       }

plt.rcParams.update(params)

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
beach_towel = ['#fe4a49', '#2ab7ca', '#fed766', '#e6e6ea', '#f4f4f8']
pastel_rainbow = ['#ff8b94','#ffaaa5','#ffd3b6','#dcedc1','#a8e6cf','#85D1F7', '#B0B2E2','#768BC9']
              
#rainbow = ['#F77B7B','#F7AE7D','#F7F287','#a8e6cf','#85D1F7','#DFB7F3']#'#F77B7B' 초록
rainbow = ['#F77B7B','#EC875B', '#F7F287','#97F782','#85D1F7','#DFB7F3']
                  
list_palette = [flatui,
             beach_towel,
             pastel_rainbow]

if n_g ==2:
    pastel_rainbow =              ['#ffd3b6','#a8e6cf']
elif n_g==3:
    pastel_rainbow = ['#ff8b94','#ffd3b6','#a8e6cf']           # 3color      
else:
    pastel_rainbow = ['#ff8b94','#ffaaa5','#ffd3b6','#dcedc1','#a8e6cf','#85D1F7', '#B0B2E2','#768BC9']

pantone = ['#8CA2C9','#EEC4C3']
          
new_pal = [pastel_rainbow, rainbow, pantone]           
    
#### temp
#current_palette = color_setting(list_palette, i_color=2, n_color =4) #temp

#os.chdir(4'F:/0GoogleDrive/Research/Result/tube/tube_OFL/fam')
#cur_path = 4'E:/0Temp_data/Tube/tube_OFL'
 ##### nno use   
 
flag_2phase=False

# early/ late ofl
if flag_2phase:
    target_x.append('ofl_early')
    target_x.append('ofl_late')
###
def read_ofl(csv_file):
    time_bin = 9
    df = pd.read_csv(csv_file, header=None, skiprows=3).iloc[:,0:time_bin+1]
    
    #print(df.keys())
    key_subject = df.keys()[0] # pandas 로 csv read 시, 기본 key 탐색 (첫번째 row)
    obs = df[df[key_subject].str.contains("dem|exclude")==False]  # dem, exclude 중 하나라도 포함되어 있으면 제외 -> obs 데이터만 추출 가능
    #print(df.values.shape)
    base_m = np.mean(obs.iloc[:,1:6], axis=1)
    ofl_m = np.mean(obs.iloc[:,6:], axis=1)
    #print(df.iloc[0,0])
    obs['b_avg']=base_m
    obs['ofl_avg']=ofl_m
    
    n_total = obs.values.shape[0]
    n_mice = 5
    
    n_cage = int(n_total/n_mice) # cage 수
    #print(n_cage)
    total_coef_var = []
    for i in range(n_cage):
        target_sample = obs.iloc[i*n_mice:i*n_mice+n_mice,-1]
        
        #print(target_sample)
        mm = np.mean(target_sample)
        ss = np.std(target_sample, ddof=1) # ddof =1 to apply sample std
        coef_var = ss/mm
        #print(mm, ss,coef_var)
        total_coef_var.append(coef_var)
        
        #obs.iloc[i*n_mice:i*n_mice+n_mice].assign(coef =coef_var)
    #obs['coef'] = total_coef_var
    np.std()
    return obs, total_coef_var



def color_setting(list_palette=list_palette, i_color=1, n_color=2):
    sns.set_style('whitegrid')
    #sns.set_color_codes()
    if (n_color <=3) and (flag_color_auto==True):  # color 종류 별로 안써도 되는 경우
        print('color mapping')
        for i, i_p in enumerate(list_palette):
            list_palette[i] = i_p[::2]  #홀수 값만 취함. 
    #### 
    #sns.set_palette('hls',n_color) # Reds
    sns.set_palette(list_palette[i_color], n_color)
    current_palette = sns.color_palette()
    print(i_color, list_palette[i_color])
    
    #sns.palplot(current_palette)
    ###
    #sns.set_context('talk', font_scale=10)#, rc={"lines.linewidth": 2}) # “paper”, “talk”, and “poster”, which are version of the notebook parameters scaled by .8, 1.3, and 1.6,
    sns.set_context("talk", font_scale=1)#,rc={"font.size":10,"axes.titlesize":10,"axes.labelsize":10})   
    return current_palette

def legend_patch(current_palette, labels):
    patches = []
    for i, _ in enumerate(labels):
        patch_i = mpatches.Patch(color=current_palette[i], label=labels[i])
        patches.append(patch_i)
    return patches

####################33 reg
def plot_reg(data, x_target, y_target):
    fig_reg, ax_reg = plt.subplots()
    
    #data = data[data['pair_stable']=='y']
    #df_unstable = data[data['pair_stable']!='y']
    #data[data['pair_stable']=='un'].pair_stable =0 
    #data[data['pair_stable']=='s'].pair_stable = 'y'
    if flag_pair:
        lm = sns.lmplot(x=x_target, y=y_target, hue="pair_stable", data=data, palette = beach_towel);
        ax_reg=lm.axes
        fig_reg = lm.fig
    #sns.pairplot(data, x_vars=[x_target], y_vars=[y_target], hue="pair_stable", height=5, aspect=.8, kind="reg", ax=ax_reg);
    else:
        sns.regplot(data=data, x=data[x_target], y=data[y_target], scatter_kws={'s':marker_size},ax=ax_reg)
        ax_reg.grid(False)
        ax_reg.spines['top'].set_visible(False)
        ax_reg.spines['right'].set_visible(False)
        
        ax_reg.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_reg.set(ylim=[0,val_max])
    time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig_reg.tight_layout()
    fig_reg.savefig('reg_{}.png'.format(time_now), transparent = True, dpi=200)    
    
    data.reset_index(inplace=True, drop=True)
    
    df_pivot_reg = data.pivot_table(index=data.index, columns=[x_target], values=[y_target]) # relation
    df_pivot_reg = df_pivot_reg.loc[:,y_target]
    
    df_pivot_reg.sort_values(by=df_pivot_reg.columns.tolist(), inplace=True)
    df_pivot_reg.to_csv('reg_.csv')
    
def plot_bar_scatter_df(data, target_hue, target_x, figs, axs):
    ## setting
    margin=np.mean(data)
    #ylim=[np.min(data.min()-margin/2),np.max(data.max()+margin)]
    out = pd.DataFrame()

    #list_target = data['exp_type'].unique()
    n_target_group = len(data[target_hue].unique())# len(list_target) 주의. 실수.
    n_target_x =len(target_x)
    params = {
       'figure.figsize': [fig_size*fig_ratio*n_target_x, fig_size] # instead of 4.5, 4.5 # 5/6
       }
    plt.rcParams.update(params)
   
    # When creating the legend, only use the first two elements
    # to effectively remove the last two.
    ### size per group
    n_label = data.groupby(target_hue).size()
    n_label = [k for k in n_label]
        #data = data[data['exp_type']==target_group]
    if target_hue =='rel_rank':
        plot_reg(data=data, x_target = 'diff_rank',y_target='ofl_norm' )
    elif target_hue =='weight':
        fig_reg, ax_reg = plt.subplots()
        #sns.regplot(data=data, x=data['w_rank'], y=data['rank'], scatter_kws={'s':marker_size},ax=ax_reg)
        plot_reg(data, x_target='weight', y_target='ofl_norm')
        ###############
    elif target_hue ==tt: ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ time series plot
        
        d_col_time = [i for i in data.columns.tolist() if ((type(i)!=str) and (i<10))] # int col 만 선택.  1~9 min for OFL      
        #col_ids = ['exp_id','type','b_avg','ofl_avg','ofl_norm']
        max_freezing = data[d_col_time].values.max() # get the max value # val_max
        
        d_col_time.append('exp_id')
        d_col_time.append(target_hue)
        
        data_time = data[d_col_time].melt(id_vars=['exp_id', target_hue],var_name='timebin',value_name='freezing')
        
        fig_line, ax_line = plt.subplots(figsize=(fig_size, fig_size*fig_ratio))
        
        fig_set(fig_line, ax_line, n_target_x, n_target_group) #/2 ?
        #data_time.type = 'age'+data_time.type.astype(str)
        sns.set_style('ticks',rc = {"lines.linewidth":0.05,"xtick.major.size": 0.1, "ytick.major.size": 1})
#        if tt=='rank':
#            sns.lineplot(x='timebin', y='freezing',hue=tt, units='exp_id', estimator=None, lw=1, data=data_time,ax=ax_line)
        sns.catplot(
            #palette={"male": "g", "female": "m"},
            x='timebin', y='freezing', hue=tt, errwidth = 0.6,capsize=0.4,
            #linewidth=0.07, marker= 'o',#["^",'o'], linestyles=["-",'--'],
            kind="point", data=data_time, ci=68, aspect=1, scale=0.6, legend=False, ax=ax_line)
        
        # Facetgrid
        ax_line.set(title = 'OFL',
              ylabel='freezing(%)',
              xlabel='(min)',
              ylim=[0,max_freezing])
        
        ax_line.grid(False)
        ax_line.spines['top'].set_visible(False)
        ax_line.spines['right'].set_visible(False)
#
        fig_line.tight_layout()
            
    #fig_total, ax = plt.subplots()#(1,1, sharey=True)#, gridspec_kw = {'width_ratios':[7, 3]})
        
    #fig_set(fig_total, ax, n_target_x, n_target_group)
    id_target = ['date','id_sub'] #1. id 
    
    d_col = [i for i in data.columns if type(i)!=str] # int col 만 선택.
    id_target+=d_col # 2. OFL raw
    id_target+= target_x# 3. target_x: OFL norm
    if 'rank' in tt:
        
        col_rank = [i for i in data.columns if 'rank' in str(i)] # int col 만 선택.
        id_target+= col_rank
    else:
        id_target.append(target_hue)# 3. target_hue: ex) weight, rank... # ! caution! only if target hue is one
        
    df_val = data[id_target]
    df_val = df_val[~df_val[target_hue].isna()]  # nan exclude
    df_val = df_val.sort_values(by=target_hue) # sort
    
    #d_list[i] = d_list[i][np.isnan(d_list[i])==False]  # r
    
    melted = data.melt(id_vars=['exp_id',target_hue], value_vars = target_x, var_name='val_type',value_name='value' ).dropna() ##
#    
    fig_set(figs, axs, n_target_x, n_target_group)
    sns.barplot(x='val_type',y='value', data=melted, hue =target_hue, ci=68, errwidth = 1.0, capsize=.03, ax= axs)# ax[j]
    sns.stripplot(x='val_type',y='value', data=melted, hue = target_hue, jitter=True, size= 7,  
                  edgecolor = 'gray', linewidth=1, ax=axs, dodge=True)#,color="0.4")  # dodge is 
    #melted.reset_index(level=0, inplace=True) #이래야 새로 insert 한 index, col 로 접근 가능??
    #melted['exp_id']=melted.index

    handles, labels = axs.get_legend_handles_labels()
    labels = labels[0:len(n_label)] 
    handles = handles[0:len(n_label)]
    for i_l, i_label in enumerate(labels):
        labels[i_l]= i_label + ' (n={})'.format(n_label[i_l])
        #legend.append(str(i_key)+ ' (n={})'.format(len(d_list[i])))
    axs.legend(handles, labels, bbox_to_anchor=(1, 0.9), loc='best')#, borderaxespad=0.)
    #ax.legend(handles = patches, loc='best', edgecolor =None,  bbox_to_anchor=(0.85,0.85)) # upper rightfontsize=13,
    axs.set(#title = j_target,
          ylabel=ylabel,
          ylim=ylim)
#       
    ## export data for statistics
    for ii in melted.groupby('val_type'):
        temp_melted = ii[1].pivot_table(index='exp_id', columns = [target_hue], values='value')#, aggfunc='first')
        out=pd.concat([out, temp_melted], sort=False, axis=1)

    change_width(axs, bar_width/(n_target_group**(2/3))) # bar_width
    
    #np_result = np.sort(out.values, axis=0)
    
    out = out.sort_values(by=list(out.columns))


    return figs, axs, out, df_val
    
def change_width(ax, new_value):
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value
        # change bar width
        patch.set_width(new_value)
        # recenter
        patch.set_x(patch.get_x() + diff * .5)
        
def data_vec(data):  # convert Pandas dataframe into vectors
    n_category = len(data.keys())
    d_list = [0 for i in range(n_category)]
    legend=[]
    for i, i_key in enumerate(data.keys()):
        d_list[i] = np.array(data[i_key])
        d_list[i] = d_list[i][np.isnan(d_list[i])==False]  # remove NaN
        print('n = {}'.format(len(d_list[i])))
        #legend.append(str(int(i_key)) + ' (n={})'.format(len(d_list[i])))
        legend.append(str(i_key)+ ' (n={})'.format(len(d_list[i])))
    return d_list, legend

def calc_group(df): # for cv, mean 
    
    #id = df["batch"].map(str) + df["cage"].map(str)
    #df1=df.iloc[:,]
    #df.insert(0,column='id', value=id)
    
    exp_id = df['exp_type']+'_'+df['batch_cage'].map(str)
    df.insert(0,column='exp_id', value=exp_id)
    
    list_sel = ['exp_id','b_avg', 'ofl_avg', 'ofl_norm', 'ret_avg', 'ret_norm',  'exp_type', 'weight', 'age_ofl']
    df = df[list_sel] # n = 40
    #df.columns # type /  value group / value
    
    #df['group']=df.index.get_value() #.str[0]
    
    m_df=pd.DataFrame([])
    s_df=pd.DataFrame([])
    #cv_df=pd.DataFrame([])
    
    for i in df['exp_type'].unique():
        temp_m = df[df['exp_type']==i].groupby('exp_id').mean()
        temp_m['exp_type']=i
        m_df = m_df.append(temp_m)
        
        temp_s = df[df['exp_type']==i].groupby('exp_id').mean()
        temp_s['exp_type']=i
        s_df = s_df.append(temp_s) # default ddof =1
        #cv_df = s_df/m_df
        
    return m_df, s_df

def fig_set(fig, ax, n_target_x, n_group):
    #plt.setp(ax.get_xticklabels(), visible=False)

    # for multiplot. 추후 업뎃

    #handles, labels = ax.get_legend_handles_labels()
    n_color = n_group#?? 왜 2로 나눴었지? 
    print('n color')
    print(n_color)
    if max([len(i) for i in list_palette])<=3:
        print('already selected')
    else:
        try:
            color_setting(list_palette, i_color=2, n_color =n_color) # 2: pastel  # bar / strip plot 2번 해서, 2로 나눠야 그룹 
        except:
            print('err color', n_color, list_palette)
    ax.grid(False)
    #i_ax.axis('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #plt.setp(axes.get_xticklabels(), visible=False) # x label false
    #plt.setp(axes_list, title='Test')
    
    #fig.show()
    
    #return fig
#    fig.subplots_adjust(bottom=0)
#    fig.subplots_adjust(top=0.9)
#    fig.subplots_adjust(left=0.1)
#    fig.subplots_adjust(right=0.9)
    
def rank2subdom(df): #subdom rank 만들때만 사
    df.loc[(df['rank']<3) & (df['n_mice']==5),'subdom']='dom'
    df.loc[(df['rank']>3) & (df['n_mice']==5),'subdom']='sub'
        
    df.loc[(df['rank']<3) & (df['n_mice']==4),'subdom']='dom'
    df.loc[(df['rank']>2) & (df['n_mice']==4),'subdom']='sub'
    # for demonstartor. n4 only
    df.loc[(df['dem_rank']<3) & (df['n_mice']==4),'dem_subdom']='dom'
    df.loc[(df['dem_rank']>2) & (df['n_mice']==4),'dem_subdom']='sub'
    return df

#######
def calc_indiv(df): #general purpose
    ### 여기서 데이터 기본 처리.
    ###### !! n5 cage 는 4, 5 등-> 3,4 로 변환
#    df.loc[(df['n_mice']==5) & (df['rank']==3),'rank']=np.nan # n5 에서 3등 제거하기 

#    df['rank']=df['rank'].fillna(-1)
#    df['rank']=df['rank'].astype(int)
#    df['rank']=df['rank'].astype(str)
#    df['rank']=df['rank'].replace('-1', np.nan)

    ### temp. caution
    
    ### 1. data trimming
    # 1.1. base freezing 10 < 제외
    df = df[df.b_avg<b_thr]
    #
    if flag_project =='tube':
        exp_id = df['group']+'_'+df['batch_cage'].map(str)+'_'+df['subject'].map(str) # df['exp_tyupe 에서 임시 수정)]
    else:
        exp_id = df['group']+'_'+df['batch_cage'].map(str)+'_'+df.index.map(str)
        
    df.insert(0,column='exp_id', value=exp_id)
    id_group = df['group']+'_'+df['batch_cage'].map(str)
    df.insert(0,column='id_group', value=id_group)
        
    #df['group']=df.index.get_value() #.str[0]
    return df

def main():
    #target_x =  ['ofl_norm']#, 'ret_avg', 'ret_norm']
    path = r'E:\0Temp_data'#\Tube'
    flag_type_2=False
    os.chdir(path)
    df = pd.read_excel(name, sheetname=name_sheet) #, dtype={'batch':int, 'cage':int})
    
    df=calc_indiv(df)
    #list_target = ['wild', 'gerbil']
    """
    d_col = [i for i in df.columns if type(i)!=str] # int col 만 선택.
    df_val = df[d_col]
    
    #d_col.append()
    
    base_m = np.mean(df_val.iloc[:,0:n_base], axis=1)
    ofl_m = np.mean(df_val.iloc[:,n_base:n_total], axis=1)
    ret_m = np.mean(df_val.iloc[:,n_total:], axis=1) 
    
    
    #df.insert(len(df.columns),column='id_original', value = df['subject'])   # backup
    if flag_2phase:
        ofl_e = np.mean(df_val.iloc[:,n_base+1:n_total-1], axis=1)
        ofl_l = np.mean(df_val.iloc[:,n_total-1:n_total+1], axis=1)
        df['ofl_early']=ofl_e-base_m 
        df['ofl_late']=ofl_l
    
    df['b_avg']=base_m
    df['ofl_avg']=ofl_m 
    df['ret_avg']=ret_m 
    df['ofl_norm']=ofl_m- base_m
    df['ret_norm']=ret_m - base_m
    """
    
    
    if flag_special=='crowd_age':
        df = df[df.n_mice >5]
        df.loc[df.exp_type =='n_mice','exp_type']='age'
    elif flag_special=='non-crowd_age':
        df = df[df.n_mice <=5]
    elif flag_special=='n5':
        df = df[df.n_mice==4]
        df.loc[df.exp_type =='n_mice','exp_type']='age'
    
        
    elif flag_special=='mature_n':
        df = df[df.age>=14 ]
        df.loc[df.exp_type =='age','exp_type']='n_mice'
        
    elif flag_special=='immature_n':
        df = df[((df.age<14) & (df.age>9)) ]
        df = df[df.n_mice!=1]
        df.loc[df.exp_type =='age','exp_type']='n_mice'
        
    if flag_group:
        print('grouping start')
        
        if tt == 'age':
        
            df.loc[((df[tt] <=12) & (df[tt]>8)),'new_col'] = '3_adult'
            df.loc[df[tt] <6,'new_col'] = '1_juv'
            df.loc[((df[tt] <=8) & (df[tt]>5)),'new_col'] = '2_juv_mature'
            df.loc[((df[tt] <=14) & (df[tt]>12)),'new_col'] = '4_13-14'
            df.loc[((df[tt] <=16) & (df[tt]>14)),'new_col'] = '5_15-16'
            df.loc[df[tt] >=17,'new_col'] = '6_>17-wk'
            df=df.sort_values(by='new_col')
            
       
        #df= df[df[tt]>8]
        elif tt == 'n_mice':
            df.loc[((df[tt] <=4) & (df[tt]>=3)),'new_col'] = '2_moderate'
            df.loc[df[tt] ==1,'new_col'] = '1_isol'
            df.loc[df[tt] >5,'new_col'] = '3_crowd'        
        
        df[tt]=df['new_col']
    
    if 'rank' in tt:
        
        df =  df[(df['n_mice']==4)]
        df=df.loc[df.ofl_pair.notnull()] # obs / dem pair 경우만 포함
        
        # n5 포함할 경
#        df.loc[(df.n_mice==5) & (df.rank==4),'rank'] =3
#        df.loc[(df.n_mice==5) & (df.rank==5),'rank'] =4
#        df.loc[(df.n_mice==5) & (df.dem_rank==4),'dem_rank'] =3
#        df.loc[(df.n_mice==5) & (df.dem_rank==5),'dem_rank'] =4
#        
        
        col_total_days = [i for i in df.columns if type(i)==int] # win /% 두 종류
        col_total_days = [i for i in col_total_days if i >=10]
        n_day=int(len(col_total_days)/2)
        col_d1 = [i for i in col_total_days if i<100] # tube days
        col_d2 = [i for i in col_total_days if i>=100] # win day
        
        df=df.fillna(0) # nan 처리
        
        df[col_d1]=df[col_d1]+np.random.normal(0,0.1,[df.shape[0],3]) # noise to jitter (rank change plot is prism)
        
        
#        df_3 =  df[(df['stable']=='y')]
#        df_2 =  df[((df['stable']=='yy')|(df['stable']=='y'))]
#        df_un =  df[((df['unstable']=='yy')|(df['unstable']=='y'))]
#        
        print(df['rank'].value_counts())
        #df =  df[((df['n_mice']==4) & (df['chk']!='y')&(df['dem_rank']!=1))] # n 4 만 고려 + 제외 대
        #df =  df[((df['n_mice']==4) & (df['chk']!='y'))]#&(df['diff_rank']!=1))] # n 4 만 고려 + 제외 대
        
        list_w_decay = [w_decay**i for i in range(n_day)][::-1]
        
        df['w_rank']=df[col_d1].dot(list_w_decay)     
        df['w_win']=df[col_d2].dot(list_w_decay)/3     
        grouped = df.groupby('id_group')
        
        # exp_id 는 individual subject

        # 1. rank norm (win )
        df['rank_std'] = df['w_rank']-df['w_rank'].mean()/df['w_rank'].std()
        df['win_std'] = df['w_win']-df['w_win'].mean()/df['w_win'].std()
        df['rank_diff_std'] = df['diff_rank']-df['diff_rank'].mean()/df['diff_rank'].std()
        
        #df=df[((df['w_win']<10) | (df['w_win']>70))]
        #df=df[((df['diff_rank']!=-3) & (df['diff_rank']!=3))]
        #df=df[(df['cage_similar']!='y')]
        #df=df[(df['chk']!='y')]
        cols = ['w_rank','w_win','diff_rank']
        cols_norm = ['rank_norm','win_norm','rank_diff_norm']
        
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(df[cols])
        #df_normalized = pd.DataFrame(np_scaled, columns = cols)
        #df[cols_norm]=np_scaled
        cond_obs_rank = (df['rank']==1) | (df['rank']==4) # 1, 4 rank only
        cond_dem_rank = (df['dem_rank']==1) | (df['dem_rank']==4) # 1, 4 rank only
        
        # 2-day continuous rank condition
        cond_obs_stable = df.stable.str.contains('y') #
        cond_dem_stable = df.dem_stable.str.contains('y') 
        
        # 3-day cont 
        #cond_obs_stable = (df['stable'] =='y')
        #cond_dem_stable = (df['dem_stable']=='y')
        
        
        df.loc[  ((cond_obs_stable  & cond_obs_rank)  &  ~( cond_dem_stable &  cond_dem_rank)),'exp_type_obs']='y'
        df.loc[  (~( cond_obs_stable & cond_obs_rank)  &  (cond_dem_stable &  cond_dem_rank)),'exp_type_dem']='y'
        
        
        #df_obs = df.query(('(rank == 1 | rank == 4 )')
        #diff_a = df[cols_norm].values-df_normalized[cols].values
        
        ds.save_excel(df, 'total_norm')
        
        ###################### for rank 
        #target_group = ['exp_type_obs','exp_type_dem']
        #now_target='w_rank'
        plot_reg(data=df, x_target=now_target, y_target ='ofl_norm' )
        
    elif tt=='n_mice':
        df.loc[df[tt]>5,tt]=6 #6 이상 한 그룹으로 통일
        
    elif tt=='age':
        df.loc[df[tt]>14,tt]=16 # 16주 이상은 한 그룹 통
        df=df[df[tt]>=6] # 6주 부터만 포함
        
    
    else:
        print('temp')
        #df = df.sort_values(by=tt)

    ####### multiple regression
#    df=df[(df.exp_type=='age') | (df.exp_type=='n_cage')]
#    
#        
#    lm = smf.ols(formula='ofl_norm ~ age+ n_cage', data=df).fit()
#    
#    lm.params
#    lm.conf_int()
#    lm.summary()
#    
#    df['n_cage']=(df['n_cage']-df['n_cage'].min())/(df['n_cage'].max()-df['n_cage'].min())      
    
    
#    #### facet grid. data 정리해서 추후 다시
#    fig_facet, ax_facet = plt.subplots(figsize=(fig_size, fig_size*fig_ratio))
#        
#    #fig_set(fig_facet, ax_facet, len(target_x), len(target_group) #/2 ?
#    melted = df.melt(id_vars=['exp_id','exp_type',target_hue], value_vars = target_x, var_name='val_type',value_name='value' ).dropna() ##
#    g=sns.FacetGrid(melted, row='exp_type', hue=target_hue, palette = pastel_rainbow) #...                   hue_order=["Dinner", "Lunch"]
#    g=g.map(sns.stripplot, x='val_type', y='value', data=melted, jitter=True, size= 7,edgecolor = 'gray', linewidth=1, ax=ax, dodge=True)
    #sns.barplot(x='val_type',y='value', data=melted, hue =target_hue, ci=68, errwidth = 1.0, capsize=.03, ax= ax)# ax[j]
    #sns.stripplot(x='val_type',y='value', data=melted, hue = target_hue, jitter=True, size= 7,  
                  
    
#g=g.map(~plt, target_x, , marker:)
    
    
    figs, axs = plt.subplots(ncols=len(target_group))
    for i_g_count, i_g in enumerate(target_group):
        
            
        try: 
            i_axs = axs[i_g_count]
        except:
            i_axs = axs
            
        ii_df = df[df['exp_type']==i_g]
           
        if  ii_df.size ==0:
            ii_df =df[df[i_g]=='y']
            flag_type_2 = True    
            
            
        if flag_type_2 & (i_g_count==1):    
            fig_total, ax_out, np_result, df_raw = plot_bar_scatter_df(ii_df,'dem_rank', target_x, figs, i_axs)
        else:
        
            fig_total, ax_out, np_result, df_raw = plot_bar_scatter_df(ii_df, target_hue, target_x, figs, i_axs)
            
        fig_total, ax_out, np_result, df_raw = plot_bar_scatter_df(ii_df, target_hue, target_x, figs, i_axs)
        #fig_total.tight_layout()
        
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        figs.savefig('{}_bar_scatter_{}-{}.png'.format(i_g,time_now,i_g_count+1), transparent = True, dpi=200)   
        #np.savetxt("{}_result_{}.csv".format(i_g,time_now), np_result, delimiter=",")
        with pd.ExcelWriter("{}_result_{}.xlsx".format(i_g,time_now)) as writer:
            df_raw.to_excel(writer,sheet_name='raw')
            (pd.DataFrame(np_result)).to_excel(writer,sheet_name='ofl_norm')
 



############ old ver. tube only

#df['n_cage']=(df['n_cage']-df['n_cage'].min())/(df['n_cage'].max()-df['n_cage'].min())      

def main_tube():
    #### 무엇을 타겟으로할지!
    
    os.chdir(path)
    df = pd.read_excel(name) #, dtype={'batch':int, 'cage':int})
     
    group_id = df['exp_type']+'_'+df['batch_cage'].map(str)
    df.insert(0,column='group_id', value=group_id)
    
    #### rank 재설정
    ##### rank day : 주의!!! obsdem, fam 은 사용불가. -> ㄱ능하게 고
#    rank_day = 3
#    list_day = ['win_d1','win_d2','win_d3','total']
#    
#    df['rank']=df.groupby('group_id')[list_day[rank_day]].rank(axis=1, method='min', ascending=False)
#    df = df.drop(columns='group_id')
#    if rank_day !=0:
#        print('3-day tube data only')
#    else:
#        print('day1 tube rank')
#       # df = df[df['linear']=='y'] # exclude unstable
    
#################    ### test
    #melted = df.melt(id_vars=['group_id'], value_vars = 'rank', var_name='val_type',value_name='value' )
    
    df_criteria = (df['b_avg']<b_thr)  & (df['n_mice']!=50) & ((df['stable']=='y')|(df['stable']=='y'))&(df['remove']!='y') & (df['linear']!='n') 
    df = df[df_criteria]
    
    df = rank2subdom(df) # designate sub or dom based on rank 
    
    # obs /dem 간의 sub / dom 관계로 비교시
    if target_hue =='obsdem_rank':
    ##### for subdom
        print('Sub x Dom rank combination')
        df = df[df['dem_subdom']!='n']
        id_same = df['subdom'] == df['dem_subdom']
        id_diff = df['subdom'] != df['dem_subdom']
        
        df['obsdem_rank'] = df['subdom']+df['dem_subdom']
        df.loc[id_same, 'relation'] ='same'
        df.loc[id_diff, 'relation'] ='diff'
        
    elif target_hue =='rel_rank':
        
        df['diff_rank'] = (df['dem_rank']-df['rank'])/(df['n_mice']-1)
        
        df.loc[df['diff_rank']>0, 'rel_rank'] = 'Obs > Dem' #'Obs dom to Dem'd
        df.loc[df['diff_rank']<0, 'rel_rank'] = 'Obs < Dem' #'Obs sub do Dem'        
        
    elif target_hue =='comb_rank':
        df=df[df['exp_type']==target_group]
        df=df[df['n_mice']==4]
        df['comb_rank']=df['rank'].astype(int).astype(str)+df['dem_rank'].astype(int).astype(str)
        id_high = df['dem_rank']>df['rank']#df['dem_rank']==1
        id_low = df['dem_rank']<df['rank']#df['dem_rank']==4
        #id_high = df['rank']==1
        #id_low = df['rank']==4
        df.loc[id_low, 'prof_group2'] = 'low'
        df.loc[id_high, 'prof_group2'] = 'high'

        
    elif target_hue =='fix_obs':
        df=df[(((df['n_mice']==4)&(df['rank']==2)|(df['rank']==3)))]# |((df['n_mice']==5)&(df['rank']==3)))]
        #df=df[(df['dem_rank']==1)|(df['dem_rank']==4)] # to exclude 2->3 or 3->2
        id_dom = df['dem_rank']<df['rank']#df['dem_rank']==1
        id_sub = df['dem_rank']>df['rank']#df['dem_rank']==4
        df.loc[id_dom, target_hue] = 'Dom Dem'
        df.loc[id_sub, target_hue] = 'Sub Dem'
    elif target_hue =='fix_dem':
        df=df[(((df['n_mice']==4)&(df['dem_rank']==2)|(df['dem_rank']==3)))]#|((df['n_mice']==5)&(df['dem_rank']==3)))]
        #df=df[(((df['n_mice']==4)&(df['dem_rank']==2)|(df['dem_rank']==3)))]#|((df['n_mice']==5)&(df['dem_rank']==3)))]
        
        id_dom = df['rank']<df['dem_rank']#df['dem_rank']==1
        id_sub = df['rank']>df['dem_rank']#df['dem_rank']==4
        df.loc[id_sub, target_hue] = 'Sub Obs'
        df.loc[id_dom, target_hue] = 'Dom Obs'
    elif target_hue =='rank':#) & (target_group=='tubeOFL')):
        
        df.loc[((df['n_mice']==4)&((df['rank']==2)|(df['rank']==3))), 'rank']=2
        df.loc[((df['n_mice']==5)&((df['rank']==2)|(df['rank']==3)|(df['rank']==4))), 'rank']=2
        df.loc[((df['n_mice']==5)&((df['rank']==5))), 'rank']=4
#        
    df = calc_indiv(df)
    
    figs, axs = plt.subplots(ncols=len(target_group))
    for i_g_count, i_g in enumerate(target_group):
        ii_df = df[df['exp_type']==i_g]
    
        fig_total, ax_out, np_result = plot_bar_scatter_df(ii_df, target_hue, target_x, figs, axs)
        fig_total.tight_layout()
        
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_total.savefig('{}_bar_scatter_{}-{}.png'.format(i_g,time_now,i_g_count+1), transparent = True, dpi=200)    
        np.savetxt("{}_result_{}.csv".format(i_g,time_now), np_result, delimiter=",")
        
    plt.show(block=False)
    #import time
    #time.sleep(5)
    #plt.close('all')
        
if __name__ =='__main__':
    #import gc
    #gc.collect(2)
    #fig.gcf()
    
    if flag_project == 'tube':
        main_tube()
    else:
        main()
### _ 찾으면 직전까지가 날짜. 그다음 batch 찾아서 그 다음숫자가 배치 번호

##### cage 별로 접근할것 따로.
# ex) variability, dob 등?

# 자잘한 raw data  와 avg 등 주로 쓰는 것 따로 저장하기

# 한번 들어가 csv 찾아 OFL obs / dem, ret 폴더 가서 csv로 ret 정보도


#
################ 1. baseline threshold 정하기. peak 값 등.
#base_thr = 10
#df = df[df['baseline']<base_thr]
##df1 = df1[df1['subdom']!=0]
#
### subject id. 
## experiment # batch # cage # no
#sub_list = ['batch','cage','no']
#df.insert(0,column='subject',value='')
##df.drop(columns='23') #제거
#for i in df.index:
#    temp = df.loc[i,:]
#    sub_temp=''
#    subject = [int(temp[j]) for j in sub_list]
#    subject =''.join(str(k) for k in subject)
#    df.loc[i,'subject']=subject    
### df.loc[:,['batch','cage']]
#
## batch / cage
## age cal
#
### rank by win no.
##df['win']
#
#
### sub dom
#df.loc[df['rank']<3,'subdom']='dom'
#df.loc[df['rank']>3,'subdom']='sub'
#df.reindex()
#
##df.reindex(columns=['subject'])
#
#df1 = df[df['baseline']<10]
#df1 = df1[df1['subdom']!=0]
#
#rank_ofl = df1.pivot_table(index=df1.index, columns=['subdom'], values = ['ofl'])
##rank_ofl = df.pivot_table(index=df.index, columns=['win'], values = ['ofl'])