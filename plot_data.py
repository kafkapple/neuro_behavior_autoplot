# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 18:49:46 2018

@author: 2014_Joon_IBS
"""

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

#os.chdir('e:')
#path='/0Temp_data/20180911_OFL_after_tube_n3_batch1/'
#os.chdir(path)

#name = 'freeze_20180911_OFL_after_tube_n3_batch1'
bar_width = .7
dpi=200
params = {
   'axes.labelsize': 18,
   'font.size': 18,
   'legend.fontsize': 10,
   'legend.frameon':False,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'text.usetex': False,
   'figure.figsize': [4, 6] # instead of 4.5, 4.5
   }

plt.rcParams.update(params)
title = 'Retrieval (after 1-day)'

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
beach_towel = ['#fe4a49', '#2ab7ca', '#fed766', '#e6e6ea', '#f4f4f8']
pastel_rainbow = ['#a8e6cf','#dcedc1','#ffd3b6','#ffaaa5', '#ff8b94']
#pastel_rainbow = ['#ffd3b6','#ffaaa5', '#ff8b94']
#flatui = [flatui[0], flatui[3]]
pkm_color = ['#78C850',  # Grass
                '#F08030',  # Fire
                '#6890F0',  # Water
                '#A8B820',  # Bug
                '#A8A878',  # Normal
                '#A040A0',  # Poison
                '#F8D030',  # Electric
                '#E0C068',  # Ground
                '#EE99AC',  # Fairy
                '#C03028',  # Fighting
                '#F85888',  # Psychic
                '#B8A038',  # Rock
                '#705898',  # Ghost
                '#98D8D8',  # Ice
                '#7038F8'  # Dragon
                ]

list_palette = [flatui,
                 beach_towel,
                 pastel_rainbow,
                 pkm_color]           

def color_setting(list_palette=list_palette, n_color=2, color_i = 2):
    sns.set_style('whitegrid')
    sns.set_color_codes()
    
    ###### customize color palette
    
    if n_color <=3:  # color 종류 별로 안써도 되는 경우
        for i, i_p in enumerate(list_palette):
            list_palette[i] = i_p[::2]  #홀수 값만 취함. 
    
    #### 
    #sns.set_palette('hls',n_color) # Reds
    sns.set_palette(list_palette[color_i], n_color)
    sns.set_palette([flatui[0], flatui[3]], n_color)
    current_palette = sns.color_palette()
    ##sns.palplot(current_palette)
    ###
    sns.set_context('talk', font_scale=1)#, rc={"lines.linewidth": 2}) # “paper”, “talk”, and “poster”, which are version of the notebook parameters scaled by .8, 1.3, and 1.6,
    return current_palette

def legend_patch(current_palette, labels):
    patches = []
    for i, _ in enumerate(labels):
        patch_i = mpatches.Patch(color=current_palette[i], label=labels[i])
        patches.append(patch_i)
    return patches

# Fixing bug
def cat_plot(data):
    fig_size=(7,5)
    fig, ax = plt.subplots(figsize=fig_size)#figsize=(10,4))
    
    sns.catplot(
                #palette={"male": "g", "female": "m"},
                x='timebin', y='freezing', hue='obsdem',
                markers=["^",'o'], linestyles=["-",'--'],linewidth=0.1,
                kind="point", data=data, ci=68, aspect=1, scale=1, legend=True, ax=ax)
    sns.set_style('ticks',rc = {"lines.linewidth":0.1,"xtick.major.size": 0.1, "ytick.major.size": 1})
    
# plot time series data
def plot_time_series(data):
    fig_size=(7,5)
    ylim_max = 30 # default ylim max value
    
    max_freezing = data.freezing.max(axis=0) # get the max value
    if max_freezing >ylim_max:
        ylim_max = max_freezing
    fig, ax = plt.subplots(figsize=fig_size)
   
    sns.lineplot(x='timebin', y='freezing', data=data, hue='obsdem',markers=True, ax=ax, ci=68 , err_style = 'bars')
    #sns.set_style('ticks',rc = {"lines.linewidth":1,"xtick.major.size": 1, "ytick.major.size": 1})
    
    ax.set(title = 'OFL',
          ylabel='freezing(%)',
          xlabel='(min)',
          ylim=[0,ylim_max])
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
#    for l in ax.lines:
#        print(l.get_linewidth())
#        #plt.setp(l,linewidth=5, ax=ax)
    
    fig.tight_layout()
    fig.savefig('time_series.png', transparent = True, dpi=200)
    fig.show()
       
def plot_bar_scatter(i=0, data=[], current_palette=[], labels=[],p=False, flag_bar_only=False, violin=False):
    data = data.dropna()
    title = 'OFL'
    ylim = np.max([m.max() for m in data.values])
    ylim=20
    ylabel='avg Freezing (%)'
    patches = legend_patch(current_palette, labels)   # to add legend
    fig, ax = plt.subplots( figsize=(5,6))
    
    if flag_bar_only: # no error bar. just simple bar plot.    
        ax.bar(np.arange(len(data)),data,color=beach_towel,edgecolor='black')
    else:
        sns.stripplot(data=data, jitter=True, size= 5,  
                  edgecolor = 'gray', linewidth=1, ax=ax)#,color="0.4")
        ax.legend(handles = patches, loc='best', edgecolor =None, fontsize=13, bbox_to_anchor=(0.8,0.7)) # upper right
        if violin:
            sns.violinplot(data=data,  ax= ax) #ci=68, errwidth = 2, capsize=.05,    
        else:
            sns.barplot(data=data, ci=68, errwidth = 2, capsize=.05, ax= ax)
        
    ### 공통 설정
    #ax.axis('equal')
    change_width(ax, bar_width) # bar_width
    ax.set(title = title,
              ylabel=ylabel,
              #xlabel='trial',
              ylim=[0,ylim+10])
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    change_width(ax, bar_width) # bar_width
    
    if p: # p 값 있는 경우, 통계 정보 추가
        add_star(ax, p, data)
    
    fig.tight_layout()
    fig.show()
    fig.savefig('bar_scatter_{}.png'.format(i), transparent = True, dpi=dpi)
    
def add_star(ax, p_value, df):
    
    y_max = np.max(np.array(df.max()))
    y_min = np.min(np.array(df.min()))
    s = cs.stars(p_value)
    print(s)
    ax.annotate("", xy=(0, y_max+1), xycoords='data',
               xytext=(1, y_max+1), textcoords='data',
               arrowprops=dict(arrowstyle="-", ec='black',#'#aaaaaa',
                               connectionstyle="bar,fraction=0.2"))
  
    if p_value < 0.0001:
        p = 'p < 0.0001'
    else:
        p = 'p = {:.4f}'.format(p_value)
    
    ax.text(0.5, y_max + abs(y_max - y_min)*0.3, s, # star
           horizontalalignment='center',
           verticalalignment='center')
    
    ax.text(0.5, y_max + abs(y_max - y_min)*0.05, p, # p value
           horizontalalignment='center',
           verticalalignment='center', fontstyle='italic')
    #ax.text.set_fontstyle('italic')


# change width of bar graph
def change_width(ax, new_value):
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value
        # change bar width
        patch.set_width(new_value)
        # recenter
        patch.set_x(patch.get_x() + diff * .5)
        
def main():
    #name = '/python/data/dom_sub.xlsx'
    os.chdir('/python/data/tube/')
    #name = 'g2b_fear_ofl.xlsx'
    name = '/python/data/tube/total_tube_ofl.xlsx'
    #name = '/python/data/ofl_rank_anova3.csv'
    df = pd.read_excel(name)
    
    n_list = [5, 5, 5, 4, 5, 5]
    idx = 0
    data = df[df.keys()[2]]
    tp=[]
    c_i=0
    print(data)
    for i in n_list:
        print(i)
        sort= data[idx:idx+int(i)].sort_values(ascending=False)
        print(sort)
        tp.append(sort)
        idx+=int(i)
        print(idx,c_i)
        #plot_bar_multi(c_i, sort)
        c_i+=1
        
        
    # single plot
    #current_pal = pkm_color
    #df_cur = df.iloc[:,3]
    if len(df.shape) ==1:

        print('only one plot')
        legend = ['OFL (n={})'.format(df.shape[0])]
        n_class = len(legend)
        current_pal = color_setting(n_color=n_class) # class 종류 color plot 준비
        plot_bar_scatter(i=0, data = df, current_palette = current_pal, labels=legend, p=False, )
    else:
        n_col = df.shape[1]
        n_pair = 2
        n_plots = int(n_col/n_pair)
        for i in range(n_plots):
            df_cur = df.iloc[:,i*n_pair:i*n_pair+2]
            
            d_list, legend = ds.data_vec(df_cur)
            x = d_list[0]
            y = d_list[1]
            _, p = cs.t_test(x,y)
            n_class = len(legend)
            current_pal = color_setting(n_color = n_class, color_i=0) # class 종류 color plot 준비
            
            current_pal = [flatui[0], flatui[3]]

            plot_violin_scatter(i, df_cur, p, current_palette = current_pal, labels = legend) # bar scatter plot
            #plot_bar_scatter(i, df_cur, p, current_palette, legend)
    #obs, dem = data_ofl_read()

    ## Statistics. T-test 
    
    
    ## plot
   
    #plot_bar_scatter_multi(df, current_palette, legend) # bar scatter plot
    
    # 2. time series plot
    path_ofl = '/python/data/ofl/'
    list_csv = glob.glob(path_ofl+'*.csv')
    
    total = ds.data_ofl_read(list_csv[0])
    #plot_time_series(total)
    
if __name__ == '__main__':
    main()
    

    
    
    
    
    

    
