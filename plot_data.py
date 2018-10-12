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
bar_width = .5

params = {
   'axes.labelsize': 8,
   'font.size': 18,
   'legend.fontsize': 10,
   'legend.frameon':False,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [4, 6] # instead of 4.5, 4.5
   }

plt.rcParams.update(params)


def color_setting(n_color=2):
    sns.set_style('whitegrid')
    sns.set_color_codes()
    
    ###### customize color palette
    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    beach_towel = ['#fe4a49', '#2ab7ca', '#fed766', '#e6e6ea', '#f4f4f8']
    pastel_rainbow = ['#a8e6cf','#dcedc1','#ffd3b6','#ffaaa5', '#ff8b94']
                      
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
                    '#7038F8',  # Dragon
                    ]
    
    list_palette = [flatui,
                 beach_towel,
                 pastel_rainbow,
                 pkm_color]                      
    
    if n_color <=3:  # color 종류 별로 안써도 되는 경우
        for i, i_p in enumerate(list_palette):
            list_palette[i] = i_p[::2]  #홀수 값만 취함. 
    #### 
    sns.set_palette('hls',n_color) # Reds
    sns.set_palette(list_palette[2], n_color)
    current_palette = sns.color_palette()
    #sns.palplot(current_palette)
    ###
    sns.set_context('poster', font_scale=0.8)
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
    fig.savefig('time_series.png', transparent = True, dpi=300)
    fig.show()
        
def plot_bar_scatter(data, p, current_palette, labels):
      
    patches = legend_patch(current_palette, labels)   # to add legend
   
    fig, ax = plt.subplots( figsize=(5,6))
    
    sns.barplot(data=data, ci=68, errwidth = 2, capsize=.05, ax= ax)
    sns.stripplot(data=data, jitter=True, size= 7,  
                  edgecolor = 'gray', linewidth=1, ax=ax)#,color="0.4")
    ax.grid(False)
    
    ax.set(title = 'Social rank vs OFL',
          ylabel='avg OFL (%)',
          #xlabel='trial',
          ylim=[0,30])
    
    ax.legend(handles = patches, loc='best', edgecolor =None, fontsize=13, bbox_to_anchor=(0.7,0.7)) # upper right
    #ax.axis('equal')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    change_width(ax, bar_width) # bar_width
    add_star(ax, p, data)
    
    fig.tight_layout()
    fig.show()
    fig.savefig('bar_scatter.png', transparent = True, dpi=300)

def add_star(ax, p_value, df):
    
    y_max = np.max(np.array(df.max()))
    y_min = np.min(np.array(df.min()))
    s = cs.stars(p_value)
    print(s)
    ax.annotate("", xy=(0, y_max+1), xycoords='data',
               xytext=(1, y_max+1), textcoords='data',
               arrowprops=dict(arrowstyle="-", ec='black',#'#aaaaaa',
                               connectionstyle="bar,fraction=0.2"))
    
    # fig size 5 -> 
    p = 'p = {:.2f}'.format(p_value)
    
    ax.text(0.5, y_max + abs(y_max - y_min)*0.3, s, # star
           horizontalalignment='center',
           verticalalignment='center')
    
    ax.text(0.5, y_max + abs(y_max - y_min)*0.05, p, # p value
           horizontalalignment='center',
           verticalalignment='center')


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
    name = '/python/data/dom_sub.xlsx'
    df = pd.read_excel(name)
    #obs, dem = data_ofl_read()

    ## Statistics. T-test 
    d_list, legend = ds.data_vec(df)
    x = d_list[0]
    y = d_list[1]
    _, p = cs.t_test(x,y)
    
    ## plot
    current_palette = color_setting(2) # 2 가지 color plot 준비

    plot_bar_scatter(df, p, current_palette, legend) # bar scatter plot
    
    # 2. time series plot
    path_ofl = '/python/data/ofl/'
    total = ds.data_ofl_read(path_ofl)
    plot_time_series(total)
    
if __name__ == '__main__':
    main()
    

    
    
    
    
    

    
