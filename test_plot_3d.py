# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 19:21:30 2018

@author: 2014_Joon_IBS
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
plt.close()
cur_path = r'E:\0Temp_data\Tube'
os.chdir(cur_path)
#data = 'total_obsdem_fam.xlsx'
data = 'on_total_test.xlsx' #'temp_total_all.xlsx'
df = pd.read_excel(data)
#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/iris.csv')
#df = df[df['fam']=='fam']
df = df[['subject','b_avg','ofl_avg','ret_avg', 'exp_type','rank','subdom']]
df = df.reset_index(drop=True)
# Dataset
#df=pd.DataFrame({'X': range(1,101), 'Y': np.random.randn(100)*15+range(1,101), 'Z': (np.random.randn(100)*15+range(1,101))*2 })
plt.close()
# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(df['X'], df['Y'], df['Z'], c='skyblue', s=60)

group =  'exp_type'
#df = np.array(df)

#
#for i in range(len(df)): #plot each point + it's index as text above
#    #ax.scatter(df[i,0],df[i,1],df[i,2],c='skyblue', s=60) 
#    #ax.text(df[i,0],df[i,1],df[i,2],  '%s' % (str(i)), size=10, zorder=1,  color='k') 
# 
#    ax.scatter(df['b_avg'][i],df['ofl_norm'][i],df['ret_norm'][i],cmap='jet', s=60) #facecolors="white",edgecolors="blue"c='skyblue',
#    ####ax.text(df['b_avg'][i],df['ofl_avg'][i],df['ret_avg'][i],  '%s' % (str(i+1)+'_'+df['subject'][i]), size=15, zorder=1,  color='k')


##### color clustering
c_lst = [plt.cm.rainbow(a) for a in np.linspace(0.0, 1.0, len(set(df[group])))]


for j, j_df in enumerate(df.groupby(group)):
    for j_row in j_df[1].iterrows():
        g_i = j_row[0]
        g_df = j_row[1]
        ax.scatter(g_df['b_avg'], g_df['ofl_avg'], g_df['ret_avg'], color = c_lst[j], alpha=0.5, s=60 ) #, label='{} (n={})'.format(j_df[0], len(j_df[1]))
        ax.text(g_df['b_avg'],g_df['ofl_avg'],g_df['ret_avg'],  '%s' % (str(g_i+1)+'_'+str(g_df['subject'])), size=10, zorder=1,  color='k')
        
   
    #ax.scatter(j_df[1]['b_avg'], j_df[1]['ofl_avg'], j_df[1]['ret_avg'], color = c_lst[j], label='{} (n={})'.format(j_df[0], len(j_df[1])), alpha=0.5, s=60 )
    #ax.text(j_df[1]['b_avg'], j_df[1]['ofl_avg'], j_df[1]['ret_avg'], color = c_lst[j], label='{} (n={})'.format(j_df[0], len(j_df[1])), alpha=0.5, s=60 )
    #ax.text(df['b_avg'][i],df['ofl_avg'][i],df['ret_avg'][i],  '%s' % (str(i+1)+'_'+df['subject'][i]), size=15, zorder=1,  color='k')
    
plt.legend()

#plt.colorbar()
ax.set_xlabel('baseline (%)')
ax.set_ylabel('OFL (%)')
ax.set_zlabel('Retrieval (%)')
#ax.view_init(30, 185)

plt.show()