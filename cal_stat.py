# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:49:29 2018

@author: 2014_Joon_IBS
"""
from scipy import stats

# make stars to show statistical power
def stars(p): 
   if p < 0.0001:
       return "****"
   elif (p < 0.001):
       return "***"
   elif (p < 0.01):
       return "**"
   elif (p < 0.05):
       return "*"
   else:
       return "-"
   
# calculate t-test. according to the normality & variance test, non-parametric test will be given. two-way is default

def anova(data):
    print('anova')
    
def t_test(x,y):
    # normality test. p>0.05 means fail
    _, p_x = stats.shapiro(x)
    _, p_y = stats.shapiro(y)
    # equal variance test. p <=0.05 means fail
    _, p_var = stats.levene(x,y) 
    
    if (p_x or p_y > 0.05) or p_var <=0.05 :  # normality fail
        print('Non-parametric test is needed. MannWhitney U test.\n')
        statistic, p_final = stats.mannwhitneyu(x,y, alternative='two-sided')
    else:
        print('T-test\n')
        statistic, p_final = stats.ttest_ind(x,y)
        
    print('Result. Statistic: {}\n p-value: {}'.format(statistic, p_final))
    return statistic, p_final

def main():
    print('Stat lib')    

if __name__ == '__main__':
    main()
    