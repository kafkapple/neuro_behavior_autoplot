# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 12:45:04 2018

@author: 2014_Joon_IBS
"""


total_data = ['no', '132', 'here123 523', '2111', '3df3', 'xy132 242'] # total data list
property_list = [ 'here', 'xy'] # 찾고 싶은 속성 리스트
total_result = []

for i_data in total_data: # 전체 데이터 하나씩 탐색하면서
    for i_p in property_list: # 찾고싶은 속성 하나하나마다
        if i_data.find(i_p)==-1: # 찾아서 없으면 no 출력
            print('no')
        else: # 있으면
            print('Found! ')
            #result  = [s for s in i_data.split()] #
            #print(result)
            result  = [int(s) for s in i_data if s.isdigit()] # 해당 키워드 있는 라인에서, 숫자인 부분 찾아 정수로 변환하여 리스트로 저장
            total_result.append(result)
            
print(total_result)