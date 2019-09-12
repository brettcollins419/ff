# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:00:42 2019

@author: u00bec7
"""

## Packages


import pandas as pd
import os
import numpy as np
import pulp

#%% FUNCTIONS



#%% LOAD DATA


os.chdir('C:\\Users\\u00bec7\\Desktop\\personal\\ff\\data')

data = pd.read_csv('Yahoo_DF_player_export_w2.csv')



for position in ['QB', 'TE', 'RB', 'DEF', 'WR']:
    data.loc[:,position] = (data['Position'] == position) * 1
    

data.loc[:,'FLEX'] = data[['TE', 'RB', 'WR']].sum(axis = 1)
    
positionLimit = {'QB':1
                 , 'TE':1
                 , 'RB':2
                 , 'DEF':1
                 , 'WR':3
                 , 'FLEX':1
                 }

statusExclude = ['IR', 'SUSP', 'O', 'Q']



dataInput = data[data['Injury Status'].map(lambda s: s not in statusExclude)]


dataInputDict = dataInput.set_index('Id').to_dict('index')


# Setup LP Problem

prob = pulp.LpProblem('The Best Team', pulp.LpMaximize)

# Define player Variables
playerVars = pulp.LpVariable.dicts('ID', list(dataInput['Id']), cat = 'Binary')



