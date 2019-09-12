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


positions = ['QB', 'TE', 'RB', 'DEF', 'WR']

for position in positions:
    data.loc[:,position] = (data['Position'] == position) * 1
    

data.loc[:,'FLEX'] = data[['TE', 'RB', 'WR']].sum(axis = 1)
    
positionLimit = {'QB':1
                 , 'TE':2
                 , 'RB':3
                 , 'DEF':1
                 , 'WR':4
#                 , 'FLEX':4
                 }

statusExclude = ['IR', 'SUSP', 'O', 'Q']



dataInput = data[data['Injury Status'].map(lambda s: s not in statusExclude)]


dataInputDict = (
        data.set_index('Id')
        [['Salary', 'FPPG'] + positions].to_dict('index')
        )


#%% LP SETUP
## ############################################################################

# Setup LP Problem
budget = 200

prob = pulp.LpProblem('The Best Team', pulp.LpMaximize)

# Define player Variables
playerVars = pulp.LpVariable.dicts('ID', dataInput['Id'], 0, 1, cat = 'Integer')

# Add objective of maximizing FPPG
prob += pulp.lpSum(
        [playerVars[i]*dataInputDict[i]['FPPG'] 
        for i in dataInput['Id']]
        )

# Salary Cap Constraint
prob += pulp.lpSum(
        [(playerVars[i] * dataInputDict[i]['Salary']) 
        for i in dataInput['Id']]
        ) <= budget


# Position Limits
for position in positions:
    prob += pulp.lpSum(
            [(playerVars[i] * dataInputDict[i][position]) 
            for i in dataInput['Id']]
            ) <= positionLimit[position]


# Team Size Limit
prob += pulp.lpSum([playerVars[i] for i in list(dataInput['Id'])]) == 9

#%% SOLVE LP
## ############################################################################
    
prob.writeLP('teamOptimization.lp')
prob.solve()
    
print("Status:", pulp.LpStatus[prob.status])



#%% FINAL TEAM

finalTeam = dataInput[[p.varValue == 1 for p in prob.variables()]]


finalTeam[['First Name', 'Last Name', 'Position', 'FPPG', 'Salary']].sort_values('Position')

finalTeam[['FPPG', 'Salary']].sum()

