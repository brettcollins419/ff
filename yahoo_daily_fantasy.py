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
import re
import socket
import copy

#%% FUNCTIONS


fantasyProsDict = {
        'QB' : {'label' : 'QB', 'column' : 'Quarterbacks'}
        , 'RB' : {'label': 'RB', 'column' : 'Running Backs'}
        , 'TE' : {'label': 'TE', 'column' : 'Tight Ends'}
        , 'WR' : {'label': 'WR', 'column' : 'Wide Receivers'}
        , 'DST' : {'label': 'DEF', 'column' : 'Team DST'}
        }


def fantasyProsRankingsDataLoad(position
                                , fantasyProsDict = fantasyProsDict
                                , fileName = 'data\\FantasyPros_2019_Week_2_{}_Rankings.csv'
                                ):
    
    '''Read data into a dataframe and append column labeling the player position'''
    
    data = pd.read_csv(fileName.format(position))
    

    # Filter empy Rows
    data = data[data['Rank'] > 0]

    # Add position label
    data.loc[:, 'position'] = fantasyProsDict[position]['label']
    
    # Rename Column
    data.rename(columns = {fantasyProsDict[position]['column'] : 'player'},
                           inplace = True)

    
    # Add Team for DST
    if position == 'DST':
        data.loc[:, 'Team'] =[
                team.split('(')[1].strip(')') for team in data['player']
                ]
    
        data.loc[:, 'player'] = data['Team']
    
    return data


def fantasyProsProjectionsDataLoad(position
                                , fantasyProsDict = fantasyProsDict
                                , fileName = 'data\\FantasyPros_Fantasy_Football_Projections_{}.csv'
                                ):
    
    '''Read data into a dataframe and append column labeling the player position'''
    
    data = pd.read_csv(fileName.format(position))
    

    # Filter empy Rows
    data = data[data['FPTS'] != None]

    # Add position label
    data.loc[:, 'position'] = fantasyProsDict[position]['label']
    
    # Rename Column
    data.rename(columns = {fantasyProsDict[position]['column'] : 'Player'},
                           inplace = True)

    
    # Add Team for DST
    if position == 'DST':
        data.loc[:, 'Team'] =[
                team.split('(')[1].strip(')') for team in data['Player']
                ]
    
        data.loc[:, 'Player'] = data['Team']
    
    return data


def fpRankingsKeyGen(keyList):
    '''Generate key for fpRankings using first letter of first name, 
    full last name, team, and position except defense is defense.'''
    
    if keyList[-1] == 'DEF':
        key = keyList[-2]
    
    else:
        key = re.sub('\W', '',
              ''.join((keyList[0].split(' ')[0][0]
                      , keyList[0].split(' ')[1]
                      , keyList[1]
                      , keyList[2])).upper()
              )
              
    return key


def salaryKeyGen(keyList):
    '''Generate key for fpRankings using first letter of first name, 
    full last name, team, and position except defense is defense.'''
    
    if keyList[-1] == 'DEF':
        key = keyList[-2]
    
    else:
        key = re.sub('\W', '', 
               ''.join((keyList[0][0]
                       , keyList[1].split(' ')[0]
                       , keyList[2]
                       , keyList[3])).upper()
               )
               
    return key

#%% LOAD DATA


# Working Directory Dictionary
pcs = {
    'WaterBug' : {'wd':'C:\\Users\\brett\\Documents\\ff',
                  'repo':'C:\\Users\\brett\\Documents\\march_madness_ml\\march_madness'},

    'WHQPC-L60102' : {'wd':'C:\\Users\\u00bec7\\Desktop\\personal\\ff',
                      'repo':'C:\\Users\\u00bec7\\Desktop\\personal\\ff'},
                      
    'raspberrypi' : {'wd':'/home/pi/Documents/ff',
                     'repo':'/home/pi/Documents/ff'},
                     
    'jeebs' : {'wd': 'C:\\Users\\brett\\Documents\\ff\\',
               'repo' : 'C:\\Users\\brett\\Documents\\ff\\'}
    }

# Set working directory & load functions
pc = pcs.get(socket.gethostname())

del(pcs)



# Set up environment
os.chdir(pc['repo'])


data = pd.read_csv('data\\Yahoo_DF_player_export_w2.csv')


positions = ['QB', 'TE', 'RB', 'DEF', 'WR']


for position in positions:
    data.loc[:,position] = (data['Position'] == position) * 1
    
# # of players required for each position
positionLimit = {'QB':1
                 , 'TE':2
                 , 'RB':2
                 , 'DEF':1
                 , 'WR':3
                 }

# Player status' to exclude
statusExclude = ['IR', 'SUSP', 'O', 'Q']


# Filter only eligble players
dataInput = copy.deepcopy(
        data[data['Injury Status'].map(lambda s: s not in statusExclude)]
        )


# Convert to dictionary for LP
dataInputDict = (
        dataInput.set_index('Id')
        [['Salary', 'FPPG'] + positions].to_dict('index')
        )


#%% LP SETUP
## ############################################################################

# Setup LP Problem
budget = 200

prob = pulp.LpProblem('The Best Team', pulp.LpMaximize)

# Define player Variables
playerVars = pulp.LpVariable.dicts('ID', dataInputDict.keys(), cat = 'Binary')

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


# Position Limits (not Flex)
for position in ['QB', 'DEF']:
    prob += pulp.lpSum(
            [(playerVars[i] * dataInputDict[i][position]) 
            for i in dataInput['Id']]
            ) == positionLimit[position]


for position in ['RB', 'WR', 'TE']:
    prob += pulp.lpSum(
            [(playerVars[i] * dataInputDict[i][position]) 
            for i in dataInput['Id']]
            ) >= positionLimit[position]

    prob += pulp.lpSum(
            [(playerVars[i] * dataInputDict[i][position]) 
            for i in dataInput['Id']]
            ) <= (positionLimit[position] + 1)


# Team Size Limit
prob += pulp.lpSum([playerVars[i] for i in list(dataInput['Id'])]) == 9

#%% SOLVE LP
## ############################################################################
    
prob.writeLP('teamOptimization.lp')
prob.solve()
    
print("Status:", pulp.LpStatus[prob.status])



#%% FINAL TEAM

finalTeam = (
        dataInput.set_index('Id')
        .loc[filter(lambda k: playerVars[k].varValue == 1,
                    playerVars.keys()), :][
        ['First Name', 'Last Name', 'Position', 
         'Team', 'Opponent', 'Salary', 'FPPG']]
        )


finalTeam[['FPPG', 'Salary']].sum()


#%% FANTASY PROS DATA
## ############################################################################

# Load and concat data
fpRankings = pd.concat([fantasyProsRankingsDataLoad(position) 
                        for position in fantasyProsDict.keys()
                        ], sort = True)

fpProjections = pd.concat([fantasyProsProjectionsDataLoad(position) 
                        for position in fantasyProsDict.keys()
                        ], sort = True)
        
# Rename JAC to JAX
fpRankings.loc[fpRankings['Team'] == 'JAC', 'Team'] = 'JAX'

# Generate key for rankings data
fpRankings.loc[:, 'key'] = list(map(lambda keyList: 
    fpRankingsKeyGen(keyList)
    , fpRankings[['player', 'Team', 'position']].values.tolist()
    ))

    
    
# Generate Key for salary data
dataInput.loc[:, 'key'] = list(map(lambda keyList: 
    salaryKeyGen(keyList)
    , dataInput[['First Name', 'Last Name', 'Team', 'Position']].values.tolist()
    ))




x = dataInput.set_index('key').merge(
        fpRankings.set_index('key')[['Avg', 'Best', 'Worst', 'Rank','player']]
        , how = 'left'
        , left_index = True
        , right_index = True
        )

x.to_csv('fpRankings_validation.csv')

#dataInput.loc[:, 'avgRanking'] = fpRankings.set_index('key')['Avg']
