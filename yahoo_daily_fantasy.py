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

# Dictionary for defense for projections
defenseDict = {
        'Arizona Cardinals' : 'ARI'
        , 'Atlanta Falcons' : 'ATL'
        , 'Baltimore Ravens' : 'BAL'
        , 'Buffalo Bills' : 'BUF'
        , 'Carolina Panthers' : 'CAR'
        , 'Chicago Bears' : 'CHI'
        , 'Cincinnati Bengals' : 'CIN'
        , 'Cleveland Browns' : 'CLE'
        , 'Dallas Cowboys' : 'DAL'
        , 'Denver Broncos' : 'DEN'
        , 'Detroit Lions' : 'DET'
        , 'Green Bay Packers' : 'GB'
        , 'Houston Texans' : 'HOU'
        , 'Indianapolis Colts' : 'IND'
        , 'Jacksonville Jaguars' : 'JAX'
        , 'Kansas City Chiefs' : 'KC'
        , 'Los Angeles Chargers' : 'LAC'
        , 'Los Angeles Rams' : 'LAR'
        , 'Miami Dolphins' : 'MIA'
        , 'Minnesota Vikings' : 'MIN'
        , 'New England Patriots' : 'NE'
        , 'New Orleans Saints' : 'NO'
        , 'New York Giants' : 'NYG'
        , 'New York Jets' : 'NYJ'
        , 'Oakland Raiders' : 'OAK'
        , 'Philadelphia Eagles' : 'PHI'
        , 'Pittsburgh Steelers' : 'PIT'
        , 'San Francisco 49ers' : 'SF'
        , 'Seattle Seahawks' : 'SEA'
        , 'Tampa Bay Buccaneers' : 'TB'
        , 'Tennessee Titans' : 'TEN'
        , 'Washington Redskins' : 'WAS'
        }


def fantasyProsRankingsDataLoad(position
                                , week
                                , fantasyProsDict = fantasyProsDict
                                , fileName = 'data\\FantasyPros_2019_Week_{}_{}_Rankings.csv'
                                ):
    
    '''Read data into a dataframe and append column labeling the player position'''
    
    data = pd.read_csv(fileName.format(week, position))
    

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
                                , week
                                , fantasyProsDict = fantasyProsDict
                                , defenseDict = defenseDict
                                , fileName = 'data\\FantasyPros_Fantasy_Football_Projections_{}_w{}.csv'
                                ):
    
    '''Read data into a dataframe and append column labeling the player position'''
    
    data = pd.read_csv(fileName.format(position, week))
    

    # Filter empy Rows
    data = data[[not i for i in np.isnan(data['FPTS'])]]

    # Add position label
    data.loc[:, 'position'] = fantasyProsDict[position]['label']
    
    # Rename Column
    data.rename(columns = {fantasyProsDict[position]['column'] : 'Player'},
                           inplace = True)

    
    # Add Team for DST
    if position == 'DST':
        data['Team'] = [defenseDict.get(player) for player in data['Player']]
    
        data.loc[:, 'Player'] = data['Team']
    
    
    return data[['Player', 'Team', 'position', 'FPTS']]



def fantasyProsAllProjectionsDataLoad(position
                                , week
                                , fantasyProsDict = fantasyProsDict
                                , defenseDict = defenseDict
                                , fileName = ('data\\FantasyPros_Fantasy_'
                                              'Football_Projections_{}_'
                                              'high_low_w{}.csv'
                                              )
                                ):
    
    '''Load fantansy pros projections with high, low, and averages.
        Pivot high, low, and avg into single record for each player
        and estimate std. dev.
        
        Return subset dataframe of just FPTS.
    '''
    
    # Load data
    data = pd.read_csv(fileName.format(position, week))
    
    # Filter empy Rows
    data = data[[not i for i in np.isnan(data['FPTS'])]]
    
    # Forward fll empty player fields
    data['Player'].fillna(method = 'ffill', inplace = True)
    
    
    # Create key for merging avg, high, and low data
    data.loc[:, 'key'] = list(map(lambda r: r//3, range(data.shape[0])))
    
    
    
    # Rename Column
    data.rename(columns = {fantasyProsDict[position]['column'] : 'Player'},
                           inplace = True)
    
    
    # Add Team for DST
    if position == 'DST':
        data['Player'] = [defenseDict.get(player) for player in data['Player']]
    
        data.loc[:, 'Team'] = [
                team[0] if team[1] not in ('high', 'low') else team[1]
                for team in data[['Player', 'Team']].values.tolist()]
    
    
    # Add position label
    data.loc[:, 'position'] = fantasyProsDict[position]['label']
    
    
    
    # Split data into avg, high, and low
    dataAvg = copy.copy(data.loc[
            [k not in ('high', 'low') for k in data['Team'].values.tolist()]
            , :]
            )
    
    dataHigh = copy.copy(
            data.loc[[k == 'high' for k in data['Team'].values.tolist()], :]
            )
    
    dataLow = copy.copy(
            data.loc[[k == 'low' for k in data['Team'].values.tolist()], :] 
            )
        
    
    # Rename high and low columns
    dataHigh.rename(columns = {
            k : '{}_high'.format(k) for k in 
                list(filter(lambda col: col not in ('Player', 'Team', 'key', 'position')
                , data.columns))
            }
            , inplace = True
            )
    
    dataLow.rename(columns = {
            k : '{}_low'.format(k) for k in 
                list(filter(lambda col: col not in ('Player', 'Team', 'key', 'position')
                , data.columns))
            }
            , inplace = True
            )
    
    
    # Combine high, low, and average projections into single record
    data = (dataAvg.set_index('key')
                   .merge(dataLow.set_index('key')
                                 .drop(['Player', 'Team', 'position'], axis = 1)
                         , how = 'inner'
                         , left_index = True
                         , right_index = True)
                   .merge(dataHigh.set_index('key')
                                 .drop(['Player', 'Team', 'position'], axis = 1)
                         , how = 'inner'
                         , left_index = True
                         , right_index = True)               
                   )

    return data[['Player', 'Team', 'position', 'FPTS', 'FPTS_low', 'FPTS_high']]




def fpRankingsKeyGen(keyList):
    '''Generate key for fpRankings using first three letters of first name, 
    full last name, team, and position except defense is defense.'''
    
    if keyList[-1] == 'DEF':
        key = keyList[-2]
    
    else:
        key = re.sub('\W', '',
              ''.join((keyList[0].split(' ')[0][0:3]
                      , keyList[0].split(' ')[1]
                      , keyList[1]
                      , keyList[2])).upper()
              )
              
    return key


def salaryKeyGen(keyList):
    '''Generate key for fpRankings using first three letters of first name, 
    full last name, team, and position except defense is defense.'''
    
    if keyList[-1] == 'DEF':
        key = keyList[-2]
    
    else:
        key = re.sub('\W', '', 
               ''.join((keyList[0][0:3]
                       , keyList[1].split(' ')[0]
                       , keyList[2]
                       , keyList[3])).upper()
               )
               
    return key



def optimizeLineup(dataInput, dataInputDict, budget, target, positionLimit):
    '''Create linear optimiziation problem to optimize target parameter
        given budget and constraints.
        
        Return lineup solution to linear problem
    '''
    
    prob = pulp.LpProblem('The Best Team', pulp.LpMaximize)
    
    # Define player Variables
    playerVars = pulp.LpVariable.dicts('ID', dataInputDict.keys(), cat = 'Binary')
    
    # Add objective of maximizing FPPG
    prob += pulp.lpSum(
            [playerVars[i]*dataInputDict[i][target] 
            for i in dataInput['ID']]
            )
    
    # Salary Cap Constraint
    prob += pulp.lpSum(
            [(playerVars[i] * dataInputDict[i]['Salary']) 
            for i in dataInput['ID']]
            ) <= budget
    
    
    # Position Limits (not Flex)
    for position in ['QB', 'DEF']:
        prob += pulp.lpSum(
                [(playerVars[i] * dataInputDict[i][position]) 
                for i in dataInput['ID']]
                ) == positionLimit[position]
    
    
    for position in ['RB', 'WR', 'TE']:
        prob += pulp.lpSum(
                [(playerVars[i] * dataInputDict[i][position]) 
                for i in dataInput['ID']]
                ) >= positionLimit[position]
    
        prob += pulp.lpSum(
                [(playerVars[i] * dataInputDict[i][position]) 
                for i in dataInput['ID']]
                ) <= (positionLimit[position] + 1)
    
    
    # Team Size Limit
    prob += pulp.lpSum([playerVars[i] for i in list(dataInput['ID'])]) == 9
    

        
    prob.writeLP('teamOptimization.lp')
    prob.solve()
        
    print("Status:", pulp.LpStatus[prob.status])

    return playerVars



#%% LOAD DATA

week = 8

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


# Load salary data
data = pd.read_csv('data\\Yahoo_DF_player_export_w{}.csv'.format(week))


# Generate Key for salary data
data.loc[:, 'key'] = list(map(lambda keyList: 
    salaryKeyGen(keyList)
    , data[['First Name', 'Last Name', 'Team', 'Position']].values.tolist()
    ))


# Identify positions with boolean field
positions = ['QB', 'TE', 'RB', 'DEF', 'WR']

for position in positions:
    data.loc[:,position] = (data['Position'] == position) * 1
    
    
    
    
# Load and concat data for rankings
fpRankings = pd.concat([fantasyProsRankingsDataLoad(position, week) 
                        for position in fantasyProsDict.keys()
                        ], sort = True)

# Rename JAC to JAX
fpRankings.loc[fpRankings['Team'] == 'JAC', 'Team'] = 'JAX'

# Generate key for rankings data
fpRankings.loc[:, 'key'] = list(map(lambda keyList: 
    fpRankingsKeyGen(keyList)
    , fpRankings[['player', 'Team', 'position']].values.tolist()
    ))

    
    


# Load and concat data for projections
fpProjections = pd.concat([fantasyProsProjectionsDataLoad(position, week) 
                        for position in fantasyProsDict.keys()
                        ], sort = True)
 
# Load and concat data for projections
fpAllProjections = pd.concat([fantasyProsAllProjectionsDataLoad(position, week) 
                        for position in fantasyProsDict.keys()
                        ], sort = True)       
    
# Generate key for projections
fpProjections.loc[:, 'key'] = list(map(lambda keyList: 
    fpRankingsKeyGen(keyList)
    , fpProjections[['Player', 'Team', 'position']].values.tolist()
    ))           

    
    
    
    
    
    
    
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


dataInput = dataInput.set_index('key').merge(
        fpRankings.set_index('key')[['Avg', 'Best', 'Worst', 'Rank'
                            ,'player', 'Proj. Pts']]
        , how = 'left'
        , left_index = True
        , right_index = True
        )


dataInput = dataInput.merge(
        pd.DataFrame(fpProjections.set_index('key')['FPTS'])
        , how = 'left'
        , left_index = True
        , right_index = True
        )


# Fill empty projections with 0
dataInput['FPTS'].fillna(0, inplace = True)
dataInput['Proj. Pts'].fillna(0, inplace = True)


#%% INTRODUCING UNCERTAINTY



# Rank projections for each group
dataInput.loc[:, 'FPTS_rank_overall'] = (
        dataInput.groupby('Position')['FPTS']
            .rank(method = 'min'
                  , ascending = False)
            )
            
            
# Rank projections for each group by team
dataInput.loc[:, 'FPTS_rank_team'] = (
        dataInput.groupby(['Position', 'Team'])['FPTS']
            .rank(method = 'min'
                  , ascending = False)
            )

dataInput.groupby('Position').agg(
        {'FPTS_rank_overall':np.max
         , 'FPTS_rank_team':np.max}
        )

# Max # of players to analyze for each position
            # # of players required for each position
positionRankCap = {'QB': {'team':1, 'overall':32}
                   , 'TE':{'team':3, 'overall':64}
                   , 'RB':{'team':3, 'overall':64}
                   , 'DEF':{'team':1, 'overall':32}
                   , 'WR':{'team':4, 'overall':100}
                 }


dataInput.loc[:, 'FPTS_team_rank_filter'] = [
        p[1] <= positionRankCap[p[0]]['team']
        for p in dataInput[['Position','FPTS_rank_team']].values
        ]
        

x = dataInput[dataInput['FPTS_team_rank_filter']].groupby('Position').agg(
        {'FPTS':(np.mean, np.std, len)
         })
    
minAdjustment = 0.9
maxAdjustment = 1.1

dataInput['FPTS_rand'] = (
        dataInput['FPTS'] * (minAdjustment + 
                np.random.rand(dataInput.shape[0])
                *(maxAdjustment - minAdjustment)
                )
        )


dataInput['Proj. Pts_rand'] = (
        dataInput['Proj. Pts'] * (minAdjustment + 
                np.random.rand(dataInput.shape[0])
                *(maxAdjustment - minAdjustment)
                )
        )

# Convert to dictionary for LP
dataInputDict = (
        dataInput.set_index('ID')
        [['Salary', 'FPPG', 'FPTS', 'Proj. Pts', 'FPTS_rand', 'Proj. Pts_rand'] + positions].to_dict('index')
        )


#%% LP SETUP
## ############################################################################

# Setup LP Problem
budget = 200
target = 'FPPG'



budget = 200
target = 'FPPG'







#%% FINAL TEAM
## ############################################################################

finalTeam = {}

for target in  ['FPPG', 'FPTS', 'Proj. Pts', 'FPTS_rand', 'Proj. Pts_rand']:

    playerVars = optimizeLineup(dataInput, dataInputDict, budget, target, positionLimit)

    finalTeam[target] = (
            dataInput.set_index('ID')
            .loc[filter(lambda k: playerVars[k].varValue == 1,
                        playerVars.keys()), :][
            ['First Name', 'Last Name', 'Position', 
             'Team', 'Opponent', 'Salary', 'FPPG', 
             'Rank', 'FPTS', 'FPTS_rand', 'Proj. Pts_rand']]
            )
    
    
    #finalTeam[target][['Salary', 'FPPG', 'FPTS', 'Proj. Pts', 'FPTS_rand']].sum()



#%% DEV
## ############################################################################


x = {position : fantasyProsAllProjectionsDataLoad(position, week)
    for position in fantasyProsDict.keys()
    }

#%% FANTASY PROS DATA
## ############################################################################


# Load and concat data for rankings
fpRankings = pd.concat([fantasyProsRankingsDataLoad(position) 
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




# Load and concat data for projections
fpProjections = pd.concat([fantasyProsProjectionsDataLoad(position) 
                        for position in fantasyProsDict.keys()
                        ], sort = True)
        
    
# Generate key for projections
fpProjections.loc[:, 'key'] = list(map(lambda keyList: 
    fpRankingsKeyGen(keyList)
    , fpProjections[['Player', 'Team', 'position']].values.tolist()
    ))           


dataInput = dataInput.set_index('key').merge(
        fpRankings.set_index('key')[['Avg', 'Best', 'Worst', 'Rank','player']]
        , how = 'left'
        , left_index = True
        , right_index = True
        )


dataInput = dataInput.merge(
        pd.DataFrame(fpProjections.set_index('key')['FPTS'])
        , how = 'left'
        , left_index = True
        , right_index = True
        )



dataInput.to_csv('fpRankings_validation.csv')

#dataInput.loc[:, 'avgRanking'] = fpRankings.set_index('key')['Avg']
