# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:00:42 2019

@author: u00bec7
"""

## PACKAGES


import pandas as pd
import os
import numpy as np
import pulp
import re
import socket
import copy
from scipy.stats import norm

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


def calculateProjectionStats(projections, ci = 0.95):
    '''Calculate range and estimate std. dev. of projections.
    
    Return same data frame with new stat columns.'''
    

    
    projections.loc[:, 'FPTS_range'] = (
            projections['FPTS_high'] - projections['FPTS_low']
            )
    
    projections.loc[:, 'FPTS_lever'] = (
            (projections['FPTS'] - projections['FPTS_low']) / 
            (projections['FPTS_range'])
            )
        
    projections.loc[:, 'FPTS_std_dev_est'] = (
            projections['FPTS_range'] / norm.ppf(ci))
    
    return projections


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



#%% SETUP ENVIRONMENT
## ############################################################################

week = 10

# Working Directory Dictionary
pcs = {
    'WaterBug' : {'wd':'C:\\Users\\brett\\Documents\\ff',
                  'repo':'C:\\Users\\brett\\Documents\\personal\\ff'},

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
    
    
#%% LOAD FANTASY PROS RANKINGS DATA
## ############################################################################
    
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

    

#%% RANKINGS FANTASY POINT REGESSION MODELING
## ############################################################################

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

rfRegDict = {}

for position in positions:

    # Random Forest Regressor
    rfRegDict[position] = RandomForestRegressor(
#            max_depth=2
                                  random_state=1127
                                  , n_estimators=100
                                  , oob_score = True)
    
    positionFilter = (fpRankings['position'] == position).values.tolist()
    
    # Fit model
    rfRegDict[position].fit(
            fpRankings.loc[positionFilter, 'Avg'].values.reshape(-1,1)
            , fpRankings.loc[positionFilter, 'Proj. Pts']
            )

    # OOB & r^2 Results
    print(position
          , round(r2_score(
                  fpRankings.loc[positionFilter, 'Proj. Pts']
                  , rfRegDict[position].predict(
                          (fpRankings.loc[positionFilter, 'Avg']
                                     .values
                                     .reshape(-1,1)))
                  )
            , 3)
          , round(rfRegDict[position].oob_score_, 3))
    
    
#%% LOAD FANTASY PROS RANKING DATA BY POSITION EXPERTS
## ############################################################################

#fpRankingsPosition = pd.concat([
#        fantasyProsRankingsDataLoad(position, week,
#        fileName='data\\FantasyPros_2019_Week_{}_{}_Rankings_position.csv') 
#        for position in fantasyProsDict.keys()
#        ], sort = True)
#
## Rename JAC to JAX
#fpRankingsPosition.loc[fpRankingsPosition['Team'] == 'JAC', 'Team'] = 'JAX'
#
## Generate key for rankings data
#fpRankingsPosition.loc[:, 'key'] = list(map(lambda keyList: 
#    fpRankingsKeyGen(keyList)
#    , fpRankingsPosition[['player', 'Team', 'position']].values.tolist()
#    ))
#
#    
## Rename Proj. Pts Column
#fpRankingsPosition.rename(
#        columns = {k:'{} Position'.format(k) for k in 
#                   ['Avg', 'Best', 'Worst', 'Rank','player', 'Proj. Pts']}
#        , inplace = True
#        )
    
#%% LOAD FANTASY PROS PROJECTIONS DATA
## ############################################################################


# Load and concat data for projections
#fpProjections = pd.concat([fantasyProsProjectionsDataLoad(position, week) 
#                        for position in fantasyProsDict.keys()
#                        ], sort = True)
 
# Load and concat data for projections
fpAllProjections = pd.concat([fantasyProsAllProjectionsDataLoad(position, week) 
                        for position in fantasyProsDict.keys()
                        ], sort = True)       
    
        
        
# Calculate stats
fpAllProjections = calculateProjectionStats(fpAllProjections)



        
# Generate key for projections
#fpProjections.loc[:, 'key'] = list(map(lambda keyList: 
#    fpRankingsKeyGen(keyList)
#    , fpProjections[['Player', 'Team', 'position']].values.tolist()
#    ))           
#
#    
    
fpAllProjections.loc[:, 'key'] = list(map(lambda keyList: 
    fpRankingsKeyGen(keyList)
    , fpAllProjections[['Player', 'Team', 'position']].values.tolist()
    ))      
    
    
#%% PROJECTIONS UNCERTAINTY
## ############################################################################
    

np.random.seed(1127)

for i in range(7):
    fpAllProjections.loc[:, 'FPTS_rand_{}'.format(i)] = (
            fpAllProjections['FPTS'] + (
                    fpAllProjections['FPTS_std_dev_est'] 
                    * np.random.randn(fpAllProjections.shape[0])
                    )
            )

  
# Columns for merging
fpAllProjectionsCols = (
        ['FPTS']
        + ['FPTS_rand_{}'.format(i) for i in range(7)] 
        )
    

#%% RANKINGS UNCERTAINTY
## ############################################################################

zScore = norm.ppf(0.95)

np.random.seed(1213)

for i in range(4):
    
    # Create uncertainty around rank
    fpRankings.loc[:, 'Proj. Pts_rand_{}'.format(i)] = (
            fpRankings['Avg'] + (
                    fpRankings['Std Dev'] 
                    * zScore
                    * np.random.randn(fpRankings.shape[0])
                    )
            )

    # Estimate projected points based on new ranking
    #   Call RF model by position
    fpRankings.loc[:, 'Proj. Pts_rand_{}'.format(i)] = (
        
        list(map(lambda p: 
            rfRegDict.get(p[0]).predict(np.array(p[1]).reshape(1,-1))
            , fpRankings[['position', 'Proj. Pts_rand_{}'.format(i)]].values.tolist()
            ))
        )


# Columns for merging
fpRankingsCols = (
        ['Proj. Pts']
        + ['Proj. Pts_rand_{}'.format(i) for i in range(4)] 
        )


#%% COMBINE DATASETS
    
# # of players required for each position
positionLimit = {'QB':1
                 , 'TE':2
                 , 'RB':2
                 , 'DEF':1
                 , 'WR':3
                 }

# Player status' to exclude
statusExclude = ['IR'
                 , 'SUSP'
                 , 'O'
#                 , 'Q'
                 ]


# Filter only eligble players
dataInput = copy.deepcopy(
        data[data['Injury Status'].map(lambda s: s not in statusExclude)]
        )


dataInput = dataInput.set_index('key').merge(
        fpRankings.set_index('key')[['Avg', 'Best', 'Worst', 'Rank'
                            ,'player'] + fpRankingsCols]
        , how = 'left'
        , left_index = True
        , right_index = True
        )


#dataInput = dataInput.merge(
#        fpRankingsPosition.set_index('key')[['{} Position'.format(k) for k in 
#                   ['Avg', 'Best', 'Worst', 'Rank','player', 'Proj. Pts']]]
#        , how = 'left'
#        , left_index = True
#        , right_index = True
#        )


dataInput = dataInput.merge(
        fpAllProjections.set_index('key')[fpAllProjectionsCols]
        , how = 'left'
        , left_index = True
        , right_index = True
        )


# Fill empty projections with 0
dataInput.fillna(0, inplace = True)





#%% LP SETUP
## ############################################################################


# Convert to dictionary for LP
lpTargets = fpAllProjectionsCols + fpRankingsCols + ['FPPG']

dataInputDict = (
        dataInput.set_index('ID')
        [['Salary'] + lpTargets + positions].to_dict('index')
        )

# Setup LP Problem
budget = 200


#%% TEAM OPTIMIZATION
## ############################################################################

finalTeam = {}

# Create optimized team to each points projection
for target in  lpTargets:

    playerVars = optimizeLineup(dataInput
                                , dataInputDict
                                , budget
                                , target
                                , positionLimit)

    finalTeam[target] = (
            dataInput.set_index('ID')
            .loc[filter(lambda k: playerVars[k].varValue == 1,
                        playerVars.keys()), :][
            ['First Name', 'Last Name', 'Position', 
             'Team', 'Opponent', 'Salary', 'Rank'] + lpTargets]
            )
    
    


x = pd.concat(finalTeam.values())


x.groupby(['Position', 'Last Name', 'First Name'])['Team'].count().groupby(level=0).nlargest(20)






#%% DEV
## ############################################################################


from matplotlib import pyplot as plt
import seaborn as sns

fpAllProjections.loc[:, 'position_rank'] = (
        fpAllProjections.groupby('position')['FPTS'].rank(ascending = False)
        )



fig, ax = plt.subplots(1, 2, sharey= True, figsize = (10, 6))

sns.scatterplot(x = 'Rank', y = 'Proj. Pts', hue = 'position'
                , data = fpRankings, ax = ax[0]
                )
sns.scatterplot(x = 'position_rank', y = 'FPTS', hue = 'position'
                , data = fpAllProjections, ax=ax[1]
                )

np.random.randn(10)

ax[0].grid()
ax[1].grid()

x = {position : fantasyProsAllProjectionsDataLoad(position, week)
    for position in fantasyProsDict.keys()
    }



dataInput.groupby('Position')['Team'].count()
dataInput['FPTS_rnd'] = dataInput['FPTS'].round(0)
dataInput['Proj. Pts_rnd'] = dataInput['Proj. Pts'].round(0)


dataInput.groupby(['Position', 'FPTS_rnd'])['Team'].count()


dataInput.groupby(['Position', 'Proj. Pts_rnd'])['Team'].count()

# Take the max of the median and mean
x = dataInput.groupby(['Position']).agg({
        'Proj. Pts': lambda x: max(np.mean(x), np.percentile(x, 80))
        , 'FPTS': lambda x: max(np.mean(x), np.percentile(x, 80))}).to_dict('index')
    
dataInputFPTS = dataInput.loc[
        map(lambda p: x[p[0]]['FPTS'] < p[1]
            , dataInput[['Position', 'FPTS']].values.tolist()
            )
        , :]



dataInputFPTS.groupby(['Position'])['Team'].count()

from itertools import combinations, product

(len(list(combinations(dataInputFPTS.loc[dataInputFPTS['Position']=='WR', 'ID'].values.tolist(), 3)))
* len(list(combinations(dataInputFPTS.loc[dataInputFPTS['Position']=='RB', 'ID'].values.tolist(), 2)))
*3*9*15)


y = product(
        combinations(dataInputFPTS.loc[dataInputFPTS['Position']=='WR', 'ID'].values.tolist(), 3)
        , combinations(dataInputFPTS.loc[dataInputFPTS['Position']=='RB', 'ID'].values.tolist(), 2)
        , dataInputFPTS.loc[dataInputFPTS['Position']=='QB', 'ID'].values.tolist()
        , dataInputFPTS.loc[dataInputFPTS['Position']=='TE', 'ID'].values.tolist()
        , dataInputFPTS.loc[dataInputFPTS['Position']=='DEF', 'ID'].values.tolist()
        , dataInputFPTS.loc[[p in ('WR', 'TE', 'RB') for p in dataInputFPTS['Position']], 'ID']
        )

yy = list(filter(lambda team: len(set(team)) == 9, y))

list(combinations(['a', 'b', 'c'], 2))

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
