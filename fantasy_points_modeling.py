# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 12:33:37 2019

@author: brett
"""

#%% SETUP ENVIRONMENT
## ############################################################################

import socket
import os

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


# Update code from gitHub

gitPull = input('Update Repo from GitHub [Y/N]? ')

if gitPull.strip().lower()[0] == 'y':
    os.system('git -C {} pull'.format(pc['repo']))

# Set up environment
os.chdir(pc['repo'])


#%% PACKAGES
## ############################################################################

import pandas as pd
import numpy as np
import pulp
import re
import copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from scipy.optimize import minimize
from itertools import combinations, product, chain, repeat, compress
import time

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("poster")

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


def fantasyProsRankingsDataLoad(
        position, week, fantasyProsDict = fantasyProsDict
        , fileName = 'data\\FantasyPros_2019_Week_{}_{}_Rankings.csv'):
    
    '''Read data into a dataframe and append 
    column labeling the player position'''
    
    data = pd.read_csv(fileName.format(week, position))
    
    
    # Filter empy Rows
    data = data[data['Rank'] > 0]

    # Add position label
    data.loc[:, 'position'] = fantasyProsDict[position]['label']
    
    # Rename Column
    data.rename(columns = {fantasyProsDict[position]['column'] : 'player'},
                           inplace = True)

    # Add week #
    data.loc[:, 'week'] = week
    
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

    # Add week #
    data.loc[:, 'week'] = week

    
    # Add Team for DST
    if position == 'DST':
        data['Team'] = [defenseDict.get(player) for player in data['Player']]
    
        data.loc[:, 'Player'] = data['Team']
    
    
    return data[['Player', 'Team', 'position', 'FPTS', 'week']]



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
   
    # Add week #
    data.loc[:, 'week'] = week
    
    
    # Add Team for DST
    if position == 'DST':
        data['Player'] = [defenseDict.get(player) for player in data['Player']]
    
        data.loc[:, 'Team'] = [
                team[0] if team[1] not in ('high', 'low') else team[1]
                for team in data[['Player', 'Team']].values.tolist()]
    
    
    # Add position label
    data.loc[:, 'position'] = fantasyProsDict[position]['label']


    # Add week #
    data.loc[:, 'week'] = week
    
    
    
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

    return data[['Player', 'Team', 'position', 'FPTS', 'FPTS_low', 'FPTS_high', 'week']]



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

#%% 
## ############################################################################
    
positions = ['QB', 'TE', 'RB', 'DEF', 'WR']
weeks = range(4, 13)

#%% LOAD FANTASY PROS RANKINGS DATA
## ############################################################################
    
# Load and concat data for rankings
fpRankings = pd.concat([fantasyProsRankingsDataLoad(position, week) 
                        for position, week in 
                        product(fantasyProsDict.keys(), weeks)
                        ], sort = True)

# Rename JAC to JAX
fpRankings.loc[fpRankings['Team'] == 'JAC', 'Team'] = 'JAX'

# Generate key for rankings data
fpRankings.loc[:, 'key'] = list(map(lambda keyList: 
    fpRankingsKeyGen(keyList)
    , fpRankings[['player', 'Team', 'position']].values.tolist()
    ))

 
    
#%% CREATE WEEKLY MATCHUPS DATASET
    
fpRankings['home'] = [
        (opp.split(' ')[0] == 'vs.') * 1 if type(opp) == str 
          else False
          for opp in fpRankings['Opp'].values.tolist()
        ]
    
# Clean opp column
fpRankings['Opp'] = [
        opp.split(' ')[-1] if type(opp) == str 
          else 'NA' 
          for opp in fpRankings['Opp'].values.tolist()
        ]
  
# Get weekly matchups  
matchups = (fpRankings.groupby(['Team', 'Opp', 'week', 'home'])
                    .agg({'player':len})
                    .reset_index()
                    )
    
#%% LOAD FANTASY PROS PROJECTIONS DATA
## ############################################################################



# Load and concat data for projections
fpProjections = pd.concat([fantasyProsProjectionsDataLoad(position, week) 
                        for position, week in 
                        product(fantasyProsDict.keys(), range(4,8))
                        ], sort = True)
 
# Load and concat data for projections
fpAllProjections = pd.concat([fantasyProsAllProjectionsDataLoad(position, week) 
                        for position, week in 
                        product(fantasyProsDict.keys(), range(8,13))
                        ], sort = True)       
    
        
 
     
# Calculate stats
fpAllProjections = calculateProjectionStats(fpAllProjections)



# Generate key for merging  
fpProjections.loc[:, 'key'] = list(map(lambda keyList: 
    fpRankingsKeyGen(keyList)
    , fpProjections[['Player', 'Team', 'position']].values.tolist()
    ))      


fpAllProjections.loc[:, 'key'] = list(map(lambda keyList: 
    fpRankingsKeyGen(keyList)
    , fpAllProjections[['Player', 'Team', 'position']].values.tolist()
    ))      
    

# Combine data
fpProjections = pd.concat(
        (fpProjections, 
         fpAllProjections[['FPTS', 'Player', 'Team', 'position', 'key', 'week']])
        , axis = 0)
  

# Rename JAC to JAX
fpProjections.loc[fpProjections['Team'] == 'JAC', 'Team'] = 'JAX'  
      
#%% LOAD PFF DATA
## ############################################################################

# Load offense data
pffData = {f : pd.read_csv(
        '{}\\pff_data\\{}_summary_agg.csv'.format(pc['wd'], f))
        for f in ('passing', 'receiving', 'rushing')
            }
        

# Load defense data
pffDataDef = pd.read_csv(
        '{}\\pff_data\\'
        '{}_summary_agg.csv'.format(pc['wd'], 'defense_special_team')
        )

# Add columns to defense data for merging
pffDataDef['player'] = pffDataDef['team_name']
pffDataDef['position'] = 'DEF'
pffDataDef['player_id'] = pffDataDef['team_name']


# Calculate team total offensive points
offensePoints = pd.concat([
        (data.groupby(['team_name', 'season', 'week'])
            .agg({'fantasy_points':np.sum})
            .rename(columns = {'fantasy_points':'fantasy_points_{}'.format(f)})
        ) for f, data in pffData.items()
        ]
        , axis = 1
        , sort = True
        ).reset_index()


offensePoints['fantasy_points_offense'] = (
        offensePoints[['fantasy_points_{}'.format(f) for f in pffData.keys()]
        ].sum(axis = 1)
        )


# Combine rushing, receiving and passing files
pffDataCols = ['player', 'player_id', 'team_name'
               , 'position', 'season', 'week', 'fantasy_points']


pffData = pd.concat([data[pffDataCols] for data in pffData.values()])


# Combine player stats
pffData = pffData.groupby(pffDataCols[:-1]).sum().reset_index()


# Rename running back position
pffData.loc[[p in ('HB', 'FB') for p in pffData['position']], 'position'] = 'RB'


# Create key for mapping
pffData.loc[:, 'key'] = list(map(lambda keyList: 
    fpRankingsKeyGen(keyList)
    , pffData[['player', 'team_name', 'position']].values.tolist()
    ))   

pffDataDef.loc[:, 'key'] = list(map(lambda keyList: 
    fpRankingsKeyGen(keyList)
    , pffDataDef[['player', 'team_name', 'position']].values.tolist()
    ))   

    
    
# Calculate EWMA by player_id and season
pffDataEWMA = (pffData.set_index([
        'player_id', 'season', 'player', 'team_name', 'key', 'position', 'week'])
                    .sort_values(['player_id', 'season', 'week'])
                    .groupby(level = ['player_id', 'season'])
                    .apply(lambda points: 
                        points.shift(1)
                            .ewm(alpha = 0.8, ignore_na = True)
                            .mean()
                    )
                .reset_index()
                .rename(columns = {'fantasy_points':'fantasy_points_ewma'})
                )
 
                        
pffDataDefEWMA = (pffDataDef.set_index([
        'player_id', 'season', 'player', 'team_name', 'key', 'position', 'week'])
                    .sort_values(['player_id', 'season', 'week'])
                    .groupby(level = ['player_id', 'season'])
                    .apply(lambda points: 
                        points.shift(1)
                            .ewm(alpha = 0.8, ignore_na = True)
                            .mean()
                    )
                .reset_index()
                .rename(columns = {
                        c : '{}_ewma'.format(c)
                        for c in ['fantasy_points',
                                 'points_allowed_passing',
                                 'points_allowed_receiving',
                                 'points_allowed_return_team',
                                 'points_allowed_rushing',
                                 'points_allowed_total',
                                 'points_scored']})
                )
                        

    
#%% MERGE DATA SETS
## ############################################################################
    
    
# Outer join FPTS and Proj. Pts and then inner join actuals
mergedPoints = (fpRankings
     .set_index(['key', 'week', 'player', 'Team', 'position'])[['Avg', 'Proj. Pts']]
     .merge(pd.DataFrame(
             fpProjections
                 .rename(columns = {'Player':'player'})
                 .set_index(['key', 'week', 'player', 'Team', 'position'])['FPTS']
             )
            , left_index = True
            , right_index = True
            , how = 'outer'
                )
    .merge(pd.DataFrame(
        pffData[pffData['season']==2019]
            .rename(columns = {'team_name_games':'Team'})
        .set_index(['key', 'week', 'player', 'Team', 'position'])['fantasy_points']
        )
        , left_index = True
        , right_index = True
        , how = 'left'
        )
    .reset_index(['player', 'position',])
    .merge(matchups.set_index(['week', 'Team']).drop('player', axis = 1)
            , left_index = True
            , right_index = True
            , how = 'inner')
    .reset_index()
    )
   
# Clean Team Name
mergedPoints['Team'] = mergedPoints['Team'].map(lambda team: team.strip())
    




#%% MODELING DATASET
## ############################################################################

# Filling missing projections
projectionMeans = (
        mergedPoints.groupby(['position', 'week'])
            .agg({'Proj. Pts' : np.nanmean
                  , 'FPTS' : np.nanmean})
            .to_dict('dict')
        )

    
projectionMeans['FPTS'][tuple(mergedPoints.iloc[0][['position', 'week']])]
    
def fillMissingDatawMeans(data
                          , levelOfDetail = ['position', 'week']
                          , columns = ['Proj. Pts', 'FPTS']):
     
    
    meanDict = (
            data.groupby(levelOfDetail)
                .agg({column:np.nanmean for column in columns})
                .to_dict('dict')
                )
    
    # Group at coarsest level of detail for error handling
    meanDict2 = (
            data.groupby(levelOfDetail[0])
                .agg({column:np.nanmean for column in columns})
                .to_dict('dict')
                )
    
    # Fill missing data with means
    for column in columns:
        data['{}_filled'.format(column)] = [
                meanDict.get(column, 0).get(tuple(r[1][levelOfDetail]), 0)
                if np.isnan(r[1][column]) else r[1][column]
                for r in data.iterrows()
                ]

        # Fill any remaining NAs
        data['{}_filled'.format(column)] = [
                (meanDict2.get(column, 
                               data[column].mean())
                        .get(r[1][levelOfDetail[0]], 
                             data[column].mean())
                        )
                if np.isnan(r[1]['{}_filled'.format(column)]) 
                else r[1]['{}_filled'.format(column)]
                for r in data.iterrows()
                ]

    return data


modelData = fillMissingDatawMeans(mergedPoints)



# OHE position, team, and opponent
modelData = pd.get_dummies(
        modelData.set_index(['week', 'key', 'player', 'position', 'FPTS', 'Proj. Pts'])
        , columns = ['Team', 'Opp']
        )


# Filter na's
modelData = modelData[modelData.apply(
        lambda r: np.any(np.isnan(r)) == False, axis = 1)]


    
    
    

for position in positions:


    train, test = train_test_split(
            modelData[modelData.index.get_level_values('position') == position]
            , test_size = 0.2)
    
    rf = RandomForestRegressor(random_state=1127)
    

    rf.fit(train.drop('fantasy_points', axis = 1).values
           , train['fantasy_points'].values)
          
    
    # Train Results
    print(position
          , 'train model'
          , r2_score(train['fantasy_points'].values,
                     rf.predict(train.drop('fantasy_points', axis = 1).values)
                     )
         )
          

    print(position
          , 'train Proj. Pts'
          , r2_score(train['fantasy_points'].values,
                     train['Proj. Pts_filled'].values)
                     
         )
  
    print(position
          , 'train FPTS'
          , r2_score(train['fantasy_points'].values,
                     train['FPTS_filled'].values)
         )
        
        
    # Test Results
    print(position
          , 'test model'
          , rf.score(test.drop('fantasy_points', axis = 1).values
           , test['fantasy_points'].values)
          )


    print(position
          , 'test Proj. Pts'
          , r2_score(test['fantasy_points'].values,
                     test['Proj. Pts_filled'].values)
                     
         )
  
    print(position
          , 'test FPTS'
          , r2_score(test['fantasy_points'].values,
                     test['FPTS_filled'].values)
         )


#%% DEV
## ############################################################################
          
x = pd.DataFrame((list(repeat('a', 5)) + list(repeat('b',5))
                , list(range(1,10))), columns = ['x', 'y'])   

list(repeat(('a', 'b'), 4))

x = pd.DataFrame(list(zip(range(10)
        , list(repeat('a', 5)) + list(repeat('b',5))))
, columns = ['x', 'y'])

x.cumsum()       

x.groupby('y').apply(lambda x: x.shift(1).ewm(alpha = .8, ignore_na = True).mean())
