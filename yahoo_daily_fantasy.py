# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:00:42 2019

@author: u00bec7
"""





#%% SETUP ENVIRONMENT
## ############################################################################

import os
import socket

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
#os.system('git -C {} pull'.format(pc['repo']))

# Set up environment
os.chdir(pc['repo'])


### WEEK RUN
yr = 2020
week = 3
calculateUncertainty = False


# Which projections to use
dataDict = {
    'fpRankings': True,
    'fpProjections': True,
    'pffProjections': False,
    }


salaryFile = os.path.join(
    'data',
    'Yahoo_DF_contest_lineups_insert_template_'
    '{}_w{}.csv'.format(yr, week)
    )



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
from scipy.stats import norm
from scipy.optimize import minimize
from itertools import combinations, product, chain, repeat, compress
import time

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("poster")

#%% FUNCTIONS
## ############################################################################

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


# Dictionary for team abbreviations
teamAbrvsDict = {
        'ARZ' : 'ARI'
        , 'BLT' : 'BAL'
        , 'CLV' : 'CLE'
        , 'HST' : 'HOU'
        , 'LA' : 'LAR'
        , 'SD' : 'LAC'
        , 'SL' : 'LAR'
        }


def fantasyProsRankingsDataLoad(
        position, yr, week, fantasyProsDict = fantasyProsDict
        , fileName = 'data\\FantasyPros_{}_Week_{}_{}_Rankings.csv'):
    
    '''Read data into a dataframe and append 
    column labeling the player position'''
    
    data = pd.read_csv(fileName.format(yr, week, position))
    
    
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



def fantasyProsProjectionsDataLoad(
        position
        , yr
        , week
        , fantasyProsDict = fantasyProsDict
        , defenseDict = defenseDict
        , fileName = 'data\\FantasyPros_Fantasy_Football_Projections_{}_{}_w{}.csv'
        ):
    
    '''Read data into a dataframe and append column labeling the player position'''
    
    data = pd.read_csv(fileName.format(position, yr, week))
    

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



def fantasyProsAllProjectionsDataLoad(
        position
        , yr
        , week
        , fantasyProsDict = fantasyProsDict
        , defenseDict = defenseDict
        , fileName = ('data\\FantasyPros_Fantasy_'
                      'Football_Projections_{}_'
                      'high_low_{}_w{}.csv'
                      )
        ):
    
    '''Load fantansy pros projections with high, low, and averages.
        Pivot high, low, and avg into single record for each player
        and estimate std. dev.
        
        Return subset dataframe of just FPTS.
    '''
    
    # Load data
    data = pd.read_csv(fileName.format(position, yr, week))
    
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

    data.index.name = 'index'

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


#def optimizeTeamSelections(optimumTeamDerivatives, numTeams = 5):
#    
#    prob = pulp.LpProblem('Most Different Teams', pulp.LpMaximize)
#
#    teamVars = pulp.LpVariable.dicts('finalTeamID'
#            , optimumTeamDerivatives.index.get_level_values(0)
#            , cat = 'Binary')
#    
#    
#     # Add objective of maximizing target
#    prob += pulp.lpSum(
#            [teamVars[i]*optimum 
#            for i in dataInput['ID']]
#            )           


def rankingsStdDevPointEstimate(model, rankingCol, stdDevCol):
    '''Estimate point projections std dev using ranking model and ranking stddev
    '''
    
    # Perform predictions 1 std above and below ranking
    lower = model.predict((rankingCol + stdDevCol).values.reshape(-1,1))
    upper = model.predict((rankingCol - stdDevCol).values.reshape(-1,1))
    mean = model.predict(rankingCol.values.reshape(-1,1))
    
    # Find max variance
    std = np.vstack(((upper - mean), (mean - lower))).max(axis = 0)
    
    return std



def optimizeLineup(dataInput, dataInputDict
                   , budget, target, positionLimit
                   , writeLPProblem = False
                   , printOutput = False):
    '''Create linear optimiziation problem to optimize target parameter
        given budget and constraints.
        
        Return lineup solution to linear problem
    '''
    
    prob = pulp.LpProblem('The Best Team', pulp.LpMaximize)
    
    # Define player Variables
    playerVars = pulp.LpVariable.dicts('ID', dataInputDict.keys()
                                        , cat = 'Binary')
    
    # Add objective of maximizing target
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
    

    if writeLPProblem == True:
        prob.writeLP('teamOptimization.lp')
    
    prob.solve()
       
    if printOutput == True:
        print("Status:", pulp.LpStatus[prob.status])

    return playerVars



def optimumTeamCombinations(
        teamInteractionsDict, 
        numTeams, 
        writeLPProblem = False
        ):
    
    
    prob = pulp.LpProblem('The Best Team Combinations', pulp.LpMaximize)
    
    # Define team variables
    teamVars = pulp.LpVariable.dicts('ID', teamInteractionsDict.keys()
                                        , cat = 'Binary')


    # Add objective of maximizing team differences
    prob += pulp.lpSum(
        [teamVars[i]*teamVars[j]*teamInteractionsDict[i][j] 
        for i, j in combinations(teamInteractionsDict.keys(), 2)]
        )
    
    # Number of teams limit
    prob += pulp.lpSum([teamVars[i] 
        for i in teamInteractionsDict.keys()]) == numTeams
        
    if writeLPProblem == True:
        prob.writeLP('teamOptimization.lp')
    
    prob.solve(GLPK(msg = 0))
        
    print("Status:", pulp.LpStatus[prob.status])
    
    
    return teamVars
    


def optimizedTeamDerivaties(optimumTeam, dataInput
                            , dataInputDict, budget
                            , target, positionLimit
                            ):

    '''Find all optimized derivatives of the optimum team by excluding all
        combinations of players in the optimum team (511 combinations)
        
        Returns dictionary of optimum teams'''
    
    optimumTeamDict = {}

    # Generate all combinations of players for excluding to create alternative
    # teams
    combinationsList = [
            combinations(optimumTeam.index.get_level_values(0), i+1)
            for i in range(9)
            ]

    # Combine into single list
    combinationsList = list(map(list, chain(*combinationsList)))
    
    
    # Create new optimium teams by excluding each player combination from
    # dataset
    st = time.time()
    
    for i, players in enumerate(combinationsList):
        
        # Create optimum team
        playerVars = optimizeLineup(
                dataInput.set_index('ID').drop(players).reset_index()
                , dataInputDict
                , budget
                , target
                , positionLimit)
    
        
        optimumTeamDict[i] = (
                dataInput.set_index('ID')
                .loc[filter(lambda k: playerVars[k].varValue == 1,
                            playerVars.keys())
                    , :][['First Name', 'Last Name', 'Position' 
                       , 'Team', 'Opponent', 'Salary', 'Rank'
                       , target
                       , '{}_std_dev_est'.format(target)
                       ]
                    ]
                )    
    
        optimumTeamDict[i]['finalTeamID'] = i
        optimumTeamDict[i]['teamPoints'] = optimumTeamDict[i][target].sum()
        optimumTeamDict[i]['teamPointsStdDev'] = (
                np.sqrt(np.square(optimumTeamDict[i][
                        '{}_std_dev_est'.format(target)
                        ].values).sum())
                )
    
    print(round(time.time() -st, 2))
    
    # Add optimum team to dictionary
    optimumTeamDict[optimumTeam['finalTeamID'].iloc[0]] = optimumTeam
    
    return optimumTeamDict



def calcTeamPlayerStats(playerList, target
                    , groupbyList = ['Position', 'Last Name', 'First Name']):
    
    '''Count player appearances and stats in multiple team sections
    
    Return grouped dataframe by groupbyList'''
    
    
    teamPlayerStats = (playerList.groupby(groupbyList)
        .agg({'Team':len
              , 'teamPoints':np.mean
              , target:np.mean})
        .rename(columns = {'Team':'teamCount'
                           , 'teamPoints':'teamAvgPoints'})
    .groupby(level = 0)
    .apply(lambda position: position.sort_values(['teamCount', 'teamAvgPoints']
                                                   , ascending = False)
        ))
        
    return teamPlayerStats
    


def pivotTeamsAndDropDuplicates(optimumTeamDict, optimumTeam, target):
    '''Find all unique derivatives of optimum team and return
        one hot encoded dataframe of each team's players'''

    # Combine results
#    playerList = pd.concat((pd.concat(optimumTeamDict.values()), optimumTeam))
    playerList = pd.concat(optimumTeamDict.values(), sort = True)
    
    
    # View player counts
#    print(calcTeamPlayerStats(playerList, target))
    
    
    # Pivot out players and one hot encode each team
    playerListDummies = pd.get_dummies(
            playerList.index.get_level_values(0)
            , dtype = np.int16)
    playerListDummies['finalTeamID'] = playerList['finalTeamID'].values
    playerListDummies = playerListDummies.groupby('finalTeamID').sum()
    
    
    # Create one hot encoding for each team with player points values
    teamList = pd.concat((
            optimumTeam[target], 
            pd.concat(
                    map(lambda t: t[target], 
                        optimumTeamDict.values())
                , axis = 1
                , sort = True
                ))
            , axis = 1
            , sort = True
            )
      
    
    # Rename Columns
    teamList.columns = (
            list(set(optimumTeam['finalTeamID'])) 
            + list(optimumTeamDict.keys())
            )
    
    # Fill empty cells with 0 to complete OHE
    teamList.fillna(0, inplace = True)
    
    # Transpose DF so rows are team observations
    teamList = teamList.transpose()
    
    


    
    # Append team point projections
    teamList[target] = (
            [optimumTeam[target].sum()] +
            [i[target].sum() for i in optimumTeamDict.values()]
            )
    
    
    # Apppend team point standard deviations
    teamList['teamPointsStdDev'] = (
            [optimumTeam['teamPointsStdDev'].mean()] +
            [i['teamPointsStdDev'].mean() 
                for i in optimumTeamDict.values()]
            )
    
    
    # Remove duplicate teams
    teamList.drop_duplicates(inplace = True)
    
    # Append team rank based on points
    teamList['teamRank'] = (
            teamList[target].rank(method = 'min', ascending = False)
            )
    
    
    # Dilute player points across all teams
    teamList['dilutedPoints'] = (
        (teamList 
         / teamList.sum(axis = 0)
         ).drop(['teamRank', target], axis = 1)
        .sum(axis = 1)
        )

    
    # Sort teams by projected points
    teamList.sort_values(target, ascending = False, inplace = True)
    
    return teamList




def calculateTeamDifference(team, otherTeam, method = 'intersection'):
    '''Caluculate similarity of two teams either by the percent intersection
    If DF is OHE of player then it'll be a comparison of players overlapped
    between the two teams. If the DF is OHE with projected points then
    it'll be a comparison of points overlapped between the two teams.
        
    If method is 'intersection' then it's the % different, otherwise
    the RMSE is calculated between the two teams
        
    return scalar of comparison
    '''

    if method == 'intersection':
        teamComparison = (
                np.abs(team.values - otherTeam.values).sum() 
                / (team.values + otherTeam.values).sum()
                )

    else:
        teamComparison = np.linalg.norm(team.values-otherTeam.values) 
        
    return teamComparison



def mapCalculateTeamDifference(teams, method):
    '''Apply calculateTeamDifferences function by comparing every team
    to every other team in the supplied dataset
    
    return n x n DataFrame where n is the number of teams in the original DF
    '''


    teamComparisons = pd.DataFrame(
            [[calculateTeamDifference(team, otherTeam, method) 
                for io, otherTeam in teams.iterrows()] 
            for i, team in teams.iterrows()]
            , columns = teams.index.get_level_values(0)
            , index = teams.index.get_level_values(0)
            )
    
    
    return teamComparisons



def clusterTeamsAndPlot(teamRelationships, teams, target, k = 10):
    '''Cluster teams using KMeans based on team relationship to otherteams
    
    Return dataframe with cluster labels appended and cluster model'''
    
    
    # Calculate clusters    
    km = KMeans(n_clusters = k, random_state=1127)
    km.fit(teamRelationships)
    
    # Assign labels
    teams['cluster'] = km.labels_
    

    
    # Swarm plot of clusters and target
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,6))
    sns.swarmplot(x = 'cluster'
                  , y = target
                  , data = teams
                  , ax = ax[0])
    
    
    # Plot distributed points and total points and color by cluster #
    sns.scatterplot(x = 'dilutedPoints'
                    , y = target
                    , hue = 'cluster'
                    , data = teams
                    , ax = ax[1])
    
    # Turn on Grid
    ax[0].grid()
    ax[1].grid()
    
    ax[0].set_title('{} Swarm Plots by Cluster'.format(target), fontsize = 20)
    ax[1].set_title('Team Diluted vs Total {}'.format(target), fontsize = 20)
    
    fig.show()
    
    return teams, km



def selectClusterBestTeams(teams, teamsDict, target, numTeams = 10):

    # Select top team in each cluster and best n teams overall
    topTeams = (teams.groupby('cluster')[target]
                    .nlargest(1)
                    .nlargest(numTeams)
                    .reset_index()
                    .drop('cluster', axis = 1)
                    .rename(columns = {'level_1':'finalTeamID'})
                    )
    
    # Get team selections from dataframe with details 
    topTeams = teams.loc[topTeams['finalTeamID'],:]


    # Identify player columns 
    playerCols = list(filter(lambda col: col.startswith('nfl')
                            , topTeams.columns))

    
    # Create dictionary for each selected team
    topTeamsDict = {team : teamsDict[team] 
        for team in topTeams.index.get_level_values(0)
        }
    
#    topTeamsDict = {
#            team : players.set_index('ID').loc[
#                    list(compress(
#                        playerCols, [p > 0  for p in topTeams.loc[team,playerCols]]
#                        ))
#                    , : # data.drop('ID', axis = 1).columns
#                    ].reset_index()
#            for team in topTeams.index.get_level_values(0)
#            }

    
    return topTeams, topTeamsDict



def assignPositionLabels(df):
    '''Position labels for importing into Yahoo'''
    
    labelDict = {
            'WR': ['WR1', 'WR2', 'WR3', 'FLEX']
            , 'RB':['RB1', 'RB2', 'FLEX']
            , 'TE':['TE', 'FLEX']
            }
    
    labels = [labelDict.get(p, [p]).pop(0) for p in df['Position']]
    
    return labels



def writeTeamSubmissions(topTeamsDict, target, week):
    '''Convert team selections into correct format for yahoo csv submission.'''
    
    # Correct column order
    sortedColumns = ['QB'
                     , 'RB1', 'RB2'
                     , 'WR1', 'WR2', 'WR3'
                     , 'TE'
                     , 'FLEX'
                     , 'DEF'
                     ]

    for key in topTeamsDict:
            topTeamsDict[key]['labels'] = (
                    assignPositionLabels(topTeamsDict[key])
                    )
            
    topTeamsSubmit = pd.concat(
        [pd.DataFrame(df.reset_index().set_index('labels')['ID']) 
        for df in topTeamsDict.values()]
        , axis = 1, sort = True).transpose()
    
    
    topTeamsSubmit[sortedColumns].to_csv(
            'data\\team_submission_{}_w{}.csv'.format(target, week)
            , index = False)
    
    print('file written: data\\'
          'team_submission_{}_w{}.csv'.format(target, week))
    
    return

#%% LOAD YAHOO SALARY DATA & LINEUP TEMPLATE
## ############################################################################

# Load salary data
# data = pd.read_csv('data\\Yahoo_DF_player_export_{}_w{}.csv'.format(yr, week))

# Load csv file for loading team selections
data = pd.read_csv(
    salaryFile  
    , usecols = ['ID',
             'First Name',
             'Last Name',
             'ID + Name',
             'Position',
             'Team',
             'Opponent',
             'Game',
             'Time',
             'Salary',
             'FPPG',
             'Injury Status',
             'Starting']
    , skiprows = range(1,6))


# Create key for merging with other data file

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
if dataDict.get('fpRankings', False) == True:
    
    # Load and concat data for rankings
    fpRankings = pd.concat([fantasyProsRankingsDataLoad(position, yr, week) 
                            for position in fantasyProsDict.keys()
                            ], sort = True)
    
    # Rename JAC to JAX
    fpRankings.loc[fpRankings['Team'] == 'JAC', 'Team'] = 'JAX'
    
    # Generate key for rankings data
    fpRankings.loc[:, 'key'] = list(map(lambda keyList: 
        fpRankingsKeyGen(keyList)
        , fpRankings[['player', 'Team', 'position']].values.tolist()
        ))
    
    # Column name alignment
    fpRankings.rename(columns = {
        'Proj. Pts': 'pts', 
        'Proj. Pts_std_dev_est': 'pts_stddev'
        },
        inplace = True
        )
        
    # fpRankingsCols = ['Proj. Pts', 'Proj. Pts_std_dev_est']



#%% RANKINGS FANTASY POINT REGESSION MODELING
## ############################################################################

if dataDict.get('fpRankings', False) == True:
    
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
                , fpRankings.loc[positionFilter, 'pts']
                )
    
        # Add model predictions to dataframe
        fpRankings.loc[positionFilter, 'pts_model_est'] = (
                rfRegDict[position].predict(
                              (fpRankings.loc[positionFilter, 'Avg']
                                         .values
                                         .reshape(-1,1)))
                )
    
        # Calculate std. dev points estimate
        fpRankings.loc[positionFilter, 'pts_stddev'] = (
                rankingsStdDevPointEstimate(
                        rfRegDict[position]
                        , fpRankings.loc[positionFilter, 'Avg']
                        , fpRankings.loc[positionFilter, 'Std Dev'])
                )
    
        # OOB & r^2 Results
        print(position
              , round(r2_score(
                      fpRankings.loc[positionFilter, 'pts']
                      , rfRegDict[position].predict(
                              (fpRankings.loc[positionFilter, 'Avg']
                                         .values
                                         .reshape(-1,1)))
                      )
                , 3)
              , round(rfRegDict[position].oob_score_, 3))
        
      
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    sns.scatterplot(x = 'Avg', y = 'pts', hue = 'position'
                    , data = fpRankings, ax = ax)
    ax.grid()
    
    sns.lineplot(x = 'Avg', y = 'pts_model_est', hue = 'position'
                    , data = fpRankings, ax = ax)
    
    # Store results
    dataDict.update({'fpRankings' : fpRankings})
    
    ### del(fpRankings)

#%% PRO FOOTBALL FOCUS PROJECTIONS
## ############################################################################

if dataDict.get('pffProjections', False) == True:
        
    
    # Load data
    pffProjections = pd.read_csv(
        os.path.join(
                'data',
                'projections_pff_{}_w{}.csv'.format(yr, week)
                )
        )
    
    # Convert to upper case
    pffProjections['position'] = [
            'DEF' if p.upper() == 'DST' else p.upper() 
            for p in pffProjections['position'].values
            ]
    
    # Rename teams for merging
    pffProjections['teamName'] = [
            teamAbrvsDict.get(team, team)
            for team in pffProjections['teamName'].values.tolist()
            ]
    
    
    # Change team name for defense
    pffProjections['playerName'] = [
            p[1] if p[2] == 'DEF' else p[0] 
            for p in pffProjections[
                    ['playerName', 'teamName', 'position']
                    ].values.tolist()
            ]
    
    # Create key for merging
    pffProjections['key'] = [
            fpRankingsKeyGen(keyList) for keyList in 
            pffProjections[['playerName', 'teamName', 'position']].values.tolist()
            ]
    

    pffProjections.rename(columns = {'fantasyPoints': 'pts'}, inplace = True)
    pffProjections['pts_stddev'] = 0


    dataDict.update({'pffProjections': pffProjections})
    
    ### del(pffProjections)

    
#%% LOAD FANTASY PROS PROJECTIONS DATA
## ############################################################################

if dataDict.get('fpProjections', False) == True:
    

    # Load and concat data for projections
    fpAllProjections = pd.concat([
        fantasyProsAllProjectionsDataLoad(position, yr, week) 
        for position in fantasyProsDict.keys()
        ], 
        sort = True
        )       
        
            
            
    # Calculate stats
    fpAllProjections = calculateProjectionStats(fpAllProjections)
    
    
    
    # Generate key for merging  
    fpAllProjections.loc[:, 'key'] = list(map(lambda keyList: 
        fpRankingsKeyGen(keyList)
        , fpAllProjections[['Player', 'Team', 'position']].values.tolist()
        ))      
        
        
    # fpAllProjectionsCols = ['FPTS', 'FPTS_std_dev_est']
    
    fpAllProjections.rename(
        columns =  {
            'FPTS': 'pts', 
            'FPTS_std_dev_est': 'pts_stddev'
            },
        inplace = True
        )
    
    dataDict['fpProjections'] = fpAllProjections
    
    ### del(fpAllProjectionsCols)
    
#%% PROJECTIONS UNCERTAINTY
## ############################################################################

if calculateUncertainty == True:    

    np.random.seed(1127)
    
    for i in range(5):
        dataDict['fpProjections'].loc[:, 'FPTS_rand_{}'.format(i)] = (
                dataDict['fpProjections']['FPTS'] + (
                        dataDict['fpProjections']['FPTS_std_dev_est'] 
                        * np.random.randn(dataDict['fpProjections'].shape[0])
                        )
                )
    
      
    # Columns for merging
    # fpAllProjectionsCols += ['FPTS_rand_{}'.format(i) for i in range(5)] 
            
    

#%% RANKINGS UNCERTAINTY
## ############################################################################

if calculateUncertainty == True:    
    
    zScore = norm.ppf(0.95)
    
    np.random.seed(1213)
    
    for i in range(5):
        
        # Create uncertainty around rank
        dataDict['fpRankings'].loc[:, 'Proj. Pts_rand_{}'.format(i)] = (
                dataDict['fpRankings']['Avg'] + (
                        dataDict['fpRankings']['Std Dev'] 
                        * zScore
                        * np.random.randn(dataDict['fpRankings'].shape[0])
                        )
                )
    
        # Estimate projected points based on new ranking
        #   Call RF model by position
        dataDict['fpRankings'].loc[:, 'Proj. Pts_rand_{}'.format(i)] = (
            
            list(map(lambda p: 
                rfRegDict.get(p[0]).predict(np.array(p[1]).reshape(1,-1))
                , dataDict['fpRankings'][['position', 'Proj. Pts_rand_{}'.format(i)]].values.tolist()
                ))
            )
    
    
    # Columns for merging
    # fpRankingsCols += ['Proj. Pts_rand_{}'.format(i) for i in range(4)] 


    
#%% COMBINE DATASETS
## ############################################################################
    

# Convert dataDict from dataframes to dictionaries
dataDict2 = {
    k: v.set_index('key')[['pts', 'pts_stddev']].to_dict('dict')
    for k,v in dataDict.items()
    if type(v) != bool
    }


dataDict['fpProjections'].groupby('key')['pts'].count().sort_values('pts', ascending = False).head()

dataDict['fpProjections']['key'].shape

# # of players required for each position
positionLimit = {'QB':1
                 , 'TE':1
                 , 'RB':2
                 , 'DEF':1
                 , 'WR':3
                 }

# Player status' to exclude
statusExclude = ['IR'
                 , 'SUSP'
                 , 'O'
                 , 'D'
#                 , 'Q'
                 ]





# Filter only eligble players
dataInput = copy.deepcopy(
        data[data['Injury Status'].map(lambda s: s not in statusExclude)]
        ).set_index('key')


dataInputDict = dataInput.to_dict('index')

# Add porjections to dataInputDict

[dataInputDict.get(player).update({k:pts}) for ]


dataInput = pd.concat([
    dataInput.set_index('key'),
    *[v.rename(columns = {'pts': '{}'.format(k)}).set_index('key')[k]
     for k,v in dataDict.items()
     if type(v) != bool
     ]
    ], axis = 1
    )
    
    
x= pd.concat([pd.DataFrame(
    (v.rename(columns = {'pts': '{}'.format(k)})
      .set_index('key')[k])
    )
     for k,v in dataDict.items()
     if type(v) != bool
    ], axis = 0)

dataInput = dataInput.set_index('key').merge(
        dataDict['fpRankings'].set_index('key')[['Avg', 'Best', 'Worst', 'Rank'
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
        dataDict['fpProjections'].set_index('key')[fpAllProjectionsCols]
        , how = 'left'
        , left_index = True
        , right_index = True
        )


dataInput = dataInput.merge(
        pd.DataFrame(dataDict['pffProjections'].set_index('key')[
                ['fantasyPoints', 'fantasyPoints_std_dev_est']])
        , how = 'left'
        , left_index = True
        , right_index = True
        )

# Fill empty projections with 0
dataInput.fillna(0, inplace = True)





#%% LP SETUP
## ############################################################################


# Convert to dictionary for LP
lpTargets = [
        'Proj. Pts', 
        'FPTS', 
#        'FPPG', 
        'fantasyPoints'
        ]

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
             'Team', 'Opponent', 'Salary', 'Rank'] 
            + lpTargets 
            + ['FPTS_std_dev_est', 'Proj. Pts_std_dev_est']
            ]
            )
    
    
    
    finalTeam[target]['finalTeamID'] = target
    finalTeam[target]['teamPoints'] = finalTeam[target][target].sum()
    
    # Calculate team overall projection std dev
    if target in ['FPTS', 'Proj. Pts']:
        stdDev = '{}_std_dev_est'.format(target)
    
        finalTeam[target]['teamPointsStdDev'] = (
                np.sqrt(np.square(finalTeam[target][stdDev].values).sum())
                )
        
    else:
        finalTeam[target]['teamPointsStdDev'] = 0


# View player counts
print(calcTeamPlayerStats(pd.concat(finalTeam.values()), target))



#%% TEAM OPTIMIZATION DERIVATIVES

#target = 'FPTS'

optimumTeamDict = {}
optimumTeamDerivatives = {}
teamPointIntersections = {}

# If true, remove all players from the first round of optimum team combinations
# and reprocess the optimum team selection process
calculate2ndRound = True

for target in ['FPTS', 'Proj. Pts', 'fantasyPoints']:

    # Generate all optimum team derivatives from base optimized team
    optimumTeamDict[target] = optimizedTeamDerivaties(
            optimumTeam = finalTeam[target]
            , dataInput = dataInput
            , dataInputDict = dataInputDict
            , budget = budget
            , target = target
            , positionLimit =  positionLimit)

    # Remove duplicates and convert to OHE dataframe 
    optimumTeamDerivatives[target] = pivotTeamsAndDropDuplicates(
            optimumTeamDict = optimumTeamDict[target]
            , optimumTeam = finalTeam[target]
            , target=target)


    # Drop duplicates from optimumTeamDict
    optimumTeamDict[target] = {
            k : optimumTeamDict[target][k] 
            for k in optimumTeamDerivatives[target].index.get_level_values(0)
            }

    print(calcTeamPlayerStats(
            pd.concat(optimumTeamDict[target].values())
            , target
            , groupbyList=['Position', 'Team', 'Last Name', 'First Name'])
            )
    


    # Identify player OHE columns
    playerCols = list(filter(lambda col: col.startswith('nfl')
                            , optimumTeamDerivatives[target].columns))
    
    
    # Calculate team point comparisons
    teamPointIntersections[target] = mapCalculateTeamDifference(
            optimumTeamDerivatives[target][playerCols]
            , method = 'intersection')
    
    
    # Plot distribution of teams & intersections
    fig, ax = plt.subplots(nrows = 1, ncols= 2, figsize = (10,6))
    sns.distplot(optimumTeamDerivatives[target][target], ax = ax[0])
    ax[0].grid()
    ax[0].set_title('Distribution of Team {}'.format(target), fontsize = 24)
    
    sns.distplot(list(filter(
            lambda p: p > 0 , teamPointIntersections[target].values.flatten()
            ))
        , ax = ax[1])
    ax[1].grid()
    ax[1].set_title('Distribution of Team Intersections', fontsize = 24)
    
    
    sns.jointplot(x = 'teamPointsStdDev'
                , y = target
                , data = optimumTeamDerivatives[target])


#    if calculate2ndRound == True:


#%% TEAM SELECTIONS

topTeams = {}
topTeamsDict = {}
teamPlayerStats = {}
clusterModels = {}

for target in [
        # 'FPTS', 
        # 'Proj. Pts', 
        'fantasyPoints'
        ]:
        
    # Cluster teams by their relationship to other teams
    optimumTeamDerivatives[target], clusterModels[target] = (
            clusterTeamsAndPlot(teamPointIntersections[target]
                                , optimumTeamDerivatives[target]
                                , target
                                 , k = 15)
            )
    
    
    
    # Get top teams from each cluster
    topTeams[target], topTeamsDict[target] = (
            selectClusterBestTeams(teams = optimumTeamDerivatives[target]
                                   , teamsDict = optimumTeamDict[target]
                                   , target = target
                                   , numTeams = 15)
            )
    
    
    teamPlayerStats[target] = (
            calcTeamPlayerStats(pd.concat(topTeamsDict[target].values()), target)
            )
    
    print(teamPlayerStats[target])

    # Write submissions
    writeTeamSubmissions(topTeamsDict = topTeamsDict[target],
                         target = target, week = week)






#%% DEV
## ############################################################################





def optimumTeamComboObjective(x):

    func = sum(chain(*[
            [x[i]*x[j]*c.iloc[j, i] for i in range(c.shape[1])] 
                for j in range(c.shape[0])]
        ))
    
    return func


def numOfTeams(x):
    return sum(x) - 3

def binaryConstraint(x):
    return sum([x[i]*(1-x[i]) for i in range(10)])

optimumTeamComboObjective(x0)



c = teamPointIntersections.iloc[:10,:10]
numTeams = 3
x0 = np.zeros(c.shape[0])
x0[:numTeams] = 0.5
bounds = list(repeat((0.0,1.0), c.shape[0]))

res = minimize(
        fun = optimumTeamComboObjective
        , x0 = x0
        , bounds = bounds
        , constraints = (
                {'type': 'eq', 'fun': lambda x: sum([x[i]*(1-x[i]) for i in range(10)])}
                , {'type': 'eq', 'fun': lambda x: sum(x) - 3}
                )
        , method = 'SLSQP'
        , tol = 1e-6)


list(map(lambda x: round(x,3),res.x))

sum(res.x)

sum(x0)



#%% pymo

from pyomo import environ as pe

from pyomo.environ import *

d = teamPointIntersections.iloc[:5, :5].to_dict('index')
teamPointIntersections.columns = range(212)
teamPointIntersections.index = range(212)

d = (teamPointIntersections
     .reset_index()
     .iloc[:5, :5]
     .melt(id_vars = 'index')
     .set_index(['index', 'variable'])
     .to_dict('index')
     )

model = ConcreteModel()

model.x = Var(list(d.keys()), domain = Binary)

model.value = Objective(expr = sum(model.x[i]*model.x[j]*d[i][j] for i in list(d.keys()) for j in list(d.keys()))
    , sense = maximize)

model.teams = Constraint(expr = sum(model.x[i] for i in list(d.keys())) == 3)

opt = SolverFactory('glpk').solve(model)

opt.write()



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
