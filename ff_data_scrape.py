# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:49:06 2019

@author: u00bec7
"""


#%% PACKAGES

from datetime import datetime
import pandas as pd
import os
import re
import string
import time
import webbrowser
import numpy as np
from itertools import product, compress
import copy


#%% FUNCTIONS

def downloadPFF(grade, season, week, savePath
                , chromePath = (
                        'C:/Program Files (x86)/Google/Chrome/'
                        'Application/chrome.exe %s')
                ):
                    
    '''Download data from premium fantasy football.  If week is a tuple or list
        it will download the aggregated data over those weeks.
        
    
        '''
    
    # Format week in case multiple weeks are provided
    if type(week) in (list, tuple):
        week = ','.join(week)
    
    # Format file path
    address = ('https://premium.pff.com/api/v1/facet/{grade}/summary'
                '?league=nfl&season={season}&week={week}&export=true'
                ).format(grade = grade, season = season, week = week)
    
 
    # file name check for special teams
    if grade =='special':
        grade = 'special_teams'
    
    # Delete any historical download files in directory
    try:
        os.remove('{path}\\{grade}_summary.csv'.format(path=savePath, grade=grade))
    except: pass
    
    # Download file
    webbrowser.get(chromePath).open(address, autoraise=False)
    
    # Pause for download
    time.sleep(1)
    
    # Wait for file to download
    while '{grade}_summary.csv'.format(grade=grade) not in os.listdir(savePath):
        time.sleep(1)


    # Rename file with season and week
    newFileName = '{grade}_{season}_w{week}_summary.csv'.format(
                    grade = grade, season = season, week = week)
    
    os.rename('{path}\\{grade}_summary.csv'.format(path=savePath
                                                  , grade=grade)
                , '{path}\\{newFileName}'.format(path=savePath
                                                   , newFileName=newFileName)
                )   

    # Add season and week to file
    data = pd.read_csv(
            '{path}\\{newFileName}'.format(path=savePath
                                         , newFileName=newFileName)
            )
    
    data.loc[:, 'season'] = season
    data.loc[:, 'week'] = week
    
    # Write csv
    data.to_csv('\\'.join([savePath, newFileName]), index = False)

    print('grade:', grade, 'season:', season, 'week:', week, 'time:', timer())

    return


def aggregatePffData(grade, csvPath, savePath):
    
    dataList = os.listdir(csvPath)
    
    # Find all files of the same format
    dataList = list(filter(
            lambda f: f.find('{grade}'.format(grade=grade)) > -1
            , dataList))
    
    dataAgg = pd.concat([pd.read_csv('\\'.join([csvPath, f]))
        for f in dataList]
        , axis = 0
        , sort = True)
    
    # Ensure deduping
    dataAgg.drop_duplicates(inplace = True)
    
    dataAgg.to_csv('{}\\{}_summary_agg.csv'.format(savePath, grade)
                    , index = False) 
    
    print('file saved to {}\\{}_summary_agg.csv'.format(savePath, grade))

    return


def timer(sigDigits = 3):
    '''Timing function'''
    global startTime   
    if 'startTime' in globals().keys():
               
        calcTime =  time.time() - startTime
        startTime = time.time()
        
        return round(calcTime, sigDigits) 
        
    
    else:
        globals()['startTime'] = time.time()
         

#%% ALL HISTORICAL DATA DOWNLOAD
## ############################################################################

# Seasons to download
seasonList = np.arange(2013, 2020).tolist()

# Positions to download
gradesList = ('rushing'
              , 'passing'
              , 'defense'
              , 'field_goal'
              , 'special'
              , 'receiving'
              )

# Weeks to download
weekList = np.arange(1,18).tolist()


# Generate list for downloading
downloadList = list(product(seasonList, weekList, gradesList))


downloadList.sort(key = lambda x: (-x[0], -x[1], x[2]))

# Filter out future dates
currentWeek = 7
downloadList = list(
        filter(lambda x: (x[0] + (x[1]/17)) <= (2019 + (currentWeek/17))
            , downloadList)
        )


errorList = []

# Iterate through files
for season, week, grade in downloadList:
    try:
        downloadPFF(grade, season, week
                    , savePath = 'C:\\Users\\brett\\Downloads'
#                    , savePath = 'C:\\Users\\u00bec7\\Downloads'
                    )
        
        
    except:
        print('Cannot download', season, week, grade)
        errorList.append((season, week, grade))
        # Random sleep
        time.sleep(1)
        continue
        

#%% INCREMENTAL DATA DOWNLOAD
## ############################################################################

# Seasons to download
seasonList = [2019]

# Positions to download
gradesList = ('rushing'
              , 'passing'
              , 'defense'
              , 'field_goal'
              , 'special'
              , 'receiving'
              )

# Weeks to download
weekList = [8, 9]


# Generate list for downloading
downloadList = list(product(seasonList, weekList, gradesList))

downloadList.sort(key = lambda x: (-x[0], -x[1], x[2]))




errorList = []

# Iterate through files
for season, week, grade in downloadList:
    try:
        downloadPFF(grade, season, week
                    , savePath = 'C:\\Users\\brett\\Downloads'
#                    , savePath = 'C:\\Users\\u00bec7\\Downloads'
                    )
        
        
    except:
        print('Cannot download', season, week, grade)
        errorList.append((season, week, grade))
        # Random sleep
        time.sleep(1)
        continue
        



#%% COMBINE FILES
## ############################################################################

for grade in gradesList: 
    aggregatePffData(grade
                     , ('C:\\Users\\brett\\Documents\\Gridiron_Gradients\\'
                        'pff_data_weeks')
                     , ('C:\\Users\\brett\\Documents\\Gridiron_Gradients\\'
                        'pff_data')
                     )





#%% AGGREGATE DEFENSE & SPECIAL TEAMS
## ############################################################################
    
defense = pd.read_csv('C:\\Users\\brett\\Documents\\Gridiron_Gradients\\'
                      'pff_data\\defense_summary_agg.csv')

# Dictonary for aggregating metrics
defAggDict = {
        'assists' : np.sum
        , 'batted_passes' : np.sum
        , 'declined_penalties' : np.sum
        , 'forced_fumbles' : np.sum
        , 'grades_coverage_defense' : np.mean
        , 'grades_defense' : np.mean
        , 'grades_pass_rush_defense' : np.mean
        , 'grades_run_defense' : np.mean
        , 'grades_tackle' : np.mean
        , 'hits' : np.sum
        , 'hurries' : np.sum
        , 'interceptions' : np.sum
        , 'missed_tackles' : np.sum
        , 'pass_break_ups' : np.sum
        , 'penalties' : np.sum
        , 'receptions' : np.sum
        , 'sacks' : np.sum
        , 'snap_counts_coverage' : np.mean
        , 'snap_counts_pass_rush' : np.mean
        , 'snap_counts_run_defense' : np.mean
        , 'snap_counts_total' : np.mean
        , 'stops' : np.sum
        , 'tackles' : np.sum
        , 'targets' : np.sum
        , 'total_pressures' : np.sum
        , 'touchdowns' : np.sum
        , 'yards' : np.sum
        , 'yards_after_catch' : np.sum
}

# Aggregate defense stats
defense = (
        defense.groupby(['season', 'week', 'team_name'])
               .agg(defAggDict)
               .reset_index()
               )

# Compbine turnovers
defense.loc[:, 'turnovers'] = (
        defense[['forced_fumbles', 'interceptions']].sum(axis = 1)
        )

# Combine penalties
defense.loc[:, 'all_penalties'] = (
        defense[['penalties', 'declined_penalties']].sum(axis = 1)
        )

defense.to_csv('C:\\Users\\brett\\Documents\\Gridiron_Gradients\\'
               'pff_data\\defense_team_summary_agg.csv', index = False)


specialTeams = pd.read_csv('C:\\Users\\brett\\Documents\\Gridiron_Gradients\\'
                           'pff_data\\special_summary_agg.csv')

specialAggDict = {
        'assists' : np.sum
        , 'declined_penalties' : np.sum
        , 'grades_fgep_kicker' : np.mean
        , 'grades_kick_return' : np.mean
        , 'grades_kickoff_kicker' : np.mean
        , 'grades_misc_st' : np.mean
        , 'grades_punt_return' : np.mean
        , 'grades_punter' : np.mean
        , 'missed_tackles' : np.sum
        , 'penalties' : np.sum
        , 'tackles' : np.sum
        }


specialTeams = (
        specialTeams.groupby(['season', 'week', 'team_name'])
                    .agg(specialAggDict)
                    .reset_index()
                    )

# Combine penalties
specialTeams.loc[:, 'all_penalties'] = (
        specialTeams[['penalties', 'declined_penalties']].sum(axis = 1)
        )

specialTeams.to_csv('C:\\Users\\brett\\Documents\\Gridiron_Gradients\\pff_data\\'
                    'special_team_summary_agg.csv', index = False)

#%% MATCHUPS DATA
## ############################################################################

# Load games
games = pd.read_csv('C:\\Users\\brett\\Documents\\Gridiron_Gradients\\'
                    'data\\spreadspoke_scores.csv')


# Load team abbreviations
teams = pd.read_csv('C:\\Users\\brett\\Documents\\Gridiron_Gradients\\'
                    'data\\nfl_teams.csv')


# Add team abbreviations to games
for team in ('home', 'away'):
    games = (games.set_index('team_{}'.format(team))
                  .merge(pd.DataFrame(teams.set_index('team_name')['team_id'])
                          , how = 'inner'
                          , left_index = True
                          , right_index = True
                          )
                  .reset_index()
                  .rename(columns = {'team_id':'team_id_{}'.format(team)
                                     , 'index':'team_'.format(team)})
                  
                  )

## Create team / opponent dataframe for referencing
    
gameCols = ['schedule_season'
            , 'schedule_week'
            , 'team_id_home'
            , 'team_id_away'
            , 'spread_favorite'
            , 'team_favorite_id'
            , 'over_under_line'
            , 'weather_temperature'
            , 'weather_wind_mph'
            , 'weather_detail']
    
# Make the home team the focus
home = copy.copy(games[gameCols])
home.rename(
        columns = {'team_id_home':'team', 'team_id_away':'opponent'}
        , inplace = True)

# Make the away team the focus
away = copy.copy(games[gameCols])
away.rename(
        columns = {'team_id_home':'opponent', 'team_id_away':'team'}
        , inplace = True)


# Concat home and away to get complete view for each team
gameLookup = pd.concat([home,away], sort = True)


# Make spread relative to the base team
gameLookup.loc[:, 'spread_team']= gameLookup.apply(lambda r: 
    r['spread_favorite'] * -1 if r['team'] != r['team_favorite_id']
    else r['spread_favorite']
    , axis = 1)
    
# Add boolean for if team is favorite
gameLookup.loc[:, 'is_favorite'] = [
        (s < 0)*1 for s in gameLookup['spread_team'].values
        ]

# Drop spread columns
gameLookup.drop(['spread_favorite', 'team_favorite_id'], axis = 1, inplace = True)

# Filter only to regular season
gameLookupReg = copy.copy(
        gameLookup.loc[[len(week) <= 2 for week in gameLookup['schedule_week']], :]
        )

gameLookupReg.loc[:, 'schedule_week'] = gameLookupReg.loc[:, 'schedule_week'].map(int)

# Reset index for unique index
gameLookupReg.reset_index(drop = True, inplace = True)

#%% MERGE OPPONENT WITH PLAYER STATS

# Dictionary for team name conversions between datasets
# (all other teams are the same between the datasets)
# Key is PFF data and value is Matchups data
teamAbrvsDict = {
        'ARZ' : 'ARI'
        , 'BLT' : 'BAL'
        , 'CLV' : 'CLE'
        , 'HST' : 'HOU'
        , 'LA' : 'LAR'
        , 'SD' : 'LAC'
        , 'SL' : 'LAR'
        }

# Merge game info and save data
for f in os.listdir('pff_data'):
    dataAgg = pd.read_csv('pff_data\\{}'.format(f))
    
    dataAgg.loc[:, 'team_name_games'] = [
            teamAbrvsDict.get(team, team) for team in dataAgg['team_name']
            ]
    
    
    dataAgg = (
        dataAgg.merge(gameLookupReg
                      , how = 'inner'
                      , left_on = ['season', 'week', 'team_name_games']
                      , right_on = ['schedule_season', 'schedule_week', 'team'])
        ).drop(['schedule_season', 'schedule_week', 'team'], axis = 1)
    
    
    dataAgg.to_csv('pff_data\\{}'.format(f), index = False)
  
    

    
#%% DEV
## ############################################################################


defense = pd.read_csv('\\'.join([savePath, '{}_summary_agg.csv'.format('defense')]))



x = dataAgg.groupby(['season','player_id', 'player']).cumsum()


