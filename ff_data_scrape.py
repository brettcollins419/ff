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



#%% FUNCTIONS

def downloadPFF(grade, season, week, savePath,
                chromePath = (
                        'C:/Program Files (x86)/Google/Chrome/'
                        'Application/chrome.exe %s')
                ):
                    
    '''Download data from premium fantasy football
    
        Do not have '''
    
    # Format week in case multiple weeks are provided
    if type(week) in (list, tuple):
        week = ','.join(week)
    
    # Format file path
    address = ('https://premium.pff.com/api/v1/facet/{grade}/summary'
                '?league=nfl&season={season}&week={week}&export=true'
                ).format(grade = grade, season = season, week = week)
    
    # Download file
    webbrowser.get(chromePath).open(address)
    

    # Find downloaded file (most recent file with file name)
    fileList = [
            (f.stat().st_mtime, f.name) 
            for f in os.scandir(savePath)
            ]
    
    fileList.sort(reverse = True)

    # file name check for special teams
    if grade =='special':
        grade = 'special_teams'

    # Most recent download    
    download = list(filter(
            lambda f: f[1].find('{grade}_summary'.format(grade=grade)) > -1
            , fileList)
        )[0]
    

    # Rename file with season and week
    newFileName = '{grade}_{season}_w{week}_summary.csv'.format(
                    grade = grade, season = season, week = week)
    
    os.rename('\\'.join([savePath, download[1]])
                , '\\'.join([savePath, newFileName])
                )   

    # Add season and week to file
    data = pd.read_csv('\\'.join([savePath, newFileName]))
    
    data.loc[:, 'season'] = season
    data.loc[:, 'week'] = week
    
    # Write csv
    data.to_csv('\\'.join([savePath, newFileName]), index = False)

    print('grade:', grade, 'season:', season, 'week:', week, 'time:', timer())

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
         

#%% DOWNLOAD DATA
## ############################################################################


seasonList = np.arange(2013, 2020).tolist()

gradesList = ('rushing'
              , 'passing'
              , 'defense'
              , 'field_goal'
              , 'special'
              , 'receiving'
              )


weekList = np.arange(1,18).tolist()


# Generate list for downloading
downloadList = list(product(seasonList, weekList, gradesList))

downloadList.sort(key = lambda x: (-x[0], -x[1], x[2]))

# Filter out future dates
currentWeek = 6
downloadList = list(
        filter(lambda x: (x[0] + (x[1]/17)) <= (2019 + (currentWeek/17))
            , downloadList)
        )



# Iterate through files
for season, week, grade in downloadList:
    try:
        downloadPFF(grade, season, week
                    , savePath = 'C:\\Users\\brett\\Downloads')
    except:
        print('Cannot download', season, week, grade)
        
    # Random sleep
    time.sleep(1 + np.random.rand())


#%% COMBINE FILES
    
for grade in gradesList:
    
    savePath = 'C:\\Users\\brett\\Downloads'
    dataList = os.listdir(savePath)
    
    dataList = list(filter(
            lambda f: f.find('{grade}'.format(grade=grade)) > -1
            , dataList))
    
    dataAgg = pd.concat([pd.read_csv('\\'.join([savePath, f]))
        for f in dataList]
        , axis = 0)
    
    dataAgg.to_csv('\\'.join([savePath, '{}_summary_agg.csv'.format(grade)])
                    , index = False)


#%% DEV
## ############################################################################

# MacOS
#chrome_path = 'open -a /Applications/Google\ Chrome.app %s'

# Windows
chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'

# Linux
# chrome_path = '/usr/bin/google-chrome %s'

## SUCCESS!!!
webbrowser.get(chrome_path).open(address)
