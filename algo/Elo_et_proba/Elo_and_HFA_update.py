from Score_estimation1 import score_estimation
import numpy as np
import pandas as pd



def elo_update_aftermatch(home_elo, away_elo, homeHFA, home_wins, K) :
    new_home_elo = home_elo + K * (home_wins - score_estimation(home_elo + homeHFA, away_elo - homeHFA) )
    new_away_elo = away_elo + K * ( (1 - home_wins) - score_estimation(away_elo - homeHFA, home_elo + homeHFA))

    return new_home_elo, new_away_elo

def HFA_update_aftermatch(homeElo, awayElo, homeHFA, home_wins, HFA_UPDATE_COEFF, K) :
    adj = K * (home_wins - score_estimation(homeElo + homeHFA, awayElo - homeHFA) )
    if(home_wins == 0) :
        home_new_HFA = homeHFA + adj * HFA_UPDATE_COEFF 
    if(home_wins == 1) :
        home_new_HFA = homeHFA + adj * HFA_UPDATE_COEFF
    if(home_wins == 0.5) : 
        home_new_HFA = homeHFA + adj * HFA_UPDATE_COEFF / 2  
    return home_new_HFA



def one_match_update(Matchs, currentElo, currentHFA, match_nb, K, HFA_UPDATE_COEFF) :
    homeTeam = Matchs[match_nb][0]
    awayTeam = Matchs[match_nb][1]

    homeScore = int(Matchs[match_nb][2])
    awayScore = int(Matchs[match_nb][3])
        
    if homeScore > awayScore :
        home_wins = 1 
    if homeScore == awayScore :
        home_wins = 0.5 
    if homeScore < awayScore :
        home_wins = 0

    for k in range(len(currentHFA)) :
        if currentHFA[k][0] == homeTeam :
            homeHFA = float(currentHFA[k][1])

    for k in range(len(currentElo)) :
        if currentElo[k][0] == homeTeam :
            homeElo = float(currentElo[k][1])
        if currentElo[k][0] == awayTeam :
            awayElo = float(currentElo[k][1])
        
    new_home_HFA = HFA_update_aftermatch(homeElo, awayElo, homeHFA, home_wins, HFA_UPDATE_COEFF, K)

    for k in range(len(currentHFA)) :
        if currentHFA[k][0] == homeTeam :
            currentHFA[k][1] = str(new_home_HFA)

    new_home_elo, new_away_elo = elo_update_aftermatch(homeElo, awayElo, homeHFA, home_wins, K)

    for k in range(len(currentElo)) :
        if currentElo[k][0] == homeTeam :
            currentElo[k][1] = str(new_home_elo)
        if currentElo[k][0] == awayTeam :
            currentElo[k][1] = str(new_away_elo)


