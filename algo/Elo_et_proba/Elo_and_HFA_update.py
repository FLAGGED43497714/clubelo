from Score_estimation1 import score_estimation
import numpy as np
import pandas as pd


with open("config\max_update_coeff.txt") as f :
    HFA_UPDATE_COEFF =  float(f.read())

with open("config/k_value.txt") as f :
    K =  int(f.read())


path_matchs = "data/saison_2021_sans_cotes.csv"
path_current_elo = "data/elo_FRA1_test.csv"
path_current_HFA = "data/HFA_FRA1_test.csv"


currentHFA = np.genfromtxt(path_current_HFA, delimiter=',', dtype=str)[0:, 0:2]
currentElo = np.genfromtxt(path_current_elo, delimiter=',', dtype=str)[0:, 0:2]
Matchs = np.genfromtxt(path_matchs, delimiter=',', dtype=str)[0:, 0:4]


def elo_update_aftermatch(home_elo, away_elo, homeHFA, home_wins) :
    new_home_elo = home_elo + K * (home_wins - score_estimation(home_elo + homeHFA, away_elo - homeHFA) )
    new_away_elo = away_elo + K * ( (1 - home_wins) - score_estimation(away_elo - homeHFA, home_elo + homeHFA))

    return new_home_elo, new_away_elo

def HFA_update_aftermatch(homeElo, awayElo, home_HFA, home_wins) :
    Delta_elo = homeElo +home_HFA - ( awayElo - home_HFA )
    if(home_wins == 0) :
        home_new_HFA = home_HFA - abs(Delta_elo) * HFA_UPDATE_COEFF 
    if(home_wins == 1) :
        home_new_HFA = home_HFA + abs(Delta_elo) * HFA_UPDATE_COEFF
    if(home_wins == 0.5) : 
        home_new_HFA = home_HFA - Delta_elo * HFA_UPDATE_COEFF / 2  
    return home_new_HFA



def one_match_update(Matchs, currentElo, currentHFA, match_nb) :
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
        
    new_home_HFA = HFA_update_aftermatch(homeElo, awayElo, homeHFA, home_wins)

    for k in range(len(currentHFA)) :
        if currentHFA[k][0] == homeTeam :
            currentHFA[k][1] = str(new_home_HFA)

    new_home_elo, new_away_elo = elo_update_aftermatch(homeElo, awayElo, homeHFA, home_wins)

    for k in range(len(currentElo)) :
        if currentElo[k][0] == homeTeam :
            currentElo[k][1] = str(new_home_elo)
        if currentElo[k][0] == awayTeam :
            currentElo[k][1] = str(new_away_elo)


nb_de_matchs = len(Matchs)
for match_nb in range(nb_de_matchs) :
    one_match_update(Matchs, currentElo, currentHFA, match_nb)

df_elo = pd.DataFrame(currentElo)
df_elo.to_csv(path_current_elo, index=False, header = False)

df_HFA = pd.DataFrame(currentHFA)
df_HFA.to_csv(path_current_HFA, index=False, header = False)

# faire les ajustement en fonction du score 