from Score_estimation1 import score_estimation
import numpy as np
import pandas as pd

with open("config/k_value.txt") as f :
    K =  int(f.read())

def elo_update_aftermatch(home_elo, away_elo, home_wins) :
    new_home_elo = home_elo + K * (home_wins - score_estimation(home_elo, away_elo) )
    print("adj : "+str(K * (home_wins - score_estimation(home_elo, away_elo) )))
    new_away_elo = away_elo + K * ( (1 - home_wins) - score_estimation(away_elo, home_elo))
    print("adj : "+str(K * ( (1 - home_wins) - score_estimation(away_elo, home_elo))))

    return new_home_elo, new_away_elo


path_matchs = "data/saison_2021_sans_cotes.csv"
path_current = "data/elo_FRA1_test.csv"


currentElo = np.genfromtxt(path_current, delimiter=',', dtype=str)[0:, 0:2]


Matchs = np.genfromtxt(path_matchs, delimiter=',', dtype=str)[0:, 0:4]


def one_match_update(Matchs, currentElo, match_nb) :
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

    for k in range(len(currentElo)) :
        if currentElo[k][0] == homeTeam :
            homeElo = float(currentElo[k][1])
        if currentElo[k][0] == awayTeam :
            awayElo = float(currentElo[k][1])
        
    new_home_elo, new_away_elo = elo_update_aftermatch(homeElo, awayElo, home_wins)

    for k in range(len(currentElo)) :
        if currentElo[k][0] == homeTeam :
            currentElo[k][1] = str(new_home_elo)
        if currentElo[k][0] == awayTeam :
            currentElo[k][1] = str(new_away_elo)

nb_de_matchs = len(Matchs)
for match_nb in range(nb_de_matchs) :
    one_match_update(Matchs, currentElo, match_nb)

df = pd.DataFrame(currentElo)
df.to_csv(path_current, index=False, header = False)



# faire les ajustement en fonction du score 