
with open("clubelo\confif\max_update_coeff.txt") as f :
    HFA_UPDATE_COEFF =  int(f.read())


# def HFA_uptdate(H_elo, A_elo, Away_wins) :
#     Delta_elo = H_elo - A_elo
#     if(Away_wins == 1) :
#         update = - abs(Delta_elo) * HFA_UPDATE_COEFF 
#     if(Away_wins == 0) :
#         update = abs(Delta_elo) * HFA_UPDATE_COEFF
#     if(Away_wins == 0.5) : 
#         update = - Delta_elo * HFA_UPDATE_COEFF / 2  
#         # Si away > home Delta elo < 0 et donc égalité => COEFF augmente
#         # Mais pas autant que si il y avait eu une victoire d'ou le facteur "/2"
#         # /!\ c'est au pif

#     return update

# il faut ensuite ajouter "update" aux HFA respectifs de chaque équipe.


import pandas as pd
import numpy as np

path_matchs = "data/saison_2021_sans_cotes.csv"
path_current = "data/elo_FRA1_test.csv"

currentHFA = np.genfromtxt(path_current, delimiter=',', dtype=str)[0:, 0:2]

Matchs = np.genfromtxt(path_matchs, delimiter=',', dtype=str)[0:, 0:4]


def HFA_update_aftermatch(homeElo, awayElo, home_HFA, away_HFA, home_wins) :
    Delta_elo = homeElo - awayElo
    if(home_wins == 0) :
        home_new_HFA = home_HFA - abs(Delta_elo) * HFA_UPDATE_COEFF 
        away_new_HFA = away_HFA - (- abs(Delta_elo) * HFA_UPDATE_COEFF )
    if(home_wins == 1) :
        home_new_HFA = home_HFA + abs(Delta_elo) * HFA_UPDATE_COEFF
        away_new_HFA = away_HFA - (abs(Delta_elo) * HFA_UPDATE_COEFF)
    if(home_wins == 0.5) : 
        home_new_HFA = home_HFA - Delta_elo * HFA_UPDATE_COEFF / 2  
        away_new_HFA = away_HFA - (- Delta_elo * HFA_UPDATE_COEFF / 2)
    return home_new_HFA, away_new_HFA
    

def one_match_update(Matchs, currentHFA, match_nb) :
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
            homeElo = float(currentHFA[k][1])
        if currentHFA[k][0] == awayTeam :
            awayElo = float(currentHFA[k][1])
        
    new_home_elo, new_away_elo = HFA_update_aftermatch(homeElo, awayElo, home_wins)

    for k in range(len(currentHFA)) :
        if currentHFA[k][0] == homeTeam :
            currentHFA[k][1] = str(new_home_elo)
        if currentHFA[k][0] == awayTeam :
            currentHFA[k][1] = str(new_away_elo)

nb_de_matchs = len(Matchs)
for match_nb in range(nb_de_matchs) :
    one_match_update(Matchs, currentHFA, match_nb)

df = pd.DataFrame(currentHFA)
df.to_csv(path_current, index=False, header = False)



# il faut ensuite ajouter "update" aux HFA respectifs de chaque équipe.