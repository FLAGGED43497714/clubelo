import numpy as np
import pandas as pd
from Elo_and_HFA_v2 import one_match_update


with open("config/k_value.txt") as f :
    K =  int(f.read())


path_matchs = "data\saisons\\2018_2019_FR1_sans.csv"
path_start_elo = "data\elo_start\elo_start_FR1_2018.csv"
path_current_elo_home = "data/elo_start/elo_FRA1_test2_home.csv"
path_current_elo_away = "data/elo_start/elo_FRA1_test2_away.csv"

#ces 3 arrays servent juste à construire les 2 prochains
currentElo_home_name = np.genfromtxt(path_start_elo, delimiter=',', dtype=str)[1:, 0]
currentElo_home_elo = np.genfromtxt(path_start_elo, delimiter=',', dtype=str)[1:, 1]
currentElo_away_elo = np.genfromtxt(path_start_elo, delimiter=',', dtype=str)[1:, 2]

nb_equipe = len(currentElo_home_name)

currentElo_home = np.array([[currentElo_home_name[k], currentElo_home_elo[k]] for k in range(nb_equipe)])
currentElo_away = np.array([[currentElo_home_name[k], currentElo_away_elo[k]] for k in range(nb_equipe)])


Matchs = np.genfromtxt(path_matchs, delimiter=',', dtype=str)[0:, 0:11]
# print(Matchs)
# print(currentElo_away)
# print(currentElo_home)


nb_de_matchs = len(Matchs)

for match_nb in range(nb_de_matchs) :
    currentElo_home, currentElo_away = one_match_update(Matchs, currentElo_home, currentElo_away, match_nb, K)

df_elo_home = pd.DataFrame(currentElo_home)
df_elo_home.to_csv(path_current_elo_home, index=False, header = False)

df_elo_away = pd.DataFrame(currentElo_away)
df_elo_away.to_csv(path_current_elo_away, index=False, header = False)

# faire les ajustement en fonction du score 