import numpy as np
import pandas as pd

from Elo_and_HFA_update import one_match_update

with open("config\max_update_coeff.txt") as f :
    HFA_UPDATE_COEFF =  float(f.read())

with open("config/k_value.txt") as f :
    K =  int(f.read())


path_matchs = "data/saison_2021_sans_cotes.csv"
path_start_elo = "data/elo_start_2020.csv"
path_current_elo = "data/elo_FRA1_test.csv"
path_current_HFA = "data/HFA_FRA1_test.csv"
path_out = "data/egaliteDeltaElo.csv"

currentHFA = np.genfromtxt(path_current_HFA, delimiter=',', dtype=str)[0:, 0:2]
currentElo = np.genfromtxt(path_start_elo, delimiter=',', dtype=str)[0:, 0:2]
Matchs = np.genfromtxt(path_matchs, delimiter=',', dtype=str)[0:, 0:4]

nb_de_matchs = len(Matchs)

res = np.array([[0, 0] for k in range(nb_de_matchs)], dtype= float)

for match_nb in range(nb_de_matchs) :

    homeTeam = Matchs[match_nb][0]
    awayTeam = Matchs[match_nb][1]

    for k in range(len(currentElo)) :
        if currentElo[k][0] == homeTeam :
            homeElo = float(currentElo[k][1])
        if currentElo[k][0] == awayTeam :
            awayElo = float(currentElo[k][1])

    deltaElo = abs(homeElo - awayElo)

    res[match_nb][0] = deltaElo

    if Matchs[match_nb][2] == Matchs[match_nb][3] :
        res[match_nb][1] = 1

    one_match_update(Matchs, currentElo, currentHFA, match_nb, K, HFA_UPDATE_COEFF)

df = pd.DataFrame(res)
df.to_csv(path_out, index=False, header = False)


