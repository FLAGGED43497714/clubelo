import numpy as np
import pandas as pd
from Elo_and_HFA_v2 import one_match_update

# path_matchs = "data/saisons/saison_2021_sans_cotes.csv"
# path_start_elo = "data/elo_start/elo_start_2020_v3.csv"
# path_out = "data/egaliteDeltaElo.csv"

def dataSetDraw(path_matchs, path_start_elo) :

    with open("config/k_value.txt") as f :
        K =  int(f.read())


    Matchs = np.genfromtxt(path_matchs, delimiter=',', dtype=str)[0:, 0:4]

    Elo_name = np.genfromtxt(path_start_elo, delimiter=',', dtype=str)[1:, 0]
    Elo_home = np.genfromtxt(path_start_elo, delimiter=',', dtype=str)[1:, 1]
    Elo_away = np.genfromtxt(path_start_elo, delimiter=',', dtype=str)[1:, 2]

    nb_de_matchs = len(Matchs)

    nb_d_equipe = len(Elo_name)

    currentElo_home = [[Elo_name[k],Elo_home[k]] for k in range(nb_d_equipe)]
    currentElo_away = [[Elo_name[k],Elo_away[k]] for k in range(nb_d_equipe)]


    res = np.array([[0, 0] for k in range(nb_de_matchs)], dtype= float)

    for match_nb in range(nb_de_matchs) :
        homeTeam = Matchs[match_nb][0]
        awayTeam = Matchs[match_nb][1]



        for k in range(nb_d_equipe) :
            if currentElo_home[k][0] == homeTeam :
                homeElo = float(currentElo_home[k][1])
            if currentElo_away[k][0] == awayTeam :
                awayElo = float(currentElo_away[k][1])

        deltaElo = abs(homeElo - awayElo)

        res[match_nb][0] = deltaElo

        if Matchs[match_nb][2] == Matchs[match_nb][3] :
            res[match_nb][1] = 1

        currentElo_home, currentElo_away = one_match_update(Matchs, currentElo_home, currentElo_away, match_nb, K)

    return res
    # df = pd.DataFrame(res)
    # df.to_csv(path_out, index=False, header = False)


