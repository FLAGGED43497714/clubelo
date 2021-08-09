import numpy as np
import pandas as pd
from Elo_and_HFA_update import one_match_update

with open("config\max_update_coeff.txt") as f :
    HFA_UPDATE_COEFF =  float(f.read())

with open("config/k_value.txt") as f :
    K =  int(f.read())


path_matchs = "data/saison_2021_sans_cotes.csv"
path_start_elo = "data\elo_start_2020.csv"
path_current_elo = "data/elo_FRA1_test.csv"
path_current_HFA = "data/HFA_FRA1_test.csv"


currentHFA = np.genfromtxt(path_current_HFA, delimiter=',', dtype=str)[0:, 0:2]
currentElo = np.genfromtxt(path_start_elo, delimiter=',', dtype=str)[0:, 0:2]
Matchs = np.genfromtxt(path_matchs, delimiter=',', dtype=str)[0:, 0:4]



nb_de_matchs = len(Matchs)
for match_nb in range(nb_de_matchs) :
    one_match_update(Matchs, currentElo, currentHFA, match_nb, K, HFA_UPDATE_COEFF)

df_elo = pd.DataFrame(currentElo)
df_elo.to_csv(path_current_elo, index=False, header = False)

df_HFA = pd.DataFrame(currentHFA)
df_HFA.to_csv(path_current_HFA, index=False, header = False)

# faire les ajustement en fonction du score 