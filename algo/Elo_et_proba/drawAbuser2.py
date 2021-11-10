import numpy as np
from draw_proba_writer import draw_proba_writer
from Score_estimation1 import score_estimation2
import pandas as pd
import math
from matplotlib import pyplot as plt

# path_matchs = "data\saisons\\2015_2016_FR2.csv"
# path_start_elo = "data\elo_start\elo_start_FR2_2015_2016.csv"

# path_matchs = "data\saisons\\2015_2016_FR1.csv"
# path_start_elo = "data\elo_start\elo_start_2015_2016_FR1_v2.csv"

# path_matchs = "data\saisons\\2016_2017_FR1.csv"
# path_start_elo = "data\elo_start\elo_start_2016_2017_FR1_v2.csv"

# path_matchs = "data\saisons\\2016_2017_FR2.csv"
# path_start_elo = "data\elo_start\elo_start_FR2_2016_2017.csv"

# path_matchs = "data\saisons\\2017_2018_FR1.csv"
# path_start_elo = "data\elo_start\elo_start_FR1_2017_2018_v2.csv"

# path_matchs = "data\saisons\\2018_2019_FR1.csv"
# path_start_elo = "data\elo_start\elo_start_FR1_2018_2019.csv"

# path_matchs = "data\saisons\\2019_2020_FR1.csv"
# path_start_elo = "data\elo_start\elo_start_2019_v2.csv"

# path_matchs = "data\saisons\SP1_2021_2022.csv"
# path_start_elo = "data\elo_start\elo_start_SP1_2021_2022.csv"

# path_matchs = "data\saisons\SP1_2020_2021_clean.csv"
# path_start_elo = "data\elo_start\elo_start_SP1_2020_2021.csv"

path_matchs = "data\saisons\\2021_2022_FR1_forbacktest.csv"
path_start_elo = "data\elo_start\elo_start_2021.csv"

# path_matchs = "data\saisons\\2020_2021_avec_v4.csv"
# path_start_elo = "data\elo_start\elo_start_2020_v3.csv"

# path_matchs = "data\saisons\\2020_2021_avec_PL_v1.csv"
# path_start_elo = "data\elo_start\elo_start_PL_2020.csv"

# path_matchs = "data\saisons\\2021_2022_PL.csv"
# path_start_elo = "data\elo_start\elo_start_PL_2021.csv"


path_out_prob = "data\probs.csv"

realistic = True


match_prediction_names, match_prediction_nbrs = draw_proba_writer(path_matchs, path_start_elo, imported=True, 
path_import_W1="data/NN_saved/W1_test11.dat", path_import_W2="data/NN_saved/W2_test11.dat")

match_prediction_names2, match_prediction_nbrs2 = draw_proba_writer(path_matchs, path_start_elo, imported=True, 
path_import_W1="data/NN_saved/W1_test8.dat", path_import_W2="data/NN_saved/W2_test8.dat")
# maxCap = 0
# while(maxCap < 1000 or realistic == False) :
# match_prediction = draw_proba_writer(path_matchs, path_start_elo, saved=True, 
# path_save_W1="data/W1_test4.dat", path_save_W2="data/W2_test4.dat")

nb_of_matchs = len(match_prediction_names)
delta_elo_array = np.array([0 for k in range(nb_of_matchs)], dtype=float)
for match_nb in range(nb_of_matchs) :
    delta_elo = float(match_prediction_nbrs[match_nb][0])
    delta_elo_array[match_nb] = delta_elo
    perct_restant = 1 - float(match_prediction_nbrs[match_nb][2])

    Eh = score_estimation2(delta_elo)

    home_percent = float(( 1 - Eh ) * perct_restant)
    away_percent = float(Eh * perct_restant)
    match_prediction_nbrs[match_nb][1] = float(home_percent)
    match_prediction_nbrs[match_nb][3] = float(away_percent)

# df = pd.DataFrame(match_prediction)



def mise(C, K, p) :
    if K == 0 :
        return 0
    if K*p < 1 :
        return 0
    f = p - (1 - p) / ( K - 1 )
    return C*f



path_out = 'data/testfile1.csv'
#K pour les cotes, p pour les probas
#W = win, D = draw, L = loose

# Ce qui ne change pas 
K_array_W = np.genfromtxt(path_matchs, delimiter=",")[:, 7]
K_array_D = np.genfromtxt(path_matchs, delimiter=",")[:, 8]
K_array_L = np.genfromtxt(path_matchs, delimiter=",")[:, 9]
winner_array = np.genfromtxt(path_matchs, delimiter=",")[:, 10]


# Ce qui dépend du NN
p_array_W = np.array([0 for match_nb in range(nb_of_matchs)], dtype=float)
p_array_D = np.array([0 for match_nb in range(nb_of_matchs)], dtype=float)
p_array_L = np.array([0 for match_nb in range(nb_of_matchs)], dtype=float)
for match_nb in range(nb_of_matchs) :
    p_array_W[match_nb] = match_prediction_nbrs[match_nb][1]
    p_array_D[match_nb] = match_prediction_nbrs[match_nb][2]
    p_array_L[match_nb] = match_prediction_nbrs[match_nb][3]

p_array_W_real = np.array([0 for match_nb in range(nb_of_matchs)], dtype=float)
p_array_D_real = np.array([0 for match_nb in range(nb_of_matchs)], dtype=float)
p_array_L_real = np.array([0 for match_nb in range(nb_of_matchs)], dtype=float)
for match_nb in range(nb_of_matchs) :
    p_array_W_real[match_nb] = match_prediction_nbrs2[match_nb][1]
    p_array_D_real[match_nb] = match_prediction_nbrs2[match_nb][2]
    p_array_L_real[match_nb] = match_prediction_nbrs2[match_nb][3]


if np.amax(p_array_W) > 1 :
    realistic = False

if np.amax(p_array_D) > 1 :
    realistic = False

if np.amax(p_array_L) > 1 :
    realistic = False


nb_match = len(K_array_W)



#Espérance des paris 
G_array_W = np.array([K_array_W[k] * p_array_W[k] for k in range(nb_match)])
G_array_D = np.array([K_array_D[k] * p_array_D[k] for k in range(nb_match)])
G_array_L = np.array([K_array_L[k] * p_array_L[k] for k in range(nb_match)])

preds = np.array([[p_array_W[k],p_array_D[k],p_array_L[k], G_array_W[k], G_array_D[k], G_array_L[k]
] for k in range(nb_of_matchs) ], dtype=float)
df = pd.DataFrame(preds)
df.to_csv(path_out_prob, index=False, header = False)



def bestBet(esp1, esp2, esp3, delta_elo) :
    if max(esp1, esp3) < 1.5 :
        if 1 < esp2 < 1.55 and delta_elo < 0 :
            return esp2


#Abuse des "égalités négatives"
best_G_array = np.array([bestBet(G_array_W[k],G_array_D[k],G_array_L[k], 
delta_elo_array[k]) 
for k in range(nb_match)])

#Paris choisis [[G,K,p]]
bets_taken = np.array([[best_G_array[k], 0, 0, 0 ] for k in range(nb_match)])

for k in range(nb_match) :
    
    if (bets_taken[k][0] == G_array_D[k]) :
        bets_taken[k][1] = K_array_D[k]
        bets_taken[k][2] = p_array_D_real[k]
        bets_taken[k][3] = 2

    
#C = Capital 
C_array = np.array([100 for k in range(nb_match+1)], dtype=float)

for i in range(nb_match) : 
    s = mise(C_array[i], bets_taken[i][1], bets_taken[i][2])
    if (bets_taken[i][3] == winner_array[i]) :
        res = 1 
    else :
        res = 0
    C_array[i+1] = C_array[i] -s + s * (res) * bets_taken[i][1]

maxCap = np.amax(C_array)
finCap = C_array[-1]

log_C = np.array([math.log(C_array[k]) for k in range(len(C_array))])
avg_log = np.mean(log_C)
if np.amin(C_array < 0) :
    avg_log = -10000    

print("realistic = " +str(realistic))
if (realistic) :
    print("Benefices finaux : "+str(finCap-100))
    print("Benefices max : "+str(maxCap-100))
    print("avg_log mean = " + str(avg_log))
    print("avg_log_mean donne : " + str(math.exp(avg_log)))

plt.plot(log_C)
plt.show()
plt.plot(C_array)
plt.show()

df = pd.DataFrame(C_array)
df.to_csv(path_out, index=False, header = False)
