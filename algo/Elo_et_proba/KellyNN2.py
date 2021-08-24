import numpy as np
from draw_proba_writer import draw_proba_writer
from Score_estimation1 import score_estimation2
import pandas as pd
import math

path_matchs = "data\saisons\\2020_2021_avec_PL_v1.csv"
# path_matchs = "data\saisons\\savga_PL_20_21.csv"
path_start_elo = "data\elo_start\elo_start_PL_2020.csv"

# path_matchs = "data\saisons\\2020_2021_avec_v4.csv"
# path_start_elo = "data\elo_start\elo_start_2020_v3.csv"
path_out_prob = "data\probs.csv"


# match_prediction = draw_proba_writer(path_matchs, path_start_elo, imported=True, 
# path_import_W1="data/W1_test4.dat", path_import_W2="data/W2_test4.dat")
maxCap = 0
# avg_cap = math.log(100)
bestFinCap = -100
best_avg_cap = - 10000
iteration = 0
maxIteration = 100
while(iteration < maxIteration) :
    realistic = True
    iteration+=1
    
    print(iteration)
    match_prediction_names, match_prediction_nbrs = draw_proba_writer(path_matchs, path_start_elo, saved=True, 
    path_save_W1="data/W1_test7.dat", path_save_W2="data/W2_test7.dat")
    good_sav_W1 = "data/W1_test11.dat"
    good_sav_W2 = "data/W2_test11.dat"

    nb_of_matchs = len(match_prediction_names)

    for match_nb in range(nb_of_matchs) :
        delta_elo = float(match_prediction_nbrs[match_nb][0])
        perct_restant = 1 - float(match_prediction_nbrs[match_nb][2])

        Eh = score_estimation2(delta_elo)

        home_percent = float(( 1 - Eh ) * perct_restant)
        away_percent = float(Eh * perct_restant)
        match_prediction_nbrs[match_nb][1] = float(home_percent)
        match_prediction_nbrs[match_nb][3] = float(away_percent)




    def mise(C, K, p) :
        if K == 0 :
            return 0
        if K*p < 1 :
            return 0
        f = p - (1 - p) / ( K - 1 )
        return C * f 



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



    def bestBet(esp1, esp2, esp3, k1, k2, k3) :
        # if (1 < max(esp1, esp2, esp3)) or ( max(esp1, esp2, esp3) > 1.5) :
        if (1 > max(esp1, esp2, esp3) or max(esp1, esp2, esp3) > 1.5) :
            return -1
        if max(k1, k2, k3) >= 6 :
            return -1
        if ( esp1 > 1.45 and esp2 > 1.45 ) or (esp1 > 1.45 and esp3 > 1.45 ) or ( esp2 > 1.45 and esp3 > 1.45 ) :
            return -1
        return max(esp1, esp2, esp3)

    #Meilleur des 3 paris
    best_G_array = np.array([bestBet(G_array_W[k],G_array_D[k], G_array_L[k], 
    K_array_W[k], K_array_D[k], K_array_L[k]) 
    for k in range(nb_match)])


    #Paris choisis [[G,K,p]]
    bets_taken = np.array([[best_G_array[k], 0, 0, 0 ] for k in range(nb_match)])
    for k in range(nb_match) :
        if (bets_taken[k][0] == G_array_W[k]) :
            bets_taken[k][1] = K_array_W[k]
            bets_taken[k][2] = p_array_W[k]
            bets_taken[k][3] = 1

        if (bets_taken[k][0] == G_array_D[k]) :
            bets_taken[k][1] = K_array_D[k]
            bets_taken[k][2] = p_array_D[k]
            bets_taken[k][3] = 2

        if (bets_taken[k][0] == G_array_L[k]) :
            bets_taken[k][1] = K_array_L[k]
            bets_taken[k][2] = p_array_L[k]
            bets_taken[k][3] = 3

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
    minCap = np.amin(C_array)
    if minCap <= 0 :
        avg_cap = -1000000000
    else :
        log_C = np.array([math.log(C_array[k]) for k in range(len(C_array))], dtype=float)
        avg_cap = np.mean(log_C)
    # print("avg_cap")
    # print(avg_cap)
    # print("best_avg_cap")
    # print(best_avg_cap)
    # print(realistic)
    finCap = C_array[-1]
    if avg_cap > best_avg_cap and realistic == True :
        # print("inIF")
        print(avg_cap)
        best_avg_cap = avg_cap
        copy_W1 = np.load("data/W1_test7.dat", allow_pickle=True)
        copy_W2 = np.load("data/W2_test7.dat", allow_pickle=True)
        parameters = np.array([copy_W1, copy_W2], dtype=object)
        parameters[0].dump(good_sav_W1)
        parameters[1].dump(good_sav_W2)



    # if (realistic) :
        # print("Benefices finaux : "+str(finCap-100))
        # print("Benefices max : "+str(maxCap-100))

    # df = pd.DataFrame(C_array)
    # df.to_csv(path_out, index=False, header = False)
