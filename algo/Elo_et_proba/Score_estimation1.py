def score_estimation(team_elo, opponent_elo) :
    
    Eh = 1 / (1 + 10**( (opponent_elo - team_elo) / 400 ) )

    return Eh  

def score_estimation2(delta_elo) :
    
    Eh = 1 / (1 + 10**( (delta_elo) / 400 ) )

    return Eh  

def score_probas(team_elo, opponent_elo) :
    Eh = score_estimation(team_elo, opponent_elo)
    
    #P_H_wins = 


#faire les probas de victoire / égalité / défaite 