
with open("clubelo\confif\max_update_coeff.txt") as f :
    HFA_UPDATE_COEFF =  int(f.read())


def HFA_uptdate(H_elo, A_elo, Away_wins) :
    Delta_elo = H_elo - A_elo
    if(Away_wins == 1) :
        update = - abs(Delta_elo) * HFA_UPDATE_COEFF 
    if(Away_wins == 0) :
        update = abs(Delta_elo) * HFA_UPDATE_COEFF
    if(Away_wins == 0.5) : 
        update = - Delta_elo * HFA_UPDATE_COEFF / 2  
        # Si away > home Delta elo < 0 et donc égalité => COEFF augmente
        # Mais pas autant que si il y avait eu une victoire d'ou le facteur "/2"
        # /!\ c'est au pif

    return update

# il faut ensuite ajouter "update" aux HFA respectifs de chaque équipe.