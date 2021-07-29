from math import *
from matplotlib import pyplot as plt

Z = [0, 1] # loose = 0 ; win = 1 

C_0 = 100

P = [60/100, 40/100]

K = 3

T = (K * P[1])**2 # T = espérance pour le test 

Q = 0.02

C_tree = [C_0]
P_tree =  [1]

depth = 20

nb_father = 1
father_by_layer = 1
for k in range(depth - 1) :
    father_by_layer *= 2
    nb_father += father_by_layer

for k in range(nb_father) :
    C_n = C_tree[k]
    P_n = P_tree[k]

    C_loose = C_n - Q*T*C_n
    P_loose = P_n * P[0]

    C_win = C_n * (1 + Q*T*(K - 1))
    P_win = P_n * P[1]
    
    C_tree += [C_loose]
    C_tree += [C_win]
    P_tree += [P_loose]
    P_tree += [P_win]

s = 0

for k in range(len(C_tree)//2, len(C_tree)) :
    s += C_tree[k] * P_tree[k]

finalC = C_tree[len(C_tree)//2 : len(C_tree)]
finalP = P_tree[len(C_tree)//2 : len(C_tree)]

plt.scatter(finalC, finalP)
plt.scatter(s, 0)
plt.show()
# print("exp")


# print(C_tree)
print(s)


