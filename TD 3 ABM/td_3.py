import numpy as np 
import random as rd 
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 

# x_t = p_t - p_t*
prix_fondamental=1

NIT=1000 # number of time step

def n_h_t(U_t,beta): 
    """ 
    Entries : U_t la mesure de la performance des h modèles au temps t et beta la vitesse d'apprentissage des agents
    ================================================
    Aim : Calculer la liste des probas de suivre un modèle au temps t+1
    ================================================
    Outputs : liste des proba de suivre un modèle pour tout h au temps t+1
    """

    U_t=np.array(U_t)
    N_h_t=[np.exp(beta*U_t[i])/np.sum(np.exp(beta*U_t)) for i in range(len(U_t))]
    return N_h_t

def f_0(g):
    return 0 
def f_1(g):
    return g 
def f_2(g) : 
    return -f_1(g)

strat=[f_0,f_1,f_2]## l'ensemble des fonctions stratégies 

def E_constante(x,strat_function,g):
    return strat_function(g) ## calcul E pour une stratégie données


def U_h_t(U_h_t_1,lamb,r,p_t):
    """ 
    Entries : 
    ================================================
    Aim : 
    ================================================
    Outputs : 
    """
    U_h=[]
    for i in range(len(U_h_t_1)):## selon les h 
        first_part=U_h_t_1[i]*(1-lamb)
        second_part=lamb*(p_t[-1]-(1-r)*p_t[-2])*E_constante(p_t[-1]-(1+r)*p_t[-2],strat[i],G)
        U_h.append(first_part+second_part)
    return U_h


def predict_x(r,x,n):
    somme=0
    for i in range(len(n)): 
        somme+=n[i]*E_constante(x,strat[i],G)
    X=(1/(r+1))*somme
    return(X)

################################################################################################
#########################        VOir l'impact de beta=1 q1  #################################
################################################################################################
lambda_cst=0.1
R=0.01
Beta=1
G=2
X,U,N=[1,1],[[1,1,1],[1,1,1]],[[1/3,1/3,1/3],[1/3,1/3,1/3]]
for i in range(2,NIT):
    U.append(U_h_t(U[-1],lambda_cst,R,X))
    N.append(n_h_t(U[-2],Beta))##attention on met -2 car on vient de calculer Ut juste au dessus or c'est Ut-1 qui doit rentrer
    X.append(predict_x(R,X,N[-1]))
'''
plt.figure()
plt.plot(X)
plt.grid()
plt.title(r"$X_t$ evolution with beta={}".format(Beta))
plt.xlabel("Time")
plt.ylabel(r"$X_t$")
plt.legend()
plt.show()

plt.figure()
X_t=X[1:]
X_t_1=X[:-1]
plt.scatter(X_t_1,X_t)
plt.scatter(X_t[-1],X_t[-1],color='red')
plt.xlabel("X_t-1")
plt.ylabel("X_t")
plt.title(r"$X_t$ evolution with $X_{t-1}$ with beta = 1")
plt.grid()
plt.show()
'''

################################################################################################
#########################        VOir l'impact de beta  #################################
################################################################################################

'''
Meta_X=[]
Betas=np.linspace(1,10,3)
for beta in Betas : 
    X,U,N=[1,1],[[1,1,1],[1,1,1]],[[1/3,1/3,1/3],[1/3,1/3,1/3]]
    for i in range(2,NIT):
        U.append(U_h_t(U[-1],lambda_cst,R,X))
        N.append(n_h_t(U[-2],beta))##attention on met -2 car on vient de calculer Ut juste au dessus or c'est Ut-1 qui doit rentrer
        X.append(predict_x(R,X,N[-1]))
    Meta_X.append(X)

color=['blue','yellow','red','green','black','orange','pink','grey','brown','purple']
plt.figure()
for index,beta in enumerate(Betas): 
    plt.plot(Meta_X[index],color=color[index],label=r"$\beta$ ={}".format(beta))
plt.grid()
plt.title(r"$X_t$ evolution with beta")
plt.xlabel("Time")
plt.ylabel(r"$X_t$")
plt.legend()
plt.show()

'''
################################################################################################
#########################        Beta = 10 q2 #################################
################################################################################################
lambda_cst=0.1
R=0.01
Beta=10
G=2
X,U,N=[1,1],[[1,1,1],[1,1,1]],[[1/3,1/3,1/3],[1/3,1/3,1/3]]
for i in range(2,NIT):
    U.append(U_h_t(U[-1],lambda_cst,R,X))
    N.append(n_h_t(U[-2],Beta))##attention on met -2 car on vient de calculer Ut juste au dessus or c'est Ut-1 qui doit rentrer
    X.append(predict_x(R,X,N[-1]))
'''
plt.figure()
plt.plot(X)
plt.grid()
plt.title(r"$X_t$ evolution with beta ={}".format(Beta))
plt.xlabel("Time")
plt.ylabel(r"$X_t$")
plt.legend()
plt.show()

X_t=X[1:]
X_t_1=X[:-1]
plt.scatter(X_t_1,X_t)
plt.scatter(X_t[-1],X_t[-1],color='red')
plt.xlabel("X_t-1")
plt.ylabel("X_t")
plt.title(r"$X_t$ evolution with $X_{t-1}$ with beta = 10")
plt.grid()
plt.show()
### on voit qu'ils apprennent trop vite 
'''
################################################################################################
#########################        Beta = 100 q3 #################################
################################################################################################

Beta=100
X,U,N=[1,1],[[1,1,1],[1,1,1]],[[1/3,1/3,1/3],[1/3,1/3,1/3]]
for i in range(2,NIT):
    U.append(U_h_t(U[-1],lambda_cst,R,X))
    N.append(n_h_t(U[-2],Beta))##attention on met -2 car on vient de calculer Ut juste au dessus or c'est Ut-1 qui doit rentrer
    X.append(predict_x(R,X,N[-1]))
'''
plt.figure()
plt.plot(X)
plt.grid()
plt.title(r"$X_t$ evolution with beta={}".format(Beta))
plt.xlabel("Time")
plt.ylabel(r"$X_t$")
plt.legend()
plt.show()

X_t=X[1:]
X_t_1=X[:-1]
plt.scatter(X_t_1,X_t)
plt.scatter(X_t[-1],X_t[-1],color='red')
plt.xlabel("X_t-1")
plt.ylabel("X_t")
plt.title(r"$X_t$ evolution with $X_{t-1}$ and beta = 100")
plt.grid()
plt.show()
'''

###################################### 2 Traditionnal strategies ################
def n_h_t(U_t,beta): 
    """ 
    Entries : U_t la mesure de la performance des h modèles au temps t et beta la vitesse d'apprentissage des agents
    ================================================
    Aim : Calculer la liste des probas de suivre un modèle au temps t+1
    ================================================
    Outputs : liste des proba de suivre un modèle pour tout h au temps t+1
    """

    U_t=np.array(U_t)
    N_h_t=[np.exp(beta*U_t[i])/np.sum(np.exp(beta*U_t)) for i in range(len(U_t))]
    return N_h_t


def E_part_2(x,strat_function):
    return strat_function(x) ## calcul E pour une stratégie donnée


def U_h_t(U_h_t_1,lamb,r,p_t):
    """ 
    Entries : 
    ================================================
    Aim : 
    ================================================
    Outputs : 
    """
    U_h=[]
    for i in range(len(U_h_t_1)):## selon les h 
        first_part=U_h_t_1[i]*(1-lamb)
        second_part=lamb*(p_t[-1]-(1-r)*p_t[-2])*E_part_2(p_t[-1]-(1+r)*p_t[-2],strat[i])
        U_h.append(first_part+second_part)
    return U_h


def predict_x(r,x,n):
    somme=0
    for i in range(len(n)): 
        somme+=n[i]*E_part_2(x[-1],strat[i])
    X=(1/(r+1))*somme
    return(X)

def f_0(x):
    return 0 
def f_1(x):
    return 0.9*x+0.2
def f_2(x) : 
    return 0.9*x-0.2
def f_3(x):
    return (1+R)*x

strat=[f_0,f_1,f_2,f_3]## l'ensemble des fonctions stratégies 

lambda_cst=0.1
R=0.01
Beta=10
X,U,N=[1,1],[[1,1,1],[1,1,1]],[[1/3,1/3,1/3],[1/3,1/3,1/3]]
for i in range(2,NIT):
    U.append(U_h_t(U[-1],lambda_cst,R,X))
    N.append(n_h_t(U[-2],Beta))##attention on met -2 car on vient de calculer Ut juste au dessus or c'est Ut-1 qui doit rentrer
    X.append(predict_x(R,X,N[-1]))
'''
plt.figure()
plt.plot(X)
plt.grid()
plt.title(r"$X_t$ evolution with beta={}".format(Beta))
plt.xlabel("Time")
plt.ylabel(r"$X_t$")
plt.legend()
plt.show()

plt.figure()
X_t=X[1:]
X_t_1=X[:-1]
plt.scatter(X_t_1,X_t)
plt.scatter(X_t[-1],X_t[-1],color='red')
plt.xlabel("X_t-1")
plt.ylabel("X_t")
plt.title(r"$X_t$ evolution with $X_{t-1}$ with beta = 1")
plt.grid()
plt.show()
'''
######################## PART 3 #####

########## Faut modifier 
def n_h_t(U_t,beta): 
    """ 
    fonction softmax
    Entries : U_t la mesure de la performance des h modèles au temps t et beta la vitesse d'apprentissage des agents
    ================================================
    Aim : Calculer la liste des probas de suivre un modèle au temps t+1
    ================================================
    Outputs : liste des proba de suivre un modèle pour tout h au temps t+1
    """
    U_t=np.array(U_t)
    N_h_t=[np.exp(beta*U_t[i])/np.sum(np.exp(beta*U_t)) for i in range(len(U_t))]
    return N_h_t


def E_part_3(x,x_e,strat_function):
    return strat_function(x,x_e) ## calcul E pour une stratégie donnée



def ADA (x,x_e) :
    """ 
    Entries : x est la liste des prix réels au temps t, p_t_e est le prix prédit par ADA au temps t 
    =============================================================================================
    Aim : Calculer la prédiction selon la méthode ADA
    =============================================================================================
    Output : on renvoit le prix du prédicteur ADA au temps t +1 
    """
    return 0.65*x[-2]+0.35*x_e

def WTR(x,x_e):
    """ 
    Entries : p_t_1 est le prix réel au temps t_1, p_t_2 est le prix réel temps t-2
    =============================================================================================
    Aim : Calculer la prédiction selon la méthode WTR
    =============================================================================================
    Output : on renvoit le prix du prédicteur WTR au temps t +1 
    """
    return x[-2] + 0.4*(x[-2] - x[-3])
def STR(x,x_e): 

    """ 
    Entries : p_t_1 est le prix réel au temps t-1, p_t_2 est le prix réel au temps t-2
    =============================================================================================
    Aim : Calculer la prédiction selon la méthode STR
    =============================================================================================
    Output : on renvoit le prix du prédicteur STR au temps t +1 
    """
    return x[-2] + 1.3*(x[-2] - x[-3])

def LAA(x,x_e):
    return 0.5 *(np.mean(x[-3:])+x[-2])+x[-2]-x[-3]
def AA(x,x_e):
    return 0.5*(prix_fondamental+x[-2])+x[-2] - x[-3]


strat=[ADA,WTR,STR,LAA,AA]## l'ensemble des fonctions stratégies 

def U_h_t(U_h_t_1,lamb,r,p_t,x_e):
    """ 
    Entries : 
    ================================================
    Aim : 
    ================================================
    Outputs : 
    """
    U_h=[]
    for i in range(len(U_h_t_1)):## selon les h 
        first_part=U_h_t_1[i]*(1-lamb)
        # second_part=lamb*(p_t[-1]-(1-r)*p_t[-2])*E_part_3(p_t[-1]-(1+r)*p_t[-2],x_e[i],strat[i])
        second_part=lamb*(p_t[-1]-(1-r)*p_t[-2])*E_part_3(p_t,x_e[i],strat[i])
        U_h.append(first_part+second_part)
    return U_h

def predict_x_3(r,x,n,p_e):
    somme=0
    list_prix=[]
    for i in range(len(n)): 
        list_prix.append(E_part_3(x,p_e[i],strat[i]))
        somme+=n[i]*E_part_3(x,p_e[i],strat[i])
    Prix_strat.append(list_prix)
    X=(1/(r+1))*somme
    return(X)

lambda_cst=0.1
R=0.01
Beta=150
X,U,N=[1,1,1],[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],[[1/5,1/5,1/5,1/5,1/5],[1/5,1/5,1/5,1/5,1/5],[1/5,1/5,1/5,1/5,1/5]]
Prix_strat=[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
for i in range(3,NIT):
    U.append(U_h_t(U[-1],lambda_cst,R,X,Prix_strat[-1]))
    N.append(n_h_t(U[-2],Beta))##attention on met -2 car on vient de calculer Ut juste au dessus or c'est Ut-1 qui doit rentrer
    X.append(predict_x_3(R,X,N[-1],Prix_strat[-1]))

plt.figure()
plt.plot(X)
plt.grid()
plt.title(r"$X_t$ evolution with beta={}".format(Beta))
plt.xlabel("Time")
plt.ylabel(r"$X_t$")
plt.legend()
plt.show()
plt.figure()

plt.plot(N)
plt.grid()
plt.title(r"$N_t$ evolution with beta={}".format(Beta))
plt.xlabel("Time")
plt.ylabel("t")
plt.legend()
plt.show()

plt.figure()
X_t=X[1:]
X_t_1=X[:-1]
plt.scatter(X_t_1,X_t)
plt.scatter(X_t_1[-1],X_t[-1],color='red')
plt.xlabel("X_t-1")
plt.ylabel("X_t")
plt.title(r"$X_t$ evolution with $X_{t-1}$ with beta = {}")
plt.grid()
plt.show()