import numpy as np 
import random as rd 
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.api as sm
import powerlaw as pw 
from statsmodels.distributions.empirical_distribution import ECDF


NIT=1000 ### nombre d'itérations 
sigma=4
alpha=2
epsilon=np.random.normal(0,sigma,size=NIT)
r0=1
color=['blue','yellow','red','green','black','orange','pink','grey','brown','purple']


def PawPat(r_0,alpha_chap_0):
    alpha_chap=[alpha_chap_0]
    r=[r_0,r_0+0.4]
    for i in range(1,NIT-1):
        alpha_chap.append(r[i]/r[i-1]+alpha_chap[i-1])
        r.append((alpha-alpha_chap[i])*r[i]+epsilon[i+1])
    return r,alpha_chap

# r_t=PawPat(r0,1.04)[0]
# plt.figure(figsize=(10,6))
# plt.plot(r_t,c="blue",alpha=0.7,label=f"return \u03B1 = {alpha} and \u03C3 = {sigma}")
# plt.title("Return evolution")
# plt.xlabel("Time")
# plt.ylabel("Return")
# plt.legend()
# plt.grid()
# plt.show()

# sigmas=np.linspace(0,3,7)
# for index,sig in enumerate(sigmas):
#     epsilon= epsilon=np.random.normal(0,sig,size=NIT)
#     plt.plot(PawPat(r0,alpha)[0],c=color[index],label=f"\u03B1 = {alpha} and \u03C3 = {round(sig,3)}",alpha=0.5)
# plt.title("Return evolution")
# plt.xlabel("Time")
# plt.ylabel("Return")
# # plt.yscale("log")
# plt.grid()
# plt.legend()
# plt.show()  

# alphas=np.linspace(1,5,7)
# for index,alpha in enumerate(alphas):
#     epsilon= epsilon=np.random.normal(0,sigma,size=NIT)
#     plt.plot(PawPat(r0,alpha)[0],c=color[index],label=f"\u03B1 = {round(alpha,3)} and \u03C3 = {sigma}",alpha=0.5)
# plt.title("Return evolution")
# plt.xlabel("Time")
# plt.ylabel("Return")
# # plt.yscale("log")
# plt.grid()
# plt.legend()
# plt.show()  

### o npeut voir des comportements extremes 
# plus on est proche de zero , plus on va avoir une explosio(n epislont/rt-1)

####  2 Optimal Learning

sigma=2
alpha=1.2
epsilon=np.random.normal(0,sigma,size=NIT)
R_abs=np.abs(PawPat(r0,1.2)[0])
X=np.linspace(0,np.max(R_abs))
ecdf=ECDF(R_abs)

Y=1-ecdf(X)

# unique_obs = np.unique(ecdf)
# X=ecdf.x
# Y = ecdf.y
# P_r=np.ones(len(Y))-Y

plt.figure(figsize=(10,6))
plt.plot(X,Y,c="blue",alpha=0.7,label=f"return \u03B1 = {alpha} and \u03C3 = {sigma}")
plt.title("Distribution")
plt.xlabel("")
plt.ylabel("Proba")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.grid()
plt.show()

mypl=pw.Fit(R_abs)
print(mypl.alpha)