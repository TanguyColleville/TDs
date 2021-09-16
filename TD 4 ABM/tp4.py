import numpy as np 
import random as rd 
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import powerlaw as pw


T=10000


def Franke_Westerhoff(phi,khi,mu,nu,alpha_0,alpha_x,alpha_d,sigma_f,sigma_c,NIT=10000,fundamental_price=0):
    P_t,R_t,S_t,X_t,Sig_t=[0,0],[0,0],[0,0],[0,0],[0,0]
    for i in range(2,NIT):
        fundamental_price=np.mean(P_t)
        fundamental_price=0
        prix_t=np.sum(R_t)
        P_t.append(prix_t)## dernier élement de la liste est prix t+1

        s_t=alpha_0+alpha_x*X_t[-1] + alpha_d*(fundamental_price-P_t[-2])**2
        S_t.append(s_t) ##dernier élément de la liste est s t+1
        x_t1=X_t[-1]+nu*((1-X_t[-1])*np.exp(S_t[-2])-(1+X_t[-1])*np.exp(-S_t[-2]))
        if x_t1>=0:
            X_t.append(min(1,x_t1))
        elif x_t1<0:
            X_t.append(max(-1,x_t1))

        ### dernier élément de la liste est x t+1

        Sig_t.append(0.5*(((1+X_t[-2])**2)*sigma_f**2+((1-X_t[-2])**2)*sigma_c**2))
        
        epsilon_t=np.random.normal(0,np.sqrt(Sig_t[-2]))
        
        R_t.append((mu/2)*((1+X_t[-2])*phi*(fundamental_price-P_t[-2])+(1-X_t[-2])*khi*(P_t[-2]-P_t[-3])+epsilon_t))
    return R_t,X_t

    

R_t,X_t=Franke_Westerhoff(0.18,2.35,0.01,2.57,-0.15,1.35,11.4,0.79,1.91,NIT=1000)

def FW(phi,khi,mu,nu,alpha_0,alpha_x,alpha_d,sigma_f,sigma_c,NIT=10000,fp=0):
    x_t=np.zeros(NIT)
    p_t=np.zeros(NIT)
    r_t=np.zeros(NIT)
    s_t=np.zeros(NIT)
    sigma=np.zeros(NIT)
    epsilon=np.zeros(NIT)
    #x_t[0],x_t[1]=0.1,0.2
    for i in range(2,NIT):
        sigma[i]=0.5*(((1+x_t[i-1])**2)*sigma_f**2+((1-x_t[i-1])**2)*sigma_c**2)
        epsilon[i]=np.random.normal(0,np.sqrt(sigma[i]))
        r_t[i]=(mu/2)*((1+x_t[i-1])*phi*(fp-p_t[i-1])+(1-x_t[i-1])*(khi*(p_t[i-1]-p_t[i-2]))+epsilon[i-1])
        p_t[i]=p_t[i-1]+r_t[i]
        x=x_t[i-1]+nu*((1-x_t[i-1])*np.exp(s_t[i-1])-(1+x_t[i-1])*np.exp(-s_t[i-1]))
        if x>=0:
            x_t[i]=min(1,x)
        elif x<0:
            x_t[i]=max(-1,x)
        s_t[i]=alpha_0+alpha_x*x_t[i]+alpha_d*(fp-p_t[i])**2
    return r_t,x_t



R_t,X_t=FW(0.18,2.35,0.01,2.57,-0.15,1.35,11.4,0.79,1.91)
# plt.figure()
# plt.plot(np.linspace(0,10000-1,10000),R_t,label=r"$R_t$",color="orange")
# plt.title(r"Evolution of $R_t$")
# plt.xlabel(r"Lags")
# plt.ylabel(r"$R_t$")
#plt.xscale("log")
#plt.yscale("log")
#plt.grid()
#plt.show()

plt.figure()

plt.plot(np.linspace(0,1000-1,1000),X_t,label=r"$X_t$",color="green")
plt.title(r"Evolution of $X_t$")
plt.xlabel(r"Lags")
plt.ylabel(r"$R_t$")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.show()

def plot_PL(xmin,xmax,ymax,alpha):
    ## plot une power law distribution
    ## y max est le facteur juste pour ajuster ta courbe et la rapporhcer de ta distribtion
    dx = np.linspace(xmin,xmax,5) #also 2 are enough 
    
    y = dx**(-alpha)
    
    return dx,ymax*y/y[0]

ft=pw.Fit(np.abs(R_t))
alpha=ft.alpha
X_min=ft.xmin
X_max=np.max(R_t)

dx,dy=plot_PL(X_min,X_max,1e-1,alpha-1)
plt.plot(dx,dy,c="red",alpha=0.7,label=r"$\alpha$")

bn = sorted(set(np.abs(R_t))) 

plt.hist(np.abs(R_t),bins=bn,density=True,cumulative=-1,histtype='step')


plt.title("Distribution")
plt.xlabel("")
plt.ylabel(r"$P(|r|>R$")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.xlim(xmin=1e-3)
plt.grid()
plt.show()

list_phi=np.linspace(0.1,2,20)
alpha_phi=[]
for index,phi in enumerate(list_phi):
    R_t=Franke_Westerhoff(phi,2.35,0.01,2.57,-0.15,1.35,11.4,0.79,1.91,NIT=1000)
    ft=pw.Fit(np.abs(R_t))
    alpha_phi.append(ft.alpha)

plt.title("Distribution")
plt.xlabel(r"$\phi$")
plt.ylabel(r"$\alpha$")
plt.plot(list_phi,alpha_phi)
plt.legend()
plt.grid()
plt.show()
    

list_khi=np.linspace(0.1,2,20)
alpha_khi=[]
for index,khi in enumerate(list_khi):
    R_t=Franke_Westerhoff(0.18,khi,0.01,2.57,-0.15,1.35,11.4,0.79,1.91,NIT=1000)
    ft=pw.Fit(np.abs(R_t))
    alpha_phi.append(ft.alpha)

plt.title("Distribution")
plt.xlabel(r"$\khi$")
plt.ylabel(r"$\alpha$")
plt.plot(list_khi,alpha_khi)
plt.legend()
plt.grid()
plt.show()
    
list_alpha0=np.linspace(-2,2,20)
alpha_0=[]
for index,alpha0 in enumerate(list_alpha0):
    R_t=Franke_Westerhoff(0.18,2.35,0.01,2.57,alpha0,1.35,11.4,0.79,1.91,NIT=1000)
    ft=pw.Fit(np.abs(R_t))
    alpha_0.append(ft.alpha)

plt.title("Distribution")
plt.xlabel(r"$\alpha0$")
plt.ylabel(r"$\alpha$")
plt.plot(list_alpha0,alpha_0)
plt.legend()
plt.grid()
plt.show()
