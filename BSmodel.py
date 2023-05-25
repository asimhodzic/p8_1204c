
import numpy as np
from scipy.stats import norm
from numba import jit
from scipy.interpolate import RegularGridInterpolator 


def core_algorithm(r=0.1,sigma=0.4,d=0,K=50,T=5/12,v_max=1600,m=1600, call=False):
    q = 2*r/sigma**2
    q_d = 2*(r-d)/sigma**2
    if call:
        G = lambda x,tau: \
        np.exp(tau/4*((q_d-1)**2+4*q))\
            *np.maximum(np.exp(x/2*(q_d+1)) - np.exp(x/2*(q_d-1)), 0)
    else:
         G = lambda x,tau: \
        np.exp(tau/4*((q_d-1)**2+4*q))\
            *np.maximum(np.exp(x/2*(q_d-1)) - np.exp(x/2*(q_d+1)), 0)

    def nu(x,tau,y):
            return K*np.exp(-1/2*(q_d-1)*x-(1/4*(q_d-1)**2+q)*tau)*y

    
    deltatau = 1/2*sigma**2*T/v_max
    x_max = m/2*np.sqrt(2*deltatau) + .1
    x_min = - x_max
    deltax = (x_max - x_min)/m
    x = np.array([x_min + i*deltax for i in range(m+1)])
    tau = np.array([i*deltatau for i in range(v_max+1)])
    g = G(x.reshape(-1,1),tau)
    #initialize the iteration vector w with
    w = g[1:-1,0]
    l = deltatau/deltax**2
    stab = 'stable'*(l<=0.5) + 'unstable'*(l>0.5)
    print(f'Lambda {l} ({stab})')
    b = np.zeros(m-1)
    sol = g
    #tau-loop
    for v in range(v_max):
        b[1:m-2] = np.array([\
            w[i] + l*(w[i+1] - 2*w[i]+w[i-1]) \
            for i in range(1,m-2)])
        b[0] = w[0] + l*(w[1] - 2*w[0]+g[0,v])
        b[-1] = w[-1] + l*(w[-2] - 2*w[-1]+g[-1,v])

        w = np.maximum(b,g[1:-1,v+1])
        #w = np.minimum(b,g[1:-1,v+1])
        sol[1:-1,v+1] = w

    V = nu(x.reshape(-1,1),tau,sol)
    #V = np.flip(V, axis=1)
    S = K*np.exp(x)
    #T = 2/sigma**2*tau
    
    return [S,V]



def BS(sigma = 0.4, knownpars=[0.1,0.01,100,100,1,False], exclude_S0=False, v_max=1600,m=1600, American=False, **kwargs):
    r,d,S0,K,T,call=knownpars
    sigma = float(sigma)
    if American:      
        S,V = core_algorithm(r=r,d=d, sigma=sigma, K=1, T=np.max(T), call=call, v_max=v_max,m=m)
        Tlong = np.linspace(0,1,v_max+1)
        interpolator = RegularGridInterpolator((S,Tlong), V)
        K = np.array([K]).flatten() #Ensure that K is an array
        T = np.array([T]).flatten() #Ensure that T is an array
        def f(S0):
            price = np.zeros((len(K),len(T)))
            for i,t in enumerate(T):
                price[:,i] = np.array([k*interpolator(np.array([S0/k,t]))[0] for k in K])
            return price
        if exclude_S0:
            return f
        price = f(S0)
    else: #European
        if exclude_S0:
            return lambda S0: BS(sigma, [r,d,S0,K,T,call], False, v_max,m, American)
        if not isinstance(K,(int,float)):
            K = np.array(K).reshape(-1,1)
        d1 = (np.log(S0/K) + (r - d + sigma**2/2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if call:
            price = np.exp(-d*T) * S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S0*np.exp(-d*T)*norm.cdf(-d1)
    return price


def BSObjFun(sigma,knownpars, MktPrice=None, American=False, **kwargs):
    ModelPrice = BS(sigma=sigma, knownpars=knownpars, American=American)
    Loss = np.mean((ModelPrice - MktPrice)**2)
    print("{:^9}{:^9}".format(round(Loss,4),round(float(sigma),4)))
    return Loss


