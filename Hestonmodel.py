import numpy as np
from scipy.stats import norm
from numba import jit
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator


def interpolate(S,V,U,S0,V0):
    return (RegularGridInterpolator((S,V), U)(np.array([[S0,V0]])))[0]


def HestonObjFun(param, knownpars, MktPrice=None, uniform=True, GP = None, American=False):
    ModelPrice = Hestonprice(param, knownpars, GP=GP, uniform=uniform, American = American)
    Loss = np.mean((ModelPrice - MktPrice)**2)
    print("{:^9}{:^9}{:^9}{:^9}{:^9}{:^9}".format(round(Loss,4),*np.round(param,decimals=4)))

    return Loss


@jit(nopython=True)
def get_U(U,S,V, NS, NV,A,B,C,D,E,r,dt,K,American,call):
        u = U.copy()
        derS, derV, derSS, derVV, derSV = np.zeros((5,*u.shape))
        for s in range(1,NS-1):
            derS[s,:] = (u[s+1,:] - u[s-1,:]) / (S[s+1]-S[s-1])
            derSS[s,:] = ((u[s+1,:] - u[s,:])   / (S[s+1]-S[s]) - (u[s,:] - u[s-1,:])/(S[s]-S[s-1])) / (S[s+1]-S[s])
        for v in range(1,NV-1):
            derV[:,v]  =  (u[:,v+1] - u[:,v-1]) / (V[v+1]-V[v-1])
            derVV[:,v] = ((u[:,v+1] - u[:,v])   / (V[v+1]-V[v]) - (u[:,v] - u[:,v-1])/(V[v]-V[v-1])) / (V[v+1]-V[v])
            derSV[:,v] = (derS[:,v+1] - derS[:,v-1])/ (V[v+1]-V[v-1])
        
        L = A*derSS + B*derSV + C*derVV - r*u + D*derS + E*derV
        U[1:-1,1:-1] = (L*dt + u)[1:-1,1:-1]    
        if American:
            if call:
                for v in range(NV):
                    U[:,v] = np.maximum(S - K, U[:,v])
                    #U[:,v] = payoff(S, U[:,v]) 
            else:
                for v in range(NV):
                    U[:,v] = np.maximum(K - S, U[:,v])
        return U

@jit(nopython=True)
def boundary(U,Smax,K,S,V,NS,r,d,kappa,theta,dt,call):
    U[0,:] = 0
    if call:
        U[-1,:] = np.maximum(Smax - K, 0)
        U[:,-1] = np.maximum(S - K, 0)
    else:
        U[-1,:] = np.maximum(K - Smax, 0)
        U[:,-1] = np.maximum(K - S, 0)

    # Update the temporary grid u(s,t) with the boundary conditions
    u = U.copy()

    # Boundary condition for Vmin.
    # Previous time step values are in the temporary grid u[s,t)
    for s in range(1,NS-1):
        derV = (u[s,1]   - u[s,0])   / (V[1]-V[0])         # Forward difference
        derS = (u[s+1,0] - u[s-1,0]) / (S[s+1]-S[s-1])     # Central difference
        LHS = - r*u[s,0] + (r-d)*S[s]*derS + kappa*theta*derV
        U[s,0] = LHS*dt + u[s,0]
    return U


def HestonExplicitPDE(params,knownpars,S,V,American=False):
    r,d,S0,K,T,call=knownpars
    kappa, theta, sigma, rho,V0 = params
    NS = len(S); NV = len(V); NT = len(T)
    Smax = S[-1]
    dt = (T[-1]-T[0])/(NT-1)

    if call:
        U = np.tile(np.maximum(S - K, 0)[:, np.newaxis], NV)
    else:
        U = np.tile(np.maximum(K - S, 0)[:, np.newaxis], NV)

    ST = S.reshape(-1,1)
    A = 0.5*V*ST**2
    B = rho*sigma*V*ST
    C = 0.5*sigma**2*V
    D = (r-d)*ST
    E = kappa*(theta-V)

    price = np.zeros(NT)
    price[0] = interpolate(S,V,U,S0,V0)

    for t in range(NT-1):
        U = boundary(U,Smax,K,S,V,NS,r,d,kappa,theta,dt,call)
        U = get_U(U,S,V, NS, NV,A,B,C,D,E,r,dt,K,American,call)
        price[t+1] = interpolate(S,V,U,S0,V0)
    return price

def make_grid(K, Mat, GP=None, uniform=True):
    if not isinstance(K, (int,float)):
        K = np.max(K)
        Mat = np.max(Mat)

    if GP is None:
        GP = [(0, 2.5*K), (0, 0.5), (0, Mat), 99, 99, 3000]
    if callable(GP):
        GP = GP(K)
    if not isinstance(GP,(np.ndarray,list)):
        raise ValueError("Grid parameters must be a list or function")
    # Minimum and maximum values for the Stock Price, Volatility, and Maturity
    if uniform:
        S, V, T = [np.linspace(*GP[i],GP[i+3] + 1) for i in range(3)]
    else:
        c = K/5
        dz = 1/GP[3]*(np.arcsinh((GP[0][1]-K)/c) - np.arcsinh(-K/c))
        S = np.zeros(GP[3]+1)
        for i in range(GP[3]+1):
            S[i] = K + c*np.sinh(np.arcsinh(-K/c) + i*dz)
            #S[i] = K + K/5*sinh(asinh(-5) + i*(1/GP[3]*(asinh(5*(GP[0][1]-K)/K) - asinh(-5))))
        d = GP[1][1]/500
        dn = np.arcsinh(GP[1][1]/d)/GP[4]
        V = np.zeros(GP[4]+1)
        for j in range(GP[4]+1):
            V[j] = d*np.sinh(j*dn)
        #V = [GP[1][1]/500 * sinh(j*asinh(500)/GP[4]) for j in range(GP[4]+1)]
        T = np.linspace(*GP[2],GP[5] + 1)
    return S,V,T


def Hestonprice(Hestonparams, knownpars, GP=None, uniform=True, American = False):
    r,d,S0,K,Mat,call = knownpars
    if isinstance(K,(float,int)):
        K = np.array([K])
    if isinstance(Mat,(float,int)):
        Mat = np.array([T])
    
    ModelPrice = np.zeros((len(K),len(Mat)))
    for i,k in enumerate(K):
        S,V,T = make_grid(k, np.max(Mat), GP=GP, uniform=uniform)
        p = HestonExplicitPDE(Hestonparams,[r,d,S0,k,T,call], S, V, American=American)
        ModelPrice[i,:] = np.interp(Mat, T, p)
    return ModelPrice



