
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import  minimize, minimize_scalar
from BSmodel import BS
from tabulate import tabulate

def plot_prices(K,S0,T,P, zlim=False, title='plot', subtitle='', American=False, elev=20, azim=-75, y=1, type="price"):
    if American:
        American = "AM"
    else:
        American = "EU"
    plt.rcParams["text.usetex"] = True
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.view_init(elev=elev, azim=azim)
    X, Y = np.meshgrid(np.log(S0/K), T)
    Z = P.T
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    
    if zlim:
        ax.set_zlim(*zlim)
    ax.set_xlabel('Log-Moneyness')
    ax.set_ylabel(r'Resterende løbetid ($\tau$)')
    if type.lower() in ("p", "price"):
        zlabel = r'Optionspris $\left(V_P^{{{}}}\right)$'.format(American)
    elif type.lower() in ("eep","early exercise", "early exercise price"):
        zlabel = r'Tidlig udnyttelsespris'
    elif type.lower() in ("iv", "implied", "implied volatility"):
        zlabel = r'Implikeret volatilitet'
    ax.set_zlabel(zlabel)
    ax.set_title('{}\n {}'.format(title,subtitle), y = y)
    #ax.annotate(subtitle, xy=(0.5, -0.1), xycoords='axes fraction',
            #ha='center', fontsize=12)
    fig.colorbar(surf, shrink=0.5, aspect=5)

#HestonObjFun(param, knownpars, MktPrice=None, uniform=True, GP = None, American=False)
# BSObjFun(sigma,knownpars, MktPrice=None, American=False)    
#calibrate(bounds[i,j],start[i,j],objfun[i,j], knownpars, MktPrice, model=models[i,j], market=markets[i,j],options=options, uniform=False, American=False, GP=GP)
def calibrate(bounds,start, objfun, knownpars, MktPrice, model="BS", market="BS", options = {'maxiter': 10000}, uniform=True,American=False, GP = None):
    # Run the Nelder-Mead algorithm with parameter bounds
    startnames, startvalues = start.keys(), np.array(list(start.values()))
    print(f"Market: {market} \nModel: {model}")
    print('-'*56,'\n')
    print(("{:^9}"*(len(start)+1)).format("MSE",*startnames))
    print('-'*56,'\n')
    result = minimize(lambda p: objfun(p,knownpars,MktPrice = MktPrice[market], American = American, uniform=uniform, GP=GP), \
                    startvalues, method='Nelder-Mead', bounds = bounds, options=options)
    param = result.x
    feval = result.fun
    print('-'*56,'\n')
    return param, feval

def parametertable(params, mse=False): 
    if mse:
        headers = ["", "BS", "Heston"]
        table_data = [["BS"] + [params[0][0]] + [params[0][1]],
                ["Heston"] + [params[1][0]] + [params[1][1]],
                ["Hestonfixed"] + [params[2][0]] + [params[2][1]]]
    else:
        headers = ["", "sigma", "kappa", "theta", "sigmav", "rho", "v0"]
        table_data = [["BS"] + list(params[0][0]) + list(params[0][1]),
                ["Heston"] + list(params[1][0]) + list(params[1][1]),
                ["Hestonfixed"] + list(params[2][0]) + list(params[2][1])]

    return tabulate(table_data, headers=headers)


def implied_vol(price, knownpars, eps = 1e-3, exclude_S0=False, v_max=1600,m=1600, American=False):
    r,d,S0,K,T,call = knownpars
    K = np.array([K]).flatten()
    T = np.array([T]).flatten()
    IV = np.zeros(price.shape)
    for j,t in enumerate(T):
        for i,k in enumerate(K):
            f = lambda sigma: abs(price[i,j] - BS(sigma, [r,d,S0,k,t,call], exclude_S0=exclude_S0, v_max=v_max,m=m, American=American))
            vol = minimize_scalar(f, method='bounded', bounds=(0, 2))
            if vol.fun > eps:
                IV[i,j] = None
            else:
                IV[i,j] = vol.x
    return IV