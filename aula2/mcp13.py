""" Spherical wave evolution using the Galerkin-Collocation method 
    for the Minkowski spacetime.

    By W. Barreto 21.01.20 """

import numpy as np
from scipy.special import chebyt, eval_chebyt, eval_gegenbauer
import matplotlib.pyplot as plt
import matplotlib.animation as animation

""" Basis """

""" The way to evaluate derivatives of Chebyshev was suggested by R. Aranha """

def T(n,x):
    T=eval_chebyt(n,x)
    return T

def dT(n, x):
    dT = n * eval_gegenbauer(n - 1, 1, x)
    return dT

def ddT(n, x):
    ddT = 2 * n * eval_gegenbauer(n - 2, 2, x)
    return ddT

def TL(n,L0,r):
    x=(r-L0)/(r+L0)  
    TL=T(n,x)
    return TL

def dTL(n,L0,r):
    x=(r-L0)/(r+L0)
    dxdr=2.*L0/(r+L0)**2.
    dTL=dxdr*dT(n,x)
    return dTL

def ddTL(n,L0,r):
    x=(r-L0)/(r+L0)
    dxdr=2.*L0/(r+L0)**2.
    d2xdr2=-4.*L0/(r+L0)**3.
    ddTL=ddT(n,x)*dxdr**2.+d2xdr2*dT(n,x)
    return ddTL

def psi(n,L0,r):
    psi=0.5*(TL(n+1,L0,r)+TL(n,L0,r))
    return psi

def dpsi(n,L0,r):
    dpsi=0.5*(dTL(n+1,L0,r)+dTL(n,L0,r))
    return dpsi

def ddpsi(n,L0,r):
    ddpsi=0.5*(ddTL(n+1,L0,r)+ddTL(n,L0,r))
    return ddpsi

""" TDD KO

x = np.linspace(-1.0, 1.0, 200)
for i in range(0,8):
    plt.plot(x,ddT(i,x))
plt.grid(True)
plt.show()

        OK """ 

""" Dynamical System """
def dynsys(x):
    ddphi=0.5*np.dot(BDDPHI,x)
    dx=np.dot(BDPHInv,ddphi)
    return dx
""" Runge Kutta """
def rk4(x,dx):
      delta=hs
      hh=delta*.5
      h6=delta/6.
      xt=x+hh*dx
      dxt=dynsys(xt)
      xt=x+hh*dxt
      dxm=dynsys(xt)
      xt=x+delta*dxm
      dxm=dxt+dxm
      dxt=dynsys(xt) 
      xout=x+h6*(dx+dxt+2.*dxm)
      return xout

""" Main """

""" Parameters """
p= 128              # Truncation
L0=1.               # Mapping parameter

A0=1.e-1            # Amplitude
sigma=.1            # Width
r_0=.5              # Peak

rinf=1.e10          # Computational infinity

hs=0.5e-3           # Time step
timemax=2.e0        # Maximum time 

vp=1                # Visualization parameter: 0 fixed at r=0; 1 free at r=0

pi=np.pi

""" Grid """

x=np.zeros(p+1)
r=np.zeros(p+1)
xx=np.zeros(p+1)
for k in range(0,p+1):
      x[k]=np.cos(pi*np.float(k+1)/np.float(p+2))
      r[k]=L0*(1.+x[k])/(1.-x[k])
      xx[p-k]=r[k]/(1.+r[k])

""" Frame """

M_PHI=np.zeros((p+1,p+1))
BPHI=np.zeros((p+1,p+1))
BDPHI=np.zeros((p+1,p+1))
BDDPHI=np.zeros((p+1,p+1))
PHI0=np.zeros(p+1)

for n in range(0,p+1):
    for k in range(0,p+1):
        BPHI[k][n]=psi(n,L0,r[k])

for n in range(0,p+1):
    for k in range(0,p+1):
       BDPHI[k][n]=dpsi(n,L0,r[k]) 

""" Inversion saving"""
M_PHI=BDPHI
BDPHInv=np.linalg.inv(M_PHI)

for n in range(0,p+1):
    for k in range(0,p+1):
       BDDPHI[k][n]=ddpsi(n,L0,r[k])

""" Initial data """
for i in range(0,p+1):
    PHI0[i]=A0*np.exp(-(r[i]-r_0)**2./sigma**2.)

RHS=PHI0
LHS=BPHI
a=np.linalg.solve(LHS,RHS)

PHI_c=np.dot(BPHI,a)

""" Test DD for the initial data """
""" KO """
#plt.plot(np.log10(np.abs(a)))
#plt.show()
#plt.plot(x,PHI0,'o',lw=2)
#plt.plot(x,PHI_c,lw=2)
#plt.show()

""" OK """

""" Evolution """

itimemax=int(timemax/hs)
g=np.zeros((itimemax+1,p+1))

for itime in range (0,itimemax+1):
    time=np.float(itime)*hs
    for i in range(vp,p+1): 
        g[itime][i-vp]=PHI_c[p-i]/(1. + vp*(r[p-i]-1.))

    da=dynsys(a)
    a=rk4(a,da)
    PHI_c=np.dot(BPHI,a)

""" Visualization """

""" Standard Plotting """
for i in range(0,itimemax,100):
    plt.plot(xx[:],g[i][:])
plt.show()

""" Animation """
#x_axis = list(range(p))
max_idx = len(g)
plt.style.use('seaborn-pastel')
fig=plt.figure()
ax=plt.axes(xlim=(0,1),ylim=(-1.,1.))
plt.grid()
line, = ax.plot([],[], lw=3)

def init():
    line.set_data([],[])
    return line,

def animate(i):
   x_axis=xx[:]
   y_axis=g[i][:]
   line.set_data(x_axis,y_axis)
   return line,

anim = animation.FuncAnimation(fig,animate, init_func=init,frames=max_idx,interval=1,blit=True)

#anim.save('movie.mp4', writer='ffmpeg')

plt.show()
      
