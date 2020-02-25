""" Solving the EKG 1D problem using the Galerkin-Collocation method 
    By W. Barreto 25.01.20 """


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

def psi_V(n,L0,r):
    psi_V=0.25*( ( 1. + 2.*np.float(n)) * psi(n+1,L0,r) / (3. + 2.*np.float(n)) + psi(n,L0,r) )
    return psi_V

def dpsi_V(n,L0,r):
    dpsi_V=0.25*( ( 1. + 2.*np.float(n)) * dpsi(n+1,L0,r) / (3. + 2.*np.float(n)) + dpsi(n,L0,r) )
    return dpsi_V

def psi_beta(n,L0,r):
    psi_beta=2.*( psi_V(n,L0,r) - (np.float(n)+1.)**3.*(5.+2.*np.float(n))*psi_V(n+1,L0,r)\
             /((2.+np.float(n))**3.*(3.+2.*np.float(n))))
    return psi_beta

def dpsi_beta(n,L0,r):
    dpsi_beta=2.*( dpsi_V(n,L0,r) - (np.float(n)+1.)**3.*(5.+2.*np.float(n))*dpsi_V(n+1,L0,r)\
             /((2.+np.float(n))**3.*(3.+2.*np.float(n))))
    return dpsi_beta

def field_phi(x):
    phi=np.dot(BPHI,x)
    dphi=np.dot(BDPHI,x)
    ddphi=np.dot(BDDPHI,x)
    return phi,dphi,ddphi

def field_phi_(x):
    phi_=np.dot(BPHI_,x)
    dphi_=np.dot(BDPHI_,x)
    return phi_,dphi_

def metric_beta(x):
    beta=np.dot(BBETA,x)
    return beta

def metric_beta_(x):
    beta_=np.dot(BBETA_,x)
    return beta_

def metric_V(x):
    V=r*(1.+np.dot(BV,x))
    dV=1.+np.dot(BV,x) + r*np.dot(BDV,x)
    return V,dV

def metric_V_(x):
    V_=r_*(1.+np.dot(BV_,x))
    dV_=1.+np.dot(BV_,x) + r_*np.dot(BDV_,x)
    return V_,dV_

""" Hypersurface equations """
def hypsur(x):
    phi_,dphi_ = field_phi_(x)
    RHS_ = 2.*pi*r_*(dphi_/r_-phi_/r_**2.)**2.
    y = np.dot(BDBETAinv,RHS_)
#    y = a=np.linalg.solve(BDBETA_,RHS_)
    beta_ = metric_beta_(y)
    RHS__ = np.exp(2.*beta_)*(1.-4.*pi*m0*m0*phi_*phi_) - 1.
    z = np.dot(LHSinv,RHS__)
    return y,z


""" Evolution equation (Dynamical System) """
def dynsys(x,y,z):
    phi,dphi,ddphi=field_phi(x)
    beta=metric_beta(y)
    V,dV=metric_V(z)
    RHS = 0.5*(dV*dphi+ddphi*V+V*phi/r**2.-V*dphi/r-phi*dV/r)/r-0.5*np.exp(2.*beta)*m0*m0*phi
#    for i in range(p+1):
#        print(i,V[i],dV[i],beta[i])
    dx=np.dot(BDPHInv,RHS)
    return dx

""" Runge Kutta """
def rk4(x,y,z,dx):
      delta=hs
      hh=delta*.5
      h6=delta/6.
      xt=x+hh*dx
      dxt=dynsys(xt,y,z)
      xt=x+hh*dxt
      dxm=dynsys(xt,y,z)
      xt=x+delta*dxm
      dxm=dxt+dxm
      dxt=dynsys(xt,y,z)
      xout=x+h6*(dx+dxt+2.*dxm)
      return xout

""" Main """

""" Parameters """
p= 100              # Truncation 1
p_= np.int(3*p/2)   # Truncation 2
L0=1.               # Mapping parameter

A0=1.4e-1            # Amplitude
sigma=.3            # Width
r_0=.7              # Peak

rinf=1.e10          # Computational infinity
m0=0.               # scalar field mass
hs=1.e-3            # Time step
timemax=2.e0     # Maximum time 

vp=1                # Visualization parameter: 0 fixed at r=0; 1 free at r=0

pi=np.pi

""" Grid """

x=np.zeros(p+1)
r=np.zeros(p+1)
xx=np.zeros(p+1)
x_=np.zeros(p_+1)
r_=np.zeros(p_+1)
xx_=np.zeros(p_+1)

for k in range(0,p+1):
    x[k]=np.cos(pi*np.float(k+1)/np.float(p+2))
    r[k]=L0*(1.+x[k])/(1.-x[k])
    xx[p-k]=r[k]/(1.+r[k])

for k in range(0,p_+1):
    x_[k]=np.cos(pi*np.float(k+1)/np.float(p_+2))
    r_[k]=L0*(1.+x_[k])/(1.-x_[k])
    xx_[p_-k]=r_[k]/(1.+r_[k])

""" Frame """

BPHI=np.zeros((p+1,p+1))
BDPHI=np.zeros((p+1,p+1))
BDDPHI=np.zeros((p+1,p+1))
BPHI_=np.zeros((p_+1,p+1))
BDPHI_=np.zeros((p_+1,p+1))
BBETA=np.zeros((p+1,p_+1))
BDBETA=np.zeros((p+1,p_+1))
BBETA_=np.zeros((p_+1,p_+1))
BDBETA_=np.zeros((p_+1,p_+1))
BV=np.zeros((p+1,p_+1))
BDV=np.zeros((p+1,p_+1))
BV_=np.zeros((p_+1,p_+1))
BDV_=np.zeros((p_+1,p_+1))
LHS_=np.zeros((p_+1,p_+1))
PHI0=np.zeros(p+1)

""" Base for Phi and derivatives"""

for n in range(0,p+1):
    for k in range(0,p+1):
        BPHI[k][n]=psi(n,L0,r[k])

for n in range(0,p+1):
    for k in range(0,p+1):
       BDPHI[k][n]=dpsi(n,L0,r[k])

""" Inversion saving """
M_PHI=np.copy(BDPHI)
BDPHInv=np.linalg.inv(M_PHI)

for n in range(0,p+1):
    for k in range(0,p+1):
       BDDPHI[k][n]=ddpsi(n,L0,r[k])

""" Base for Phi_ and derivatives"""

for n in range(0,p+1):
    for k in range(0,p_+1):
        BPHI_[k][n]=psi(n,L0,r_[k])


for n in range(0,p+1):
    for k in range(0,p_+1):
        BDPHI_[k][n]=dpsi(n,L0,r_[k])

""" Base for beta and derivatives """

for n in range(0,p_+1):
    for k in range(0,p+1):
        BBETA[k][n]=psi_beta(n,L0,r[k])

for n in range(0,p_+1):
    for k in range(0,p+1):
        BDBETA[k][n]=dpsi_beta(n,L0,r[k])

""" Base for beta_ and derivatives """

for n in range(0,p_+1):
    for k in range(0,p_+1):
        BBETA_[k][n]=psi_beta(n,L0,r_[k])

for n in range(0,p_+1):
    for k in range(0,p_+1):
        BDBETA_[k][n]=dpsi_beta(n,L0,r_[k])

""" Inversion saving """

M_BETA=np.copy(BDBETA_)
BDBETAinv=np.linalg.inv(M_BETA)

""" Base for V and derivatives """

for n in range(0,p_+1):
    for k in range(0,p+1):
        BV[k][n]=psi_V(n,L0,r[k])

for n in range(0,p_+1):
    for k in range(0,p+1):
        BDV[k][n]=dpsi_V(n,L0,r[k])


""" Base for V_ and derivatives """

for n in range(0,p_+1):
    for k in range(0,p_+1):
        BV_[k][n]=psi_V(n,L0,r_[k])

for n in range(0,p_+1):
    for k in range(0,p_+1):
        BDV_[k][n]=dpsi_V(n,L0,r_[k])

""" Inversion saving """
for n in range(0,p_+1):
    for k in range(0,p_+1):
        LHS_[k][n] = BV_[k][n] + r_[k]*BDV_[k][n]
LHSinv=np.linalg.inv(LHS_)

print("+==================+") 
print("|        _         |") 
print("|       (_)        |") 
print("|   _ __ _  ___    |") 
print("|  | '__| |/ _ \\   |") 
print("|  | |  | | (_) |  |") 
print("|  |_|  |_|\\___/   |") 
print("|                  |") 
print("+==================+") 
print("riocode@UERJ")


""" Initial data """

#for i in range(0,p+1):
#    PHI0[i]=A0*r[i]*r[i]*r[i]*np.exp(-(r[i]-r_0)**2./sigma**2.) #Brady's initial data

PHI0=A0*r*r*r*np.exp(-(r-r_0)**2./sigma**2.) #Brady's initial data
#PHI0=A0*np.exp(-(r-r_0)**2./sigma**2.) 

RHS=PHI0

LHS=np.copy(BPHI)

""" Initial modes for the scalar field, beta and V """
a=np.linalg.solve(LHS,RHS) 
b,c = hypsur(a)

""" Test Driven Development (TDD) <<<Begin>>>"""
""" Initial modes for the scalar field and reconstruction """
PHI_c=np.dot(BPHI,a)
BETA_c=np.dot(BBETA_,b)
V_c=np.dot(BV_,c)
#plt.plot(np.log10(np.abs(a)))
#plt.ylabel('$\log|a(p_g)|$')
#plt.xlabel('p$_{g}$')
#plt.show()
#plt.plot(np.log10(np.abs(b)))
#plt.ylabel('$\log|b(p_\\beta)|$')
#plt.xlabel('p$_{\\beta}$')
#plt.show()
#plt.plot(np.log10(np.abs(c)))
#plt.ylabel('$\log|a(p_V)|$')
#plt.xlabel('p$_{V}$')
#plt.show()
#plt.plot(x,PHI0,'o',lw=2)
#plt.xlabel('x')
#plt.ylabel('g')
#plt.plot(x,PHI_c,lw=2)
#plt.show()
#plt.plot(xx_,BETA_c,lw=2)
#plt.show()
#plt.plot(xx_,V_c,lw=2)
#plt.show()

""" <<< TDD End>>> """

""" <<< Convergence Test (CT) BEGIN >>> """
""" <<< CT END >>> """

""" <<< TDD BEGIN >>> """
""" Initial modes for beta and V """
""" <<< TDD END >>> """

""" Evolution """

itimemax=int(timemax/hs)
g=np.zeros((itimemax+1,p+1))
#print(itimemax,np.shape(g))

for itime in range (0,itimemax+1):
    time=np.float(itime)*hs
    for i in range(vp,p+1):
        g[itime][i-vp]=PHI_c[p-i]/(1. + vp*(r[p-i]-1.))
    da=dynsys(a,b,c)
#    if time == 0.:
#       for i in range(p+1):
#           print(i,da[i])
    a=rk4(a,b,c,da)
    b,c = hypsur(a)
    PHI_c=np.dot(BPHI,a)
    beta = metric_beta(b)
    if np.exp(2.*beta[1]) >= rinf:
        print('Redshift =', np.exp(2.*beta[1]))
        print('Exceeds maximum specified: ',rinf)
        quit()

""" Visualization """
#plt.plot(xx[:],g[0][:])
#plt.plot(xx[:],g[1000][:])
for i in range(0,itimemax,100):
    plt.plot(xx[:],g[i][:])
plt.show()
""" Animation """
#x_axis = list(range(p))
max_idx = len(g)
plt.style.use('seaborn-pastel')
fig=plt.figure()
ax=plt.axes(xlim=(0,1),ylim=(-.5,.5))
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

