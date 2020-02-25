import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Parameters

#h=1.e-3
umax=2.

A0=1.e-1
sigma=0.1
r0=.5

nx=128
dx=1./np.float128(nx)
CFL=0.5
h=CFL*dx
iumax=int(umax/h)
dt=h

dis=.1

x=np.zeros(nx)
xh=np.zeros(nx)
r=np.zeros(nx)
g=np.zeros((iumax+1,nx))
gold=np.zeros(nx)
gnew=np.zeros(nx)
gdis=np.zeros(nx)

#print(np.shape(gold),np.shape(gnew))
for i in range(nx-1):
    x[i] = np.float128(i)*dx
    xh[i]= (np.float128(i)-0.5)*dx
    r[i] = x[i]/(1.-x[i])

x[nx-1]=1.
xh[nx-1]=1.-0.5*dx
r[nx-1]=r[nx-2]

#f = open('phi.global','w')

gold=A0*np.exp(-(r-r0)**2./sigma**2.)

#print(np.shape(r))
for iu in range(iumax):
    u=np.float(iu)*dt
#    g[iu][:]=(A0*np.exp(-(r+u/2.-r0)**2./sigma**2.)-A0*np.exp(-(u/2.-r0)**2./sigma**2.))/r
#    print(u,gold[nx-1])
    for i in range(1,nx): 
        g[iu][i-1]=gold[i]/r[i]
#        phi[i]=gold[i]/r[i]
    
    """ Start (as in gstart for kink Pitt code)"""
    i=1
    dxdrim = (1. - x[i-1]) ** 2
    dxdri  = (1. - x[i])   ** 2
    delim  = 0.5 * dt * dxdrim
    deli   = 0.5 * dt * dxdri

    gp = gnew[i-1]

    gr = gold[i-1] + (gold[i]   - gold[i-1]) * delim * np.float128(nx)
    gs = gold[i]   + (gold[i+1] - gold[i])   * deli  * np.float128(nx)

    gnew[i] = gp + gs - gr
#    With the following start if the field is free it gets noise
#    gnew[i]=gold[i+1]

    for i in range(2,nx-1):
         #print(i)
         dxdrim = (1. - x[i-1]) ** 2
         dxdri  = (1. - x[i])   ** 2
         delim  = 0.5 * dt * dxdrim
         deli   = 0.5 * dt * dxdri
         gp = gnew[i-1]    - (gnew[i-1]    - gnew[i-2])    * 0.5 * delim * np.float128(nx)
         gr = gold[i-1] + (gold[i]   - gold[i-1]) * 0.5 * delim * np.float128(nx)
         gs = gold[i]   + (gold[i+1] - gold[i])   * 0.5 * deli  * np.float128(nx)
         gnew[i] = (- gnew[i-1] * (0.5 * deli * np.float128(nx)) \
                   + gp + gs - gr )                 \
                   / (1. - (0.5 * deli * np.float128(nx)))
    i=nx-1
    dxdrim = (1. - x[i-1]) ** 2
    delim  = 0.5 * dt * dxdrim

    gp = gnew[i-1]    - (gnew[i-1]  - gnew[i-2])    * 0.5 * delim * np.float128(nx)
    gr = gold[i-1] + (gold[i] - gold[i-1]) * 0.5 * delim * np.float128(nx)
    gs = gold[nx-1]

    gnew[i] = gp + gs - gr

    gold[:]=gnew[:]

    """ Oliger-Kreiss filtering as in Rezzolla et. al., gr-qc/0606104 """
#    fact=dis/16.
#    for i in range(3,nx-2):
#        gdis[i]=gold[i-2] -4.*gold[i-1] + 6.*gold[i] -4.*gold[i+1] + gold[i+2]
#    a=4
#    b=nx-3
#    c=256./(a-b)**8
#    for i in range(3,nx-2):
#        shape = c * (i-a)**4 * (i-b)**4
#        gold[i]=gold[i] -fact*gdis[i]*shape

for i in range(0,iumax,10): 
    plt.plot(x[:],g[i][:])
plt.show()
#plt.plot(g[:][nx])
#plt.grid()
#plt.show()
"""
<<< Comment here >>>
"""
# Plotting with animation

max_idx = len(g)
plt.style.use('seaborn-pastel')
fig=plt.figure()
ax=plt.axes(xlim=(0.,1.),ylim=(-1.,1.))
plt.grid()
line, = ax.plot([],[], lw=3)

def init():
    line.set_data([],[])
    return line,

def animate(i):
   x_axis=x[:]
   y_axis=g[i][:]
   line.set_data(x_axis,y_axis)
   return line,

anim = animation.FuncAnimation(fig,animate, init_func=init,frames=max_idx,interval=1,blit=True)

#anim.save('movie.mp4', writer='ffmpeg')

plt.show()
#plt.close()
#plt.clf()


#    for i in range(nx+1):
#        f.write(str(g[i])+'\n')
#f.close()
