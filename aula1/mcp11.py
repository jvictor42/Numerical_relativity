import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Parameters

h=1.e-3
umax=2.

A0=1.e-1
sigma=0.1
r0=.5

nx=101 
nxmax=101
iumax=int(umax/h)

for inx in range(100,nxmax,100):
    nx=inx+1
    dx=1./np.float(nx)

    x=np.zeros(nx)
    xh=np.zeros(nx)
    r=np.zeros(nx)
    g=np.zeros((iumax+1,nx))
    Phi=np.zeros((iumax+1,nx))
    time=np.zeros(iumax)
    p=np.zeros(iumax+1)
    E=np.zeros(iumax)
    Flux=np.zeros(iumax)

#    print(np.shape(x))
#    print(iumax)

    for i in range(nx-1):
        x[i] = np.float(i+1)*dx
        xh[i]= (np.float(i+1)-0.5)*dx
        r[i] = x[i]/(1.-x[i])

    x[nx-1]=1.
    xh[nx-1]=1.-0.5*dx
    r[nx-1]=r[nx-2]

    p[0]=0.
    massrad=0.
#f = open('phi.global','w')

    for iu in range(iumax):
        u=np.float(iu)*h
        time[iu]=u
        g[iu][:]=(A0*np.exp(-(r+u/2.-r0)**2./sigma**2.)-A0*np.exp(-(u/2.-r0)**2./sigma**2.))
        Phi[iu][:]=g[iu][:]/r
        News=A0*(u/2.-r0)*np.exp(-(u/2.-r0)**2./sigma**2.)/sigma**2.
        mass=0.
        news=0.
        for i in range(1,nx):
            phixh=(g[iu][i]-g[iu][i-1])/dx
            phih=0.5*(g[iu][i]+g[iu][i-1])
            bracket=phixh*(1.-xh[i])-phih/xh[i]
            mass=mass+2.*np.pi*bracket*bracket*dx
            news=news+0.5*bracket*dx/xh[i]
        p[iu]=-4.*np.pi*News**2.
        massrad=massrad+0.5*h*(p[iu]+p[iu-1])
        E[iu]=mass
        Flux[iu]=-massrad
#print(Flux[iumax-1]-E[0],dx)
    print(Flux[iumax-1]-(E[int(iumax/2)]+Flux[int(iumax/2)]),dx)
plt.plot(time,E)
plt.plot(time,Flux)
plt.plot(time,E+Flux)
plt.show()
#for i in range(iumax): 
#    plt.plot(g[i][:])
#plt.show()
# Plotting with animation
#x_axis = list(range(nx))
max_idx = len(g)
#print(np.shape(x_axis),max_idx)
plt.style.use('seaborn-pastel')
fig=plt.figure()
ax=plt.axes(xlim=(0,1),ylim=(-1.,1.))
plt.grid()
line, = ax.plot([],[], lw=3)

def init():
    line.set_data([],[])
    return line,

def animate(i):
    x_axis=x[:]
    y_axis=Phi[i][:]
    line.set_data(x_axis,y_axis)
    return line,

anim = animation.FuncAnimation(fig,animate, init_func=init,frames=max_idx,interval=1,blit=True)

#anim.save('movie.mp4', writer='ffmpeg')

plt.show()
plt.close()
plt.clf()


#    for i in range(nx+1):
#        f.write(str(g[i])+'\n')
#f.close()
