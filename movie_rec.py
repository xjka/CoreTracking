import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d
import argparse
from scipy.optimize import minimize 

def init(im):
    im.set_data( np.array([]),np.array([]) )
    im.set_3d_properties(np.array([]))
    return im,

def animate(fig,idx,im1,im2,pos,trpos,trmu):
    #im1.set_data(pos[idx,0,:], pos[idx,1,:])
    #im1.set_3d_properties(pos[idx,2,:])
    #im2.set_data(trpos[idx][0], trpos[idx][1])
    #im2.set_3d_properties(trpos[idx][2])
    #im2.set_markersize(trmu[idx])

    #im1._offsets3d = (pos[idx,0,:],pos[idx,1,:], pos[idx,2,:])
    #im1.set_offsets(pos[idx,0,:], pos[idx,1,:], pos[idx,2,:])
    im2._offsets3d = (trpos[idx][0], trpos[idx][1], trpos[idx][2])
    im2.set_sizes(trmu[idx]) 

    plt.gca().relim()
    plt.gca().autoscale_view()
    fig.suptitle('snapshot:{:d}, #collisions:{:d}'.format(idx,len(pos[idx,0,:])-len(trpos[idx][0])))
    #return (im1,im2) 

def prepare_animation(bar_container, vel, imm, axh):

    def animate(frame_number, vel, imm, axh):
        # simulate new data coming in
        n, _ = np.histogram(vel[frame_number], bins=30, density=True)
        for count, rect in zip(n, bar_container.patches):
            rect.set_height(count)
        sigma2 = (vel[frame_number]**2).sum()/(3*len(v[frame_number]))
        vl = np.linspace(0,np.max(vel[frame_number]),100)
        imm.set_data(vl , 4*np.pi*vl**2/np.power(np.pi*2*sigma2, 3.0/2) * np.exp(-vl**2/2/sigma2))
        axh.set(title=r'$N(kT/m_p|v) = \frac{{4\pi v^2}}{{(\pi 2 k T/m_p)^{{3/2}} }} e^{{\frac{{-v^2}}{{2kT/m_p}} }},\;\; kT/m_p={:.3f}$'.format(sigma2))
        return bar_container.patches
    return lambda frame_number : animate(frame_number, vel, imm, axh)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="make reconstruciton movie from directory of snapshots")
    parser.add_argument('-n', type=int, nargs='?', dest='numframes',default=600, help='number of snapshots')
    parser.add_argument('-o', type=str, nargs='?', dest='outfile', default='',help='output file to save video')
    parser.add_argument('-s', type=float, nargs='?', dest='scl', default=1e0, help='scale to scale down the axes')
    parser.add_argument('-i', type=str, nargs='?', dest='indir', default='output/', help='input file directory')
    args = parser.parse_args()

    plt.rcParams.update({"font.size":14})
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(121,projection='3d')
    axh = fig.add_subplot(122)

    pos = []
    vel = []
    trpos = []
    trvel = []
    trmu = []
    
    for i in range(1,args.numframes+1):
        h5file = h5py.File(args.indir+"test_grav_"+str(i)+".hdf5")
        pos.append(h5file["particles/pos"][:,:])
        vel.append(h5file["particles/vel"][:,:])
        trpos.append(h5file["tracers/pos"][:,:])
        trvel.append(h5file["tracers/vel"][:,:])
        trmu.append(h5file["tracers/mu"][:])

    
    pos = np.array(pos)
    vel = np.array(vel)

    im1 = []# ax.scatter(pos[0,0,:], pos[0,1,:], pos[0,2,:], marker='*')#,s=8)
    im2 = ax.scatter(trpos[0][0], trpos[0][1], trpos[0][2], marker='8', color='r', alpha=0.3)#, s=8*np.ones_like(trpos[0][0]))
    
    scl = args.scl
    ax.set_xlim([pos.min()/scl, pos.max()/scl])
    ax.set_ylim([pos.min()/scl, pos.max()/scl])
    ax.set_zlim([pos.min()/scl, pos.max()/scl])
    ax.grid()
    
    E = []
    sigs = []
    Sig = []
    v = []
    for i,tv in enumerate(trvel):
        tv = np.array(tv)
        v2 = (tv**2).sum(axis=0)
        E.append((0.5*trmu[i]*v2).sum() )
        sigs.append(tv.var(axis=1))
        Sig.append(v2.sum()/(3*len(v2)))
        v.append(np.sqrt(v2))
    
    E = np.array(E)
    sigs=np.array(sigs) 

    #dEdt = np.diff(E)/np.diff(np.ones(len(E)))
    fig_E, ax_E = plt.subplots(2,1,figsize=(10,10))
    ax_E[0].plot(E / E.max(),'.-')
    ax_E[0].grid()
    ax_E[0].set(title='E(t)')
    ax_E[1].plot(Sig,'.-')
    #ax_E[1].plot(sigs, '.-')
    ax_E[1].grid()
    ax_E[1].set(title = r'$\sigma(t)$')
    ax_E[1].legend((r'$\sigma^2$', r'$\sigma_x^2$',r'$\sigma_y^2$',r'$\sigma_z^2$'))
    ani = animation.FuncAnimation(fig, lambda idx : animate(fig,idx,im1,im2,pos,trpos,trmu), blit=False, interval=25, frames=args.numframes, repeat_delay=300)
    
    #figh, axh = plt.subplots(1,figsize=(10,10))
    _,_,box_container = axh.hist(v[0], density=True, bins=30)
    #res = minimize(lambda s2,ve : (ve**2).sum()/2/s2 + len(ve)*3/2*np.log(np.pi*2*s2), x0=4, args=(v[0]) )
    sigma2 = (v[0]**2).sum()/(3*len(v[0]))
    vl = np.linspace(0,np.max(v[0]),100)
    imm, = axh.plot(vl, 4*np.pi*vl**2/np.power(np.pi*2*sigma2, 3.0/2) * np.exp(-vl**2/2/sigma2), 'b-')
    axh.set_ylim(top=axh.get_ylim()[1]*1.3)
    axh.grid()
    ani2 = animation.FuncAnimation(fig, prepare_animation(box_container, v, imm, axh), blit=False, interval=25, frames=args.numframes, repeat_delay=300) 


    if len(args.outfile)>0:
        writermp4 = animation.FFMpegWriter(bitrate=600, fps=10)
        ani.save(args.outfile+'.mp4', writer=writermp4)

    plt.show()
