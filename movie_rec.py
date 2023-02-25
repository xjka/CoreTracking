import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d
import argparse
from scipy.optimize import minimize 
from matplotlib import cm
from cycler import cycler

def get_cycler(n):
    return cycler(color=cm.get_cmap('viridis')(np.linspace(0,0.9,n)))

def init(im):
    im.set_data( np.array([]),np.array([]) )
    im.set_3d_properties(np.array([]))
    return im,

def animate(fig,idx,im1,im2,pos,trpos,trmu,time):
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

def prepare_animation(bar_containers, vels, imms, axh, Mu):

    def animate(frame_number, vels, imms, axh, Mu):
        legnd = []
        max_count = -1e34
        min_max_count = 1e34
        for idx,vel in enumerate(vels[frame_number]):
            if len(vel)>100:
                n, _ = np.histogram(vel, bins=30, density=True)
                for count, rect in zip(n, bar_containers[idx].patches):
                    rect.set_height(count)
                    max_count = np.max((count, max_count)) 
                min_max_count = np.min((min_max_count, max_count))
                sigma2 = (vel**2).sum()/(3*len(vel))
                vl = np.linspace(0, np.max(vel)*1.3, 200)
                imms[idx].set_data(vl , 4*np.pi*vl**2/np.power(np.pi*2*sigma2, 3.0/2) * np.exp(-vl**2/2/sigma2))
                axh.set(title=r'$N(\sigma|v) = \frac{{4\pi v^2}}{{(\pi 2 \sigma)^{{3/2}} }} e^{{\frac{{-v^2}}{{2\sigma}} }}$')
                legnd.append(r'$\sigma_{{m_p={:.0f}}}={:.3f}$'.format(Mu[frame_number][idx],sigma2))
            else:
                for rect in bar_containers[idx].patches:
                    rect.set_height(0)
                imms[idx].set_data([0,0],[0,0])
        axh.legend(legnd)
        axh.set_ylim(top=min_max_count*2.3)
        if idx<len(bar_containers)-1:
            for i in range(idx, len(bar_containers)):
                for rect in bar_containers[i].patches:
                    rect.set_height(0)
                imms[i].set_data([0,0],[0,0])
        #return bar_container.patches
        return [container.patches for container in bar_containers]
    return lambda frame_number : animate(frame_number, vels, imms, axh, Mu)


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
    n_collisions = [] 
    time = []
    for i in range(1,args.numframes+1):
        h5file = h5py.File(args.indir+"test_grav_"+str(i)+".hdf5")
        pos.append(h5file["particles/pos"][:,:])
        vel.append(h5file["particles/vel"][:,:])
        trpos.append(h5file["tracers/pos"][:,:])
        trvel.append(h5file["tracers/vel"][:,:])
        trmu.append(h5file["tracers/mu"][:])
        n_collisions.append(len(pos[-1][0])-len(trpos[-1][0]))
        time.append(h5file["time"])

    pos = np.array(pos)
    vel = np.array(vel)
    time = np.array(time).squeeze()
    n_collisions = np.array(n_collisions)
    Gamma = np.polyfit(time[1:], n_collisions[1:], 1)[0]

    im1 = []# ax.scatter(pos[0,0,:], pos[0,1,:], pos[0,2,:], marker='*')#,s=8)
    im2 = ax.scatter(trpos[0][0], trpos[0][1], trpos[0][2], marker='8', color='r', alpha=0.3)#, s=8*np.ones_like(trpos[0][0]))
    
    scl = args.scl
    ax.set_xlim([pos.min()/scl, pos.max()/scl])
    ax.set_ylim([pos.min()/scl, pos.max()/scl])
    ax.set_zlim([pos.min()/scl, pos.max()/scl])
    ax.grid()
    
    E = []
    v = []
    Mu = []
    #Sig = []
    for i,tv in enumerate(trvel): #tv is 3 x N_particles
        v2 = (tv**2).sum(axis=0)
        E.append((0.5*trmu[i]*v2).sum())
        #Sig.append(v2.sum()/(3*len(v2)))
        v_ = []
        Mu_ = []
        #Sig_ = []
        for mu in np.unique(trmu[i]):
            mask = trmu[i] == mu
            tv_ = np.array(tv[:,mask])
            v2_ = (tv_**2).sum(axis=0)
            v_.append(np.sqrt(v2_))
            Mu_.append(mu)
            #Sig_.append(v2_.sum()/(3*len(v2_)))
        v.append(v_)
        Mu.append(Mu_)
        #Sig.append(Sig_)

    E = np.array(E)


    num_mu = np.max(Mu[-1])/Mu[0][0]
    bin_edges = np.append(np.arange(1,num_mu+1)-0.5, num_mu+0.5)
    trmu_hist = []
    for tm in trmu:
        trmu_hist.append(np.histogram(tm/Mu[0][0], bins=bin_edges)[0] * np.arange(1,num_mu+1)*Mu[0][0] /np.sum(tm) )
    trmu_hist = np.array(trmu_hist)
   
    sig_consistent = np.empty((len(time), int(num_mu)))
    for i, tv in enumerate(trvel):
        v2 = (tv**2).sum(axis=0)
        for j, mu in enumerate(np.arange(1, num_mu+1)*Mu[0][0]):
            mask = trmu[i] == mu
            if np.any(mask):
                v2_ = (np.array(tv[:,mask])**2).sum(axis=0)
                sig_consistent[i,j] = np.mean(v2) #v2_.sum()/(3*len(v2_))
            
    #dEdt = np.diff(E)/np.diff(np.ones(len(E)))
    fig_E, ax_E = plt.subplots(2,2,figsize=(10,10))
    ax_E[0,0].plot(time, E/E.max(),'.-')
    ax_E[0,0].grid()
    ax_E[0,0].set(title='KE(t), $t_{{cool}}$={:.2e}'.format(np.abs(np.polyfit(np.arange(len(E)-1),E[1:],1)[0])/np.mean(E)))
    ax_E[0,1].plot(time, sig_consistent[:,:8],'.-')
    #ax_E[1].plot(sigs, '.-')
    ax_E[0,1].grid()
    ax_E[0,1].set(title = r'$\sigma(t)$')
    #ax_E[0,1].legend((r'$\sigma^2$', r'$\sigma_x^2$',r'$\sigma_y^2$',r'$\sigma_z^2$'))
    
    ax_E[1,0].plot(time, n_collisions,'.-')
    ax_E[1,0].grid()
    ax_E[1,0].set(title='# coliisions vs t, $\Gamma={:.4f}$'.format(Gamma))
    ax_E[1,1].set_prop_cycle(get_cycler(trmu_hist.shape[0]))
    ax_E[1,1].plot(np.tile(np.expand_dims(np.arange(1,trmu_hist.shape[1]+1),axis=1),(1,trmu_hist.shape[0])), trmu_hist.T)
    ax_E[1,1].grid()
    ax_E[1,1].set(title=r'Mass distribution over time', xscale='log', ylim=[0., ax_E[1,1].get_ylim()[1]])

    ani = animation.FuncAnimation(fig, lambda idx : animate(fig,idx,im1,im2,pos,trpos,trmu,time), blit=False, interval=25, frames=args.numframes, repeat_delay=300)
    
    bx_list = []
    im_list = []
    cmap = cm.get_cmap('viridis')
    clr_list = cmap(np.linspace(0,0.9,8))
    for i in range(8): #get the full number of histogram plots you could need 
        _,_,box_container = axh.hist(v[0][0], density=True, bins=30, color=clr_list[i], alpha=0.65)
        bx_list.append(box_container)
        #res = minimize(lambda s2,ve : (ve**2).sum()/2/s2 + len(ve)*3/2*np.log(np.pi*2*s2), x0=4, args=(v[0]) )
        sigma2 = (v[0][0]**2).sum()/(3*len(v[i]))
        vl = np.linspace(0,np.max(v[0][0]),100)
        imm, = axh.plot(vl, 4*np.pi*vl**2/np.power(np.pi*2*sigma2, 3.0/2) * np.exp(-vl**2/2/sigma2), c=clr_list[i])
        im_list.append(imm)

    #axh.hist(v0, density=True, bins=30)
    #res = minimize(lambda s2,ve : (ve**2).sum()/2/s2 + len(ve)*3/2*np.log(np.pi*2*s2), x0=4, args=(v[0]) )
    #sigma2 = (v[0]**2).sum()/(3*len(v[0]))
    #vl = np.linspace(0,np.max(v[0]),100)
    #imm, = axh.plot(vl, 4*np.pi*vl**2/np.power(np.pi*2*sigma2, 3.0/2) * np.exp(-vl**2/2/sigma2), 'b-')
    
    axh.set_ylim(top=2)#axh.get_ylim()[1]*1.3)
    axh.grid()
    
    ani2 = animation.FuncAnimation(fig, prepare_animation(bx_list, v, im_list, axh, Mu), blit=False, interval=25, frames=args.numframes, repeat_delay=300) 

    if len(args.outfile)>0:
        writermp4 = animation.FFMpegWriter(bitrate=600, fps=10)
        ani.save(args.outfile+'.mp4', writer=writermp4)

    plt.show()
