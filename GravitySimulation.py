#####################################################################################################|Side comments go here.
# 3D Gravity Particle Simulation                                                                    
# Ben Bartlett                                                                                      
# benjamincbartlett AT gmail DOT com                                                                
# Losely based on my previous "Animator5D" code


import os, shutil, subprocess, signal                                                               #|System stuff
import numpy as np                                                                                  #|The package takes structured numpy arrays as arguments, but it shouldn't be terrible to modify it to take a list instead.
import matplotlib.pyplot as plt                                                                     #|Matplotlib stuff
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D, art3d 
from scipy.spatial.distance import cdist
from FastProgressBar import progressbar

def powerSet(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        powerSet(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out.T

def randomParticles(N, xyzrange = [[-1,1],[-1,1],[-1,1]]):
    '''Initializes N particles randomly over a given xyz range'''
    xrn, yrn, zrn = xyzrange
    xpos = np.random.uniform(xrn[0],xrn[1],N)
    ypos = np.random.uniform(yrn[0],yrn[1],N)
    zpos = np.random.uniform(zrn[0],zrn[1],N)
    return np.vstack((xpos,ypos,zpos)).T

def uniformParticles(sideLength, variance, xyzrange = [[-1,1],[-1,1],[-1,1]]):
    '''Initializes N particles randomly over a given xyz range'''
    xrn, yrn, zrn = xyzrange
    s = np.arange(sideLength, dtype=float)
    positions = 2*(powerSet((s, s, s)) - sideLength/2)/sideLength + .001 #|.001 is to avoid division by zero errors
    noise = np.random.uniform(0,variance,(3,sideLength**3))
    positions += noise
    return positions.T

def zeroVelocity(N):
    return np.zeros((N,3))

def forces(pos, mass=False):
    '''Returns a matrix giving the force on each particle'''
    posmatrix = cdist(pos, pos, 'euclidean') 
    massmatrix = 1                                                                                  #|Turn this off for uniform particles for faster processing
    np.fill_diagonal(posmatrix, -1)                                                                 #|Avoids division by zero errors
    forceMagnitudeMatrix = -G * massmatrix / np.square(posmatrix)
    np.fill_diagonal(forceMagnitudeMatrix,0)                                                        #|No force to yourself obviously
    normalpos = pos/np.linalg.norm(pos, axis=1)[:,None]                                             #|Normalized position matrix for computing xyz portions of the force matrix 
    forceMatrix = np.dot(forceMagnitudeMatrix, normalpos)                                           #|xyz force felt by each particle
    return forceMatrix

def update(pos, vel, tstep):
    masses = 1
    acc = forces(pos)
    vel += acc * tstep
    pos += vel * tstep
    magpos = np.linalg.norm(pos, axis=1)
    pos = np.compress(magpos < ignoreDist, pos, axis=0)
    vel = np.compress(magpos < ignoreDist, vel, axis=0)
    return pos, vel


def simulate(N, duration, tstep=1, renderEveryNthFrame=1, path="Simulation", viewradius=1,\
             xyzrange=[[-1,1],[-1,1],[-1,1]], projections=False, transparency=True, delete=False):
    # Keyboard interrupt handling
    def signal_handler(signal, frame):
        import sys
        print '\n(!) Rendering aborted: KeyboardInterrupt.'
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    xyzrange = np.multiply(viewradius, xyzrange)

    # Set up and clear path
    if os.path.exists(path):                                                                        #|Remove previous crap if anything is there
        shutil.rmtree(path)
        os.makedirs(path+"/frames")
    else:
        os.makedirs(path+"/frames")

    # Set some stuff up
    t = 0
    #pos = randomParticles(N)
    sideLength = np.floor(N**(1.0/3))
    pos = uniformParticles(sideLength, 0.002)
    vel = zeroVelocity(sideLength**3)

    numSteps = (duration-t)/float(tstep)
    step = 0
    count = 1
    pbar = progressbar("Computing &count& steps:", numSteps+1)
    pbar.start()
    # Simulation loop
    while t <= duration:
        if  step%renderEveryNthFrame == 0:
            renderFrame(pos, t, count, path, xyzrange, projections, transparency)
            count += 1
        pos, vel = update(pos, vel, tstep)
        t += tstep
        step += 1
        pbar.update(step)
    pbar.finish()

    print "Combining frames; may take a minute..."
    args = (['convert', '-delay', '.1', '-loop', '0', path+"/frames/*.gif", path+"/animation.gif"]) #|This part requires ImageMagick to function. Change the arguments as you wish.
    subprocess.check_call(args)

    if delete:
        shutil.rmtree(path+"/frames")
        print "Successfully deleted frames." 


def renderFrame(pos, t, count, path, xyzrange = [[-1,1],[-1,1],[-1,1]], projections=False, transparency=False):
    marker = '.'
    color = '#000000'
    size = 20

    # Extract xyz
    x,y,z = pos.T
    xrn, yrn, zrn = xyzrange

    # Set up figure
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_xlim(xrn[0],xrn[1])                                                                      #|Set axes and then modify later
    ax.set_ylim(yrn[0],yrn[1])
    ax.set_zlim(zrn[0],zrn[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Draw
    txt  = ax.text2D(0.1, 0.9,'$t=%.3f$'%t, transform=ax.transAxes)
    mark = ax.scatter(x,y,z, s=size, c=color, marker=marker, lw=0, alpha=0.7)
    if projections: 
        cx   = np.ones_like(x) * ax.get_xlim3d()[0]                                                 #|Again, not a typo with mixing x and z
        cy   = np.ones_like(y) * ax.get_ylim3d()[1]
        cz   = np.ones_like(z) * ax.get_zlim3d()[0]
        ax.scatter(x, y, cz, c='#444444', marker=marker, lw=0, s=size, alpha=0.3)                   #|Plot the projections
        ax.scatter(x, cy, z, c='#444444', marker=marker, lw=0, s=size, alpha=0.3)
        ax.scatter(cx, y, z, c='#444444', marker=marker, lw=0, s=size, alpha=0.3)
    if transparency == False: mark.set_edgecolors = mark.set_facecolors = lambda *args:None         #|Super-hacky way to disable transparency in the 3D plot, makes it cleaner to read.

    plt.draw()
    plt.savefig(path + "/frames/"+str(count).zfill(3)+".gif")                                       #|Save the frame. zfill(3) supports up to 999 frames, change as you want.
    plt.close()



if __name__ == "__main__":
    # Define some constants
    N = 2000                                                                                         #|Number of particles
    N = np.floor(N**(1.0/3))**3
    G = .002/N                                                                                      #|Universal gravitational constant
    ignoreDist = 3                                                                                  #|Distance after which you can ignore the particle

    simulate(N, 300, tstep=.05, viewradius=2, renderEveryNthFrame=10)







