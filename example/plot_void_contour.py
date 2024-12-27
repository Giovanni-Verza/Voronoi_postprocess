import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline 
import time
import vide.voidUtil as vu


##### to reactivate interctive matplotlib mode after having imported vide utils: 
#plt.rcParams.update({'backend':'QtAgg',
#                     'backend_fallback':True})
#%matplotlib qt



def void_border(dx,dy,nTheta_bins=101,sharpest_ang=80.,frac_min_dist=-1.):

    #############################################################################################################################
    #############################################################################################################################
    # dx,dy: tracers 2D coord wrt void center
    # nTheta_bins: number of theta bins in finding outer tracers
    # sharpest_ang: sharpest angle allowed, to avoid jagged contours. If sharpest_ang < 0. not considered.
    # frac_min_dist: merge vertex closer than the mean distance between consecoutive vertex. If frac_min_dist < 0. not considered.
    #############################################################################################################################
    #############################################################################################################################

    # circular coordinates
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan(dy/dx) #* 180. / np.pi
    theta[(dx < 0)] += np.pi
    theta[(theta < 0)] += 2.*np.pi

    # select the most distant tracer in each bin in theta
    theta_bins = np.linspace(0.,2*np.pi,nTheta_bins)
    ids = np.arange(numPart)
    ids_border = []
    for i in range(theta_bins.shape[0]-1):
        mask = (theta >= theta_bins[i]) & (theta < theta_bins[i+1])
        
        if np.sum(mask) > 0:
            ind_max = np.argmax(r[mask])
            ids_border.append(ids[mask][ind_max])

    # void border
    x_border = dx[ids_border] #r[ids_border]*np.cos(theta[ids_border])
    y_border = dy[ids_border] #r[ids_border]*np.sin(theta[ids_border])
    r_border = r[ids_border] #r[ids_border]*np.sin(theta[ids_border])
    theta_border = theta[ids_border]
    
    #plt.scatter(dx[ids_border],dy[ids_border])
    #plt.scatter(0.,0.,c='k')
    #plt.axis('equal')
    #plt.plot(r[ids_border]*np.cos(theta[ids_border]),r[ids_border]*np.sin(theta[ids_border]))


    # cleaning to avoid jagged lines:
    condition = sharpest_ang > 0.
    cosMax = np.cos(sharpest_ang * np.pi/180.) #0.75
    while condition:
        i0 = 0
        n_vertex = x_border.shape[0]
        id_check = []

        # select verteces for which the subdtended angle is smaller than sharpest_ang, i.e. cos ang > cosMax
        for i in range(1,n_vertex+1):
            i1 = i % n_vertex
            i2 = (i1 + 1) % n_vertex
            prod = (x_border[i1] - x_border[i0]) * (x_border[i1] - x_border[i2]) + (y_border[i1] - y_border[i0]) * (y_border[i1] - y_border[i2])
            L01 = ((x_border[i1] - x_border[i0]) ** 2 + (y_border[i1] - y_border[i0]) ** 2) ** 0.5
            L12 = ((x_border[i1] - x_border[i2]) ** 2 + (y_border[i1] - y_border[i2]) ** 2) ** 0.5
            cosTheta = prod / (L01 * L12)
            #print(i0,i1,i2,cosTheta)
            if cosTheta > cosMax:
                #print(i1)
                id_check.append(i1)
            i0 = i1

        id_check = np.array(id_check)
        n_check = len(id_check)
        ii = 0
        id_to_remove = []
        cosTheta_arr = np.empty(2)
        #if n_check >0:
        #    plt.scatter(x_border,y_border,c='k',alpha=0.5)

        # 2 criteria to removo jagged line: for isolated verteces for which cos ang > cosMax, and for consecuvive vertexes
        while ii < n_check:
            #print(ii)
            ii_loop = ii
            if (id_check[(ii+1) % n_check] - id_check[ii]) == 1:
                id_to_remove_loop = ii

                # 1. consecuvive verteces: consider all consecuvive verteces satisfing cos(ang) > cosMax and remove the closest to the void center
                while (id_check[(ii+1) % n_check] - id_check[ii]) == 1:
                    #print('    while: ii:',ii,id_to_remove_loop,ii,(ii+1) % n_check,id_check[[id_to_remove_loop,ii,(ii+1) % n_check]],
                    #    r_border[id_check[[id_to_remove_loop,ii,(ii+1) % n_check]]])
                    id_to_remove_loop = [id_to_remove_loop,ii,(ii+1) % n_check][np.argmin(r_border[id_check[[id_to_remove_loop,ii,(ii+1) % n_check]]])]
                    ii += 1
                    if ii - ii_loop >= n_check:
                        break
                ii_next = ii    
                # extra loop in case the last and first verteces are conescutive
                ii = id_to_remove_loop
                while (id_check[ii] + n_vertex - id_check[(ii+n_check-1)%n_check]) == 1:
                    #print('    special while: ii:',ii,id_to_remove_loop,ii,(ii+n_check-1)%n_check,id_check[[id_to_remove_loop,ii,(ii+n_check-1)%n_check]])
                    id_to_remove_loop = [id_to_remove_loop,ii,(ii+n_check-1)%n_check][np.argmin(r_border[id_check[[id_to_remove_loop,ii,(ii+n_check-1)%n_check]]])]
                    ii -= 1
                    n_check -= 1
                    if ii_loop - ii >= n_check:
                        break
                ii = ii_next #+ 1
                id_to_remove.append(id_check[id_to_remove_loop])
                #print('    exit while: ii:',ii,id_to_remove_loop,id_check[[ii,id_to_remove_loop]])
            else:
                # 2. Isoloted verteces. We measure the angle of the previous and next step and we consider the one with smaller angle. 
                # Remove the closest to the void center among the isoloted vertex and the adiacent with smaller angle. 
                
                # previous step
                i0 = (id_check[ii] + n_vertex - 2) % n_vertex
                i1 =  (i0 + 1) % n_vertex
                i2 =  (i1 + 1) % n_vertex
                #print('    ',i0,i1,i2)
                prod = (x_border[i1] - x_border[i0]) * (x_border[i1] - x_border[i2]) + (y_border[i1] - y_border[i0]) * (y_border[i1] - y_border[i2])
                L01 = ((x_border[i1] - x_border[i0]) ** 2 + (y_border[i1] - y_border[i0]) ** 2) ** 0.5
                L12 = ((x_border[i1] - x_border[i2]) ** 2 + (y_border[i1] - y_border[i2]) ** 2) ** 0.5
                cosTheta_arr[0] = prod / (L01 * L12)

                # next step
                i0 = id_check[ii] % n_vertex
                i1 =  (i0 + 1) % n_vertex
                i2 =  (i1 + 1) % n_vertex
                prod = (x_border[i1] - x_border[i0]) * (x_border[i1] - x_border[i2]) + (y_border[i1] - y_border[i0]) * (y_border[i1] - y_border[i2])
                L01 = ((x_border[i1] - x_border[i0]) ** 2 + (y_border[i1] - y_border[i0]) ** 2) ** 0.5
                L12 = ((x_border[i1] - x_border[i2]) ** 2 + (y_border[i1] - y_border[i2]) ** 2) ** 0.5
                cosTheta_arr[1] = prod / (L01 * L12)
                #print('    ',i0,i1,i2)

                ids_compare = [id_check[ii],(id_check[ii] + n_vertex + 2*np.argmax(cosTheta_arr)-1) % n_vertex]

                id_to_remove.append(ids_compare[np.argmin(r_border[ids_compare])])
                #print('    single:',ids_compare,id_to_remove[-1])  
            ii += 1
        condition = len(id_to_remove) > 0
        if not condition:
            break
        mask = np.ones(n_vertex,dtype=np.bool_)
        mask[id_to_remove] = False
        #plt.scatter(x_border[~mask],y_border[~mask],c='m')
        x_border = x_border[mask]
        y_border = y_border[mask]
        r_border = r_border[mask]
        theta_border = theta_border[mask]
        n_vertex -= np.sum(~mask)
        #plt.scatter(x_border,y_border)
        #plt.plot(x_border,y_border)

    condition = frac_min_dist > 0
    # merge verteces closer then mean_dist * frac_min_dist, where mean_dist is the mean distance among consecuvite verteces of the close polygonal line.
    # sobstitute these verteces with the mean of each close pair.
    while condition:
        rel_dist = np.empty(n_vertex)
        rel_dist[:-1] = np.sqrt((x_border[1:] - x_border[:-1])**2 + (y_border[1:] - y_border[:-1])**2)
        rel_dist[-1] = np.sqrt((x_border[0] - x_border[-1])**2 + (y_border[0] - y_border[-1])**2)
        mean_dist = np.mean(rel_dist)
        #std_dist = np.std(rel_dist)
        mask = rel_dist < mean_dist * frac_min_dist

        id_mask = np.arange(n_vertex)[mask]
        x_mean = 0.5 * (x_border[id_mask] + x_border[np.remainder(id_mask+1,n_vertex)])
        y_mean = 0.5 * (y_border[id_mask] + y_border[np.remainder(id_mask+1,n_vertex)])
        theta_mean = np.arctan(y_mean/x_mean) 
        theta_mean[(x_mean < 0)] += np.pi
        theta_mean[(theta_mean < 0)] += 2.*np.pi
        r_mean = np.sqrt(x_mean**2 + y_mean**2)

        id_remove = []
        n_mask = len(id_mask)
        for ii in range(n_mask):
            if (id_mask[(ii+1)%n_mask] - id_mask[ii] + n_vertex) % n_vertex >= 1:
                id_remove.append((id_mask[ii] + 1) % n_vertex)
        #print(np.sum(mask),id_mask,id_remove)
        x_border[mask] = x_mean
        y_border[mask] = y_mean
        r_border[mask] = r_mean
        theta_border[mask]= theta_mean

        mask[:] = True
        mask[id_remove] = False
        x_border = x_border[mask]
        y_border = y_border[mask]
        r_border = r_border[mask]
        theta_border = theta_border[mask]
        n_vertex -= np.sum(~mask)
        #plt.scatter(x_border,y_border)
        #plt.plot(x_border,y_border)

        condition = np.sum(~mask) > 0


    return x_border, y_border, theta_border, r_border




# load dataset:
path_to_test= 'data/box/examples/example_simulation/sim_ss1.0/sample_sim_ss1.0_z0.00_d00'
catalog = vu.loadVoidCatalog(path_to_test, 
                            dataPortion="all", 
                            loadParticles=True,
                            untrimmed=True)
print('box info:','\n    nvoids:',catalog.numVoids,'\n    num part:',catalog.numPartTot,
      '\n    box len',catalog.boxLen,'\n    vol norm:',catalog.volNorm,'\n    fake density:',catalog.sampleInfo.fakeDensity)


# select void
Reff = vu.getArray(catalog.voids,'radius')
ivd = np.argmax(Reff)


numPart = catalog.voids[ivd].numPart
xyz_ref = catalog.voids[ivd].macrocenter

voidPart = vu.getVoidPart(catalog, catalog.voids[ivd].voidID)
xyz_part = np.empty((numPart,3))
xyz_part[:,0] = vu.getArray(voidPart,'x')
xyz_part[:,1] = vu.getArray(voidPart,'y')
xyz_part[:,2] = vu.getArray(voidPart,'z')
# PBC correction
for i in range(3):
    xyz_part[:,i] += ((xyz_part[:,i] - xyz_ref[i]) < -0.5 * catalog.boxLen[i]) *  catalog.boxLen[i]
    xyz_part[:,i] -= ((xyz_part[:,i] - xyz_ref[i]) > 0.5 * catalog.boxLen[i]) *  catalog.boxLen[i]

#recenter wrt void center
dx = xyz_part[:,0] - xyz_ref[0]
dy = xyz_part[:,1] - xyz_ref[1]

# find void border
dx_border, dy_border, theta_border, r_border = void_border(dx,dy,nTheta_bins=101,sharpest_ang=80.,frac_min_dist=0.5)



plt.scatter(dx,dy,s=5)
plt.scatter(dx_border,dy_border,s=5)
plt.axis('equal')

# make boreder periodic
dx_border = np.append(dx_border,[dx_border[0]])
dy_border = np.append(dy_border,[dy_border[0]])
theta_border = np.append(theta_border,[theta_border[0]+ 2.*np.pi])
r_border = np.append(r_border,[r_border[0]])
plt.plot(dx_border,dy_border)
plt.scatter(0.,0.,c='k')

# make border smooth with cubic interpolation
th = np.linspace(0.,2*np.pi,301)
r_interp = CubicSpline(theta_border,r_border,bc_type='periodic')(th)
plt.plot(r_interp*np.cos(th),r_interp*np.sin(th),lw=10,alpha=0.2)
plt.savefig('test_contour.png')
