#!usr/bin/python

import SEM_IM
import glob
import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt
import glob
import pylab
from pylab import rcParams
import seaborn as sns
from pylab import rcParams
import os
import pickle
import matplotlib as mpl
mpl.rcParams['axes.color_cycle'] = ['k','r', 'b', 'y', 'm']

rcParams['figure.figsize'] = 12, 10
sns.set_style('ticks', {'xtick.direction': u'in', 'ytick.direction': u'in'})

def main():
    
    coverage = [ 0.33, 0.54, 0.72]
    tcr = [ .00135, -.0000071, -.001]
    

    image_list = pickle.load( open( "images.pickle", "rb" ) )
    area_dist = []
    perim_dist = []
    rg_dist = []
    coverage = []
    
    for image in image_list:
        area_dist.extend(image.size_dist[1:-1])
        perim_dist.extend(image.perimeters[1:-1])
        rg_dist.extend(image.rg_m[1:-1])
        coverage.append(image.coverage)

    area_dist = np.asarray(area_dist)*image_list[0].scale**2
    perim_dist = np.asarray(perim_dist)*image_list[0].scale
    rg_dist = np.asarray(rg_dist)*image_list[0].scale
    coverage = np.asarray(coverage)
    
    # Plot the cropped picture
    pylab.gray()
    plt.subplot(421)
    plt.xticks([])
    plt.yticks([])
    plt.title(image_list[0].name)
    pylab.imshow(image_list[0].image_orig)
    
    # Plot thresholded image
    plt.subplot(422)
    plt.xticks([])
    plt.yticks([])
    plt.title('coverage = %.2f +- %.2f' % (coverage.mean(), coverage.std()))
    pylab.imshow(image_list[0].image > image_list[0].thresh)
    
    # Plot distinguished image
    pylab.plasma()
    plt.subplot(423)
    plt.xticks([])
    plt.yticks([])
    plt.title('%d seperate islands' % image_list[0].num_islands)
    pylab.imshow(image_list[0].labeled)
    
    # Plot distinguished image
    plt.subplot(424)
    plt.xticks([])
    plt.yticks([])
    pylab.imshow(image_list[0].labeled_p)
    
    # Plot histogram
    
    plt.subplot(425)
    plt.hist(area_dist, normed = True, align = 'mid', bins= 100)
    plt.xlabel('Projected Area (nm$^2$)')
    plt.ylabel('Probability')
    plt.xlim((0, area_dist.mean() + 4*area_dist.std()))
    plt.title('Average Area =  %.2f nm$^2$' % area_dist.mean())
    
    # Plot histogram
    plt.subplot(426)
    plt.hist(perim_dist , normed = True, align = 'mid', bins = 100)
    plt.xlabel('Perimeter (nm)')
    plt.ylabel('Probability')
    plt.xlim((0, perim_dist.mean() + 4*perim_dist.std()))
    plt.title('Average Perimeter = %.2f nm' % perim_dist.mean())
    
    
    plt.subplot(428)
    print len(rg_dist), len(area_dist)
    plt.plot( rg_dist, area_dist, 'o')
    plt.xlabel('Radius of gyration [nm]')
    plt.ylabel('Area [nm$^2$]')
    plt.yscale('log')
    plt.xscale('log')
    
    # Plot histogram
    plt.subplot(427)
    plt.hist(rg_dist , normed = True, align = 'mid', bins= 100)
    plt.xlim((0, rg_dist.mean() + 4*rg_dist.std()))
    plt.xlabel('Radius of Gyration [nm]')
    plt.ylabel('Probability')


    plt.show()
    return

if __name__=='__main__': main()
