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

rcParams['figure.figsize'] = 12, 10
sns.set_style('ticks', {'xtick.direction': u'in', 'ytick.direction': u'in'})

def main():
    
    fileList = glob.glob('cu_au_*.tif')
    #fileList = ['cu_pd_5_300_200_2.tif']
    image_list = []
    for file in fileList:
        image_list.append(SEM_IM.SEM_IM(file))
        image_list[-1].crop_bottom()
        image_list[-1].run_analysis()
        image_list[-1].compute_rg()
        #image_list[-1].compute_rdf()
        image_list[-1].plot_results()
    pickle.dump( image_list, open( "images.pickle", "wb" ) )



if __name__=='__main__': main()
