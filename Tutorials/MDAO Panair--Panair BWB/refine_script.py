# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:09:23 2016
This script refines a given mesh and exports it as a structured mesh that can be copied to a 
wgs file
@author: © Gilberto Ruiz
"""
import Generator as G
import Converter as C
import numpy as np
import os

upper_mesh = C.convertFile2Arrays('upper.mesh')
lower_mesh = C.convertFile2Arrays('lower.mesh')
wingtip_mesh = C.convertFile2Arrays('wingtip.mesh')
print('----Reading complete----')
aux_upper = C.array('x,y,z', 1, 11, 15); # first numer is the number of sections
aux_lower = C.array('x,y,z', 1, 11, 15);  # second number is number of lines in wgs
aux_wingtip = C.array('x,y,z', 1, 11, 3); #third number is numner of points per line of wgs
aux_upper[1] = upper_mesh[0][1]
refined_upper = G.refine(aux_upper, 3, 2)
aux_lower[1] = lower_mesh[0][1]
refined_lower = G.refine(aux_lower, 3, 2)
aux_wingtip[1] = wingtip_mesh[0][1]
refined_wingtip = G.refine(aux_wingtip, 3, 2)
print('----Refinement complete----')

np.savetxt('upper_h.txt', np.transpose([refined_upper[1][0],refined_upper[1][1],refined_upper[1][2]]), fmt='%.12f', delimiter='  ')
np.savetxt('lower_h.txt', np.transpose([refined_lower[1][0],refined_lower[1][1],refined_lower[1][2]]), fmt='%.12f', delimiter='  ')
np.savetxt('wingtip_h.txt', np.transpose([refined_wingtip[1][0],refined_wingtip[1][1],refined_wingtip[1][2]]), fmt='%.12f', delimiter='  ')