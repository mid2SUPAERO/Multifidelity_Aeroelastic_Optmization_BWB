# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:05:31 2016

@author: jmascolo
"""

import sqlitedict

import numpy as np
import matplotlib.pyplot as plt

recfile = input('Specify the fidelity level for the plot ("high" "low"): ')
if recfile == 'high':
    recfile = 'mda_h.sqlite3'
    displacement_var = 'structures_h.u'
elif recfile == 'low':
    recfile = 'mda_l.sqlite3'
    displacement_var = 'structures.u'
db = sqlitedict.SqliteDict(recfile, 'iterations')

font_size=12.5
label_size=12.5

#Create a numpy array with the number of iterations
X = np.arange(1, len(db)+1)

#Lists contaning the norm of the unknowns for all the iterations
norm = []

#Vertical displacement of the wing tip
disp = []

for it in db:
    unknowns = []
    for key in db[it]['Unknowns']:
        unknowns.append(db[it]['Unknowns'][key])
    
    unknowns_vec = unknowns[0].flatten()
    
    for i in range(len(unknowns)-1):
        unknowns_vec = np.append(unknowns_vec, unknowns[i+1].flatten())
        
    norm.append(np.linalg.norm(unknowns_vec))
    
    disp.append(max(db[it]['Unknowns'][displacement_var][:, 2].min(), db[it]['Unknowns'][displacement_var][:, 2].max(), key=abs))

xmin = 1
#xmax = 10

norm_min = 0
norm_max = np.max(norm)

plt.figure(1)
plt.plot(X, norm, color="green")
plt.xlabel('Iterations', fontsize=font_size)
plt.ylabel('Norm of the unknowns', fontsize=font_size)
plt.tick_params(labelsize=label_size)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([norm_min,norm_max])

plt.savefig('plot_unknowns_norm.svg', bbox_inches='tight')
plt.close()

#plt.show()

disp_min = np.min(disp)
disp_max = np.max(disp)

plt.figure(2)
plt.plot(X, disp, color="green")
plt.xlabel('Iterations', fontsize=font_size)
plt.ylabel('Wingtip vertical displacement', fontsize=font_size)
plt.tick_params(labelsize=label_size)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([disp_min,disp_max])

plt.savefig('plot_wingtip_disp.svg', bbox_inches='tight')
plt.close()

#plt.show()
