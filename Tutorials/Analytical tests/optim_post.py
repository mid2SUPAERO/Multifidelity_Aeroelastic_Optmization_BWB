# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:05:31 2016

@author: jmascolo
"""

import sqlitedict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

filename = input('Sqlite file to process ? ')
namenoext = filename
filename = filename + '.sqlite3'
db = sqlitedict.SqliteDict(filename, 'iterations')

line_width = 2.0
label_size = 20
font_size = 20

#Create a numpy array with the number of iterations
X = np.arange(1, len(db)+1)

#Lists contaning the objective function value and constraints of all the iterations
obj = []
con1 = []
con2 = []

for it in db:
    obj.append(db[it]['Unknowns']['obj'])
    con1.append(db[it]['Unknowns']['con1'])
    con2.append(db[it]['Unknowns']['con2'])

xmin = 1
#xmax = 10

f_min = 0
f_max = np.max(obj)

con1_min = np.min(con1)
con1_max = np.max(con1)

con2_min = min(0., np.min(con2))
con2_max = np.max(con2)

plt.figure(1)
plt.plot(X, obj, color="green", linewidth=line_width)
plt.xlabel('Iterations', fontsize=font_size)
plt.ylabel('Objective function (obj)', fontsize=font_size)
plt.tick_params(labelsize=label_size)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([f_min,f_max])
axes.axhline()

plt.tight_layout()
plt.savefig(namenoext + '_objective_sellar.pdf', bbox_inches='tight')

plt.figure(2)
plt.plot(X, con1, color="red", linewidth=line_width)
plt.xlabel('Iterations', fontsize=font_size)
plt.ylabel('Constraint 1', fontsize=font_size)
plt.tick_params(labelsize=label_size)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([con1_min,con1_max])
axes.axhline()

plt.tight_layout()
plt.savefig(namenoext + '_constraint_1.pdf', bbox_inches='tight')

plt.figure(3)
plt.plot(X, con2, color="red", linewidth=line_width)
plt.xlabel('Iterations', fontsize=font_size)
plt.ylabel('Constraint 2', fontsize=font_size)
plt.tick_params(labelsize=label_size)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([con2_min,con2_max])
axes.axhline()

plt.tight_layout()
plt.savefig(namenoext + '_constraint_2.pdf', bbox_inches='tight')
plt.show()
