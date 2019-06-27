# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:05:31 2016

@author: jmascolo
"""

import sqlitedict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

db = sqlitedict.SqliteDict('mdao.sqlite3', 'iterations')

line_width = 2.0
label_size = 20
font_size = 20

#Create a numpy array with the number of iterations
X = np.arange(1, len(db)+1)

#Lists contaning the objective function value and constraints of all the iterations
CDi = []
con_l_u = []
con_s = []

for it in db:
    CDi.append(db[it]['Unknowns']['CDi'])
    con_l_u.append(db[it]['Unknowns']['con_l_u'])
    con_s.append(db[it]['Unknowns']['con_s'])

xmin = 1
#xmax = 10

f_min = 0
f_max = np.max(CDi)

con_l_u_min = np.min(con_l_u)
con_l_u_max = np.max(con_l_u)

con_s_min = min(0., np.min(con_s))
con_s_max = np.max(con_s)

plt.figure(1)
plt.plot(X, CDi, color="green", linewidth=line_width)
plt.xlabel('Iterations', fontsize=font_size)
plt.ylabel('Objective function (CDi)', fontsize=font_size)
plt.tick_params(labelsize=label_size)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([f_min,f_max])
axes.axhline()

plt.tight_layout()
plt.savefig('objective_cdi.pdf', bbox_inches='tight')

plt.figure(2)
plt.plot(X, con_l_u, color="red", linewidth=line_width)
plt.xlabel('Iterations', fontsize=font_size)
plt.ylabel('Cruise Lift Constraint', fontsize=font_size)
plt.tick_params(labelsize=label_size)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([con_l_u_min,con_l_u_max])
axes.axhline()

plt.tight_layout()
plt.savefig('cruise_lift_constraint.pdf', bbox_inches='tight')

plt.figure(3)
plt.plot(X, con_s, color="red", linewidth=line_width)
plt.xlabel('Iterations', fontsize=font_size)
plt.ylabel('Stress Constraint', fontsize=font_size)
plt.tick_params(labelsize=label_size)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([con_s_min,con_s_max])
axes.axhline()

plt.tight_layout()
plt.savefig('stress_constraint.pdf', bbox_inches='tight')
plt.show()