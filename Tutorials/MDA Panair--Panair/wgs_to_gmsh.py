# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 10:46:32 2016

@author: © Joan Mas Colomer
"""

import numpy as np

#Function that checks whether a string can be converted into a float
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

#Function that checks whether a string can be converted into an int
def isint(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

user_input = input('Enter the name of the .wgs file you want to convert to .msh, please\n')

points = {}
cols = {}
rows = {}

networks = {}
panels = {}

prec_points = {}
prec_pan = {}

#network counter
nw = 0

#Open the current WGS file
with open(user_input+'.wgs') as f:
    lines = f.readlines()
    lines = [i.split() for i in lines]

    for line in lines:

        if all(isint(item) for item in line):
            nw += 1
            cols[nw] = int(line[1])
            rows[nw] = int(line[2])
            points[nw] = {}
            point_num = 0

        if all(isfloat(item) for item in line):
            if len(line) == 3:
                point_num += 1
                points[nw][point_num] = [float(line[0]), float(line[1]), float(line[2])]

            if len(line) == 6:
                point_num += 1
                points[nw][point_num] = [float(line[0]), float(line[1]), float(line[2])]

                point_num += 1
                points[nw][point_num] = [float(line[3]), float(line[4]), float(line[5])]

#For each network, create an array with the same shape containing the point numbers
for item in points:
    networks[item] = np.zeros((rows[item], cols[item]), dtype=np.dtype(int))
    panels[item] = {}

    point_counter = 0
    for j in range(cols[item]):
        for i in range(rows[item]):
            point_counter += 1
            networks[item][i][j] = point_counter

    #For each panel of the network, create a list containing the points of the panel
    panel_counter = 0
    for j in range(cols[item]-1):
        for i in range(rows[item]-1):
            panel_counter += 1
            panels[item][panel_counter] = [networks[item][i][j], networks[item][i][j+1], networks[item][i+1][j+1], networks[item][i+1][j]]

#Compute total number of points
total_points = 0
for item in points:
    total_points += len(points[item])

#Compute total number of elements (panels)
total_elements = 0
for item in panels:
    total_elements += len(panels[item])

#Dictionary that contains the number of points of the previous networks
counter = 0
for item in points:
    prec_points[item] = counter
    counter += len(points[item])

#Dictionary that contains the number of panels of the previous networks
counter = 0
for item in panels:
    prec_pan[item] = counter
    counter += len(panels[item])

#Write the .msh file
with open(user_input+'.msh','w') as f:
    f.write('$MeshFormat\n')
    f.write('2.2 0 8\n')
    f.write('$EndMeshFormat\n')
    f.write('$Nodes\n')
    f.write(str(total_points)+'\n')

    for item in points:
        for point in points[item]:
            f.write(str(point+prec_points[item])+' '+str(points[item][point][0])+' '+str(points[item][point][1])+' '+str(points[item][point][2])+'\n')

    f.write('$EndNodes\n')
    f.write('$Elements\n')
    f.write(str(total_elements)+'\n')

    for item in panels:
        for panel in panels[item]:
            f.write(str(panel+prec_pan[item])+' 3 2 1 1 '+str(panels[item][panel][0]+prec_points[item])+' '+str(panels[item][panel][1]+prec_points[item])+' '+str(panels[item][panel][2]+prec_points[item])+' '+str(panels[item][panel][3]+prec_points[item])+'\n')

    f.write('$EndElements\n')
