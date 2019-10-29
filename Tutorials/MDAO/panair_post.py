# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:09:23 2016

@author: © Joan Mas Colomer
"""

from shutil import copyfile

#Function that checks whether a string can be converted into an integer
def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

user_input = input('Enter the name of the .msh file of the wing, please\n')

#Make a copy of the original .msh file containing the mesh
copyfile(user_input+'.msh', user_input+'_post.msh')

#List containing the identification of the elements and their Cp value
cp_elm = []

#Open the PANAIR output file
with open('./alpha/panair.out') as f:
    lines = f.readlines()
    lines = [i.split() for i in lines]

    results_begin = lines.index(['0*b*solution'])
    results_end = lines.index(['full', 'configuration', 'forces', 'and', 'moments', 'summary'])

    for line in lines:
        #Get panel pressure coefficients
        if len(line) > 1 and lines.index(line) > results_begin and lines.index(line) < results_end and len(lines[lines.index(line)-1]) == 0:
            if isint(line[0]) and isint(line[1]):
                cp_elm.append([int(line[1]), float(lines[lines.index(line)+1][10])])

with open(user_input+'_post.msh', 'a') as f:
    f.write('$ElementData\n')
    f.write('1\n')
    f.write('"Cp distribution"\n')
    f.write('1\n')
    f.write('0.0\n')
    f.write('3\n')
    f.write('0\n')
    f.write('1\n')
    f.write(str(len(cp_elm))+'\n')
    for elm in cp_elm:
        f.write(str(elm[0])+' '+str(elm[1])+'\n')
    f.write('$EndElementData')
