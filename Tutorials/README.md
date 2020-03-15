# Multifidelity aeroelastic optimization tutorials
Each tutorial case requires the aerostructures package, specifically the g.ruiz branch. 
The tutorial file is a jupyter notebook, and there is also a python script with the same commands for direct execution.
## MDA Panair--Panair
Includes the necessary files to execute the MDA of the sample case, using Panair for both fidelity levels and introducing the variation with two different aerodynamic meshes for each fidelity.
## MDAO Panair--Panair 
Includes the necessary files to execute the full MDAO of the sample case, using Panair for both fidelity levels and introducing the variation with two different aerodynamic meshes for each fidelity.
## MDAO Panair--ADflow 
Includes the necessary files to execute the full MDAO of the sample case, using Panair for the low fidelity case and ADflow for the high fidelity case. Note that the solution of this problem might be unstable depending on the input parameters and constraints.
## Analytical tests
Includes the necessary files to test the multifidelity MDAO architecture with "simple" analytical functions used to benchmarck MDAO problems. 