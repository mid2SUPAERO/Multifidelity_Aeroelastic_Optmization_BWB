# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:50:10 2016

@author: © Joan Mas Colomer
"""

from __future__ import print_function

import timeit

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ScipyGMRES, SqliteRecorder, view_model

from aerostructures import NastranStatic, DisplacementTransfer, Panair, LoadTransfer, Interpolation, StaticStructureProblemDimensions, StaticStructureProblemParams, AeroProblemDimensions, AeroProblemParams, NLGaussSeidel, Filter

if __name__ == "__main__":

    #Interpolation function type and setup
    function_type = 'thin_plate'
    bias = (1.,100.,1.)

    #Symmetry plane index
    sym_plane_index = 2

    case_name = 'alpha_low'
    case_name_h = 'alpha_high'

    #Problem parameters
    Sw = 383.689555
    V = 252.16168
    rho_a = 0.38058496
    Mach = 0.85
    alpha = 1.340
    b = 58.7629
    c = 7.00532
    E = 6.89e10
    nu = 0.31
    rho_s = 2795.67
    n = 1.
    #Aerodynamic template files for both fidelities
    #Hi-Fi
    aero_template_l = 'aero_template_l.wgs'
    #Lo-Fi
    aero_template_h = 'aero_template_h.wgs'
    # Multi-fidelity options 'low', for low-fidelity; 'high', for high-fidelity; 'multi', for multi-fidelity
    fidelity = input('Please enter the fidelity level: low, high or multi: ')
    # Iterations for the low-fidelity part in multi-fidelity mode
    it_l = None
    if fidelity == 'multi':
        it_l = int(input('Please enter the iteration limit for the Lo-Fi level: '))
    

    structure_problem_dimensions = StaticStructureProblemDimensions()
    
    ns = structure_problem_dimensions.ns
    ns_all = structure_problem_dimensions.ns_all
    node_id = structure_problem_dimensions.node_id
    node_id_all = structure_problem_dimensions.node_id_all
    n_stress = structure_problem_dimensions.n_stress
    u = np.zeros((ns, 3))
    ul = np.zeros((ns, 3)) #Auxiliary variable to transfer the displacement field between fidelities
    tn = 0
    mn = 0
    sn = 0

    #Low fidelity instance -- aero_template_l.wgs
    aero_problem_dimensions = AeroProblemDimensions(aero_template_l)
    na = aero_problem_dimensions.na
    network_info = aero_problem_dimensions.network_info
    
    #High fidelity instance -- aero_template_h.wgs
    aero_problem_dimensions_h = AeroProblemDimensions(aero_template_h)
    na_h = aero_problem_dimensions_h.na
    network_info_h = aero_problem_dimensions_h.network_info
    
    structure_problem_params = StaticStructureProblemParams(node_id, node_id_all)
    #Low fidelity instance -- aero_template_l.wgs 
    aero_problem_params = AeroProblemParams(aero_template_l)
    
    #High fidelity instance -- aero_template_h.wgs
    aero_problem_params_h = AeroProblemParams(aero_template_h)

    node_coord = structure_problem_params.node_coord
    node_coord_all = structure_problem_params.node_coord_all

    apoints_coord = aero_problem_params.apoints_coord
    apoints_coord_h = aero_problem_params_h.apoints_coord

    top = Problem()
    root = top.root = Group()

    #Add independent variables
    root.add('wing_area', IndepVarComp('Sw', Sw))
    root.add('airspeed', IndepVarComp('V', V))
    root.add('air_density', IndepVarComp('rho_a', rho_a))
    root.add('Mach_number', IndepVarComp('Mach', Mach))
    root.add('angle_of_attack', IndepVarComp('alpha', alpha))
    root.add('wing_span', IndepVarComp('b', b))
    root.add('wing_chord', IndepVarComp('c', c))
    root.add('s_coord', IndepVarComp('node_coord', node_coord))
    root.add('s_coord_all', IndepVarComp('node_coord_all', node_coord_all))
    root.add('a_coord', IndepVarComp('apoints_coord', apoints_coord))
    root.add('a_coord_h', IndepVarComp('apoints_coord', apoints_coord_h))
    root.add('load_factor', IndepVarComp('n', n))
    
    #Lo-Fi Group
    mda_l = Group()

    #Add disciplines to the low fidelity group CHECK INPUTS
    mda_l.add('displacement_transfer', DisplacementTransfer(na, ns)) 
    mda_l.add('aerodynamics', Panair(na, network_info, case_name, aero_template_l, sym_plane_index=sym_plane_index)) 
    mda_l.add('load_transfer', LoadTransfer(na, ns))
    mda_l.add('structures', NastranStatic(node_id, node_id_all, n_stress, tn, mn, sn, case_name))
    
    
    #Hi-Fi Group
    mda_h = Group()
    
    #Add disciplines to the high-fidelity group CHECK INPUTS
    mda_h.add('mult_filter', Filter(ns, fidelity))
    mda_h.add('displacement_transfer_h', DisplacementTransfer(na_h, ns))
    mda_h.add('aerodynamics_h', Panair(na_h, network_info_h, case_name_h, aero_template_h, sym_plane_index=sym_plane_index))    
    mda_h.add('load_transfer_h', LoadTransfer(na_h, ns))
    mda_h.add('structures_h', NastranStatic(node_id, node_id_all, n_stress, tn, mn, sn, case_name_h))
    
    mda_l.add('inter', Interpolation(na, ns, function = function_type, bias = bias))
    mda_h.add('inter_h', Interpolation(na_h, ns, function = function_type, bias = bias))

    #Define solver type and tolerance for MDA Lo-Fi
    mda_l.nl_solver = NLGaussSeidel()
    #The solver execution limit is used to control fidelity levels
    if fidelity == 'high':
        mda_l.nl_solver.options['maxiter'] = 0 #No Lo-Fi iterations
    elif fidelity == 'multi':
        mda_l.nl_solver.options['maxiter'] = it_l #Adds the limit for the execution of the MDA Solver
          
    mda_l.nl_solver.options['rutol'] = 1.e-2 
    mda_l.nl_solver.options['use_aitken'] = True
    mda_l.nl_solver.options['aitken_alpha_min'] = 0.1
    mda_l.nl_solver.options['aitken_alpha_max'] = 1.5

    mda_l.ln_solver = ScipyGMRES()

    #Define solver type and tolerance for MDA Hi-Fi
    mda_h.nl_solver = NLGaussSeidel()
    #The solver execution limit is used to control fidelity levels
    if fidelity == 'low':
        mda_h.nl_solver.options['maxiter'] = 0
        
    mda_h.nl_solver.options['rutol'] = 1.e-2
    mda_h.nl_solver.options['use_aitken'] = True
    mda_h.nl_solver.options['aitken_alpha_min'] = 0.1
    mda_h.nl_solver.options['aitken_alpha_max'] = 1.5

    mda_h.ln_solver = ScipyGMRES()
    
    root.add('mda_group_l', mda_l, promotes=['*'])
    
    #Explicit connection Lo-Fi
    root.mda_group_l.connect('displacement_transfer.delta','aerodynamics.delta')
    root.mda_group_l.connect('inter.H','displacement_transfer.H')
    root.mda_group_l.connect('structures.u','displacement_transfer.u')
    root.mda_group_l.connect('aerodynamics.f_a','load_transfer.f_a')
    root.mda_group_l.connect('load_transfer.f_node','structures.f_node')
    root.mda_group_l.connect('inter.H','load_transfer.H')
    root.connect('a_coord.apoints_coord','inter.apoints_coord')
    root.connect('a_coord.apoints_coord', 'aerodynamics.apoints_coord')
    #Connect Indep Variables
    root.connect('wing_area.Sw', 'aerodynamics.Sw')
    root.connect('airspeed.V', 'aerodynamics.V')
    root.connect('air_density.rho_a', 'aerodynamics.rho_a')
    root.connect('Mach_number.Mach', 'aerodynamics.Mach')
    root.connect('angle_of_attack.alpha', 'aerodynamics.alpha')
    root.connect('wing_span.b', 'aerodynamics.b')
    root.connect('wing_chord.c', 'aerodynamics.c')
    root.connect('load_factor.n','structures.n')
    root.connect('s_coord.node_coord', 'inter.node_coord')
    root.connect('s_coord_all.node_coord_all', 'structures.node_coord_all')
      
    root.add('mda_group_h', mda_h, promotes=['*'])
    
    #Explicit connection Hi-Fi
    root.mda_group_h.connect('displacement_transfer_h.delta','aerodynamics_h.delta')
    root.mda_group_h.connect('inter_h.H','displacement_transfer_h.H')
    root.mda_group_h.connect('mult_filter.us','displacement_transfer_h.u')
    root.mda_group_h.connect('aerodynamics_h.f_a','load_transfer_h.f_a')
    root.mda_group_h.connect('load_transfer_h.f_node','structures_h.f_node')
    root.mda_group_h.connect('inter_h.H','load_transfer_h.H')
    root.mda_group_h.connect('structures_h.u','mult_filter.u')
    root.connect('a_coord_h.apoints_coord','inter_h.apoints_coord')
    root.connect('a_coord_h.apoints_coord', 'aerodynamics_h.apoints_coord')
    #Connect Indep Variables
    root.connect('wing_area.Sw', 'aerodynamics_h.Sw')
    root.connect('airspeed.V', 'aerodynamics_h.V')
    root.connect('air_density.rho_a', 'aerodynamics_h.rho_a')
    root.connect('Mach_number.Mach', 'aerodynamics_h.Mach')
    root.connect('angle_of_attack.alpha', 'aerodynamics_h.alpha')
    root.connect('wing_span.b', 'aerodynamics_h.b')
    root.connect('wing_chord.c', 'aerodynamics_h.c')
    root.connect('load_factor.n','structures_h.n')
    root.connect('s_coord.node_coord', 'inter_h.node_coord')
    root.connect('s_coord_all.node_coord_all', 'structures_h.node_coord_all')
    
    #Multifidelity explicit connections
    
    root.connect('structures.u', 'mult_filter.ul')
    
    #Recorder Lo-Fi
    recorder_l = SqliteRecorder('mda_l.sqlite3')
    recorder_l.options['record_metadata'] = False
    #Recorder Hi-Fi
    recorder_h = SqliteRecorder('mda_h.sqlite3')
    recorder_h.options['record_metadata'] = False
    # recorder.options['includes'] =
    top.root.mda_group_l.nl_solver.add_recorder(recorder_l)
    top.root.mda_group_h.nl_solver.add_recorder(recorder_h)


    #Define solver type
    root.ln_solver = ScipyGMRES()

    tic=timeit.default_timer()                    
    top.setup()
    toc=timeit.default_timer()
    print("Set up time = " + str(toc - tic)) #elapsed time in seconds    
    view_model(top, show_browser=False)
    tic=timeit.default_timer()
    top.run()
    toc=timeit.default_timer()
    print("Run time = " + str(toc - tic)) #elapsed time in seconds
    
