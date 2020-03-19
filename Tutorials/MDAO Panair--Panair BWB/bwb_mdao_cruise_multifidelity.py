# -*- coding: utf-8 -*-
"""

@author: © Gilberto Ruiz
"""

from __future__ import print_function

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyGMRES, SqliteRecorder, view_model, ScipyOptimizer

from aerostructures import NastranStatic, DisplacementTransfer, Panair, LoadTransfer, Interpolation, PanairMesher, StructureMesher, StaticStructureProblemDimensions, AeroProblemDimensions, StaticStructureProblemParams, AeroProblemParams, NLGaussSeidel, WingSegmentProps, WaveDrag, XLeadingEdge, Filter

import time 

if __name__ == "__main__":

    #Interpolation function type and setup
    function_type = 'thin_plate'
    bias_morph = (1.,1.,1.)
    bias_inter = (1.,1.,1.)

    #Symmetry plane index
    sym_plane_index = 2
    
    #Cases
    case_name = 'alpha_low'
    case_name_h = 'alpha_high'    

    #Number of wing sections
    n_sec = 5

    #Airfoil files list
    ref_airfoil_files = []
    for i in range(n_sec):
        ref_airfoil_files.append('airfoil_'+str(i+1)+'.txt')

    #Problem parameters
    #Speed of sound
    a = 295.1

    #Wing area of the baseline design (for normalization purposes only)
    S_ref = 313.18
    Mach = 0.78
    V = Mach*a
    rho_a = 0.294
    # h = 10500.
    alpha_0 = 2.
    b_0 = 41.15
    b_baseline = 41.15
    c = 7.61
    E = 7.17e10
    nu = 0.33
    rho_s = 2810.
    #Reference aircraft weight (mass units)
    W_ref = 80200.
    #Airframe weight (complete aircraft including payload and reserve fuel, excluding wing structure and fuel burn, mass units, from Kenway et al., 2014)
    W_airframe = 43200.
    #Yield stress (can also be used as ultimate stress if FS = 1.5)
    sigma_y = 572.e6
    #Factor of safety
    FS = 1.5
    #Cruise load factor
    n = 1.
    #Equivalent friction coefficient
    Cfe = 0.003
    #Technology factor for critical Mach number
    k = 0.9
    #Thrust-specific fuel consumption [kg/sN]
    SFC = 1.5013e-5
    #Design range [m]
    R = 5093000.
    #Minimum cabin height [m]
    h_min = 4.
    #Minimum cabin area [m2]
    area_min = 120.
    
    #Aerodynamic template files for both fidelities
    #Hi-Fi
    aero_template_l = 'aero_template_l.wgs'
    #Lo-Fi
    aero_template_h = 'aero_template_h.wgs'
    # Multi-fidelity options 'low', for low-fidelity; 'high', for high-fidelity; 'multi', for multi-fidelity
    fidelity = input('Please enter the fidelity level: low, high or multi: ')

    #Sectional properties (that are not design variables)
    y_le_baseline = np.array([0., 3.68, 6.2652, 9.03685, 20.57685])
    z_le = np.array([0., 0., 0., 0., 0.,])
    chords_0 = np.array([20., 16.91, 11.5, 4.3, 0.91])
    tc_0 = np.array([0.18, 0.19, 0.2, 0.14, 0.1])
    camc = np.array([0.02, 0.01, 0., 0.015, 0.011])

    structure_problem_dimensions = StaticStructureProblemDimensions()

    ns = structure_problem_dimensions.ns
    ns_all = structure_problem_dimensions.ns_all
    node_id = structure_problem_dimensions.node_id
    node_id_all = structure_problem_dimensions.node_id_all
    n_stress = structure_problem_dimensions.n_stress
    tn = structure_problem_dimensions.tn
    mn = 0
    sn = 0
    an = 0

    #Low fidelity instance -- aero_template_l.wgs
    aero_problem_dimensions = AeroProblemDimensions(aero_template_l)                                         
    na = aero_problem_dimensions.na
    na_unique = aero_problem_dimensions.na_unique
    network_info = aero_problem_dimensions.network_info

    #High fidelity instance -- aero_template_h.wgs
    aero_problem_dimensions_h = AeroProblemDimensions(aero_template_h)
    na_h = aero_problem_dimensions_h.na
    na_unique_h = aero_problem_dimensions_h.na_unique
    network_info_h = aero_problem_dimensions_h.network_info

    structure_problem_params = StaticStructureProblemParams(node_id, node_id_all)
    
    #Low fidelity instance -- aero_template_l.wgs
    aero_problem_params = AeroProblemParams(aero_template_l)
    
    #High fidelity instance -- aero_template_h.wgs
    aero_problem_params_h = AeroProblemParams(aero_template_h)

    #Design variable initial values (and other parameters)
    t_0 = 0.05*np.ones(15)

    theta_0 = np.zeros(5)

    # theta_0 = np.ones(n_sec)

    sweep_0 = np.array([26.62107398, 50.40477233, 64.93032268, 26.74549099])

    #X-position of the leading edge at the root
    xr = 50.

    #Design variable boundaries
    t_max = 2.*t_0
    t_min = 0.1*t_0 #0.05
    
    chords_max = 1.5*chords_0
    chords_min = 0.5*chords_0

    sweep_max = 75.*np.ones(4)
    sweep_min = np.zeros(4)

    b_max = 60.
    b_min = 30.

    alpha_max = 5.
    alpha_min = 0.

    theta_max = 5.*np.ones(len(theta_0))
    theta_min = -5.*np.ones(len(theta_0))

    tc_max = 0.25*np.ones(len(tc_0))
    tc_min = 0.075*np.ones(len(tc_0))

    #Coordinates of aerodynamic and structure matching meshes
    xa_b = aero_problem_params.apoints_coord_unique
    xa_b_h = aero_problem_params_h.apoints_coord_unique
    xs_b = structure_problem_params.node_coord_all

    top = Problem()
    top.root = root = Group()

    #Add independent variables (parameters)
    root.add('Airspeed', IndepVarComp('V', V), promotes=['*'])
    root.add('air_density', IndepVarComp('rho_a', rho_a), promotes=['*'])
    root.add('Mach_number', IndepVarComp('Mach', Mach), promotes=['*'])
    root.add('baseline_wing_span', IndepVarComp('b_baseline', b_baseline), promotes=['*'])
    root.add('wing_chord', IndepVarComp('c', c), promotes=['*'])
    root.add('Youngs_modulus', IndepVarComp('E', E), promotes=['*'])
    root.add('Poissons_ratio', IndepVarComp('nu', nu), promotes=['*'])
    root.add('material_density', IndepVarComp('rho_s', rho_s), promotes=['*'])
    root.add('airframe_mass', IndepVarComp('W_airframe', W_airframe), promotes=['*'])
    root.add('Tensile_Yield_Strength', IndepVarComp('sigma_y', sigma_y), promotes=['*'])
    root.add('factor_safety', IndepVarComp('FS', FS), promotes=['*'])
    root.add('y_leading_edge_baseline', IndepVarComp('y_le_baseline', y_le_baseline), promotes=['*'])
    root.add('z_leading_edge', IndepVarComp('z_le', z_le), promotes=['*'])
    root.add('camber_chord_ratio', IndepVarComp('camc', camc), promotes=['*'])
    root.add('base_aerodynamic_mesh', IndepVarComp('xa_b', xa_b))
    root.add('base_aerodynamic_mesh_h', IndepVarComp('xa_b', xa_b_h))
    root.add('base_structure_mesh', IndepVarComp('xs_b', xs_b), promotes=['*'])
    root.add('cruise_load_factor', IndepVarComp('n', n), promotes=['*'])
    root.add('root_leading_edge_x', IndepVarComp('xr', xr), promotes=['*'])
    root.add('equivalent_friction_coefficient', IndepVarComp('Cfe', Cfe), promotes=['*'])
    root.add('critical_mach_factor', IndepVarComp('k', k), promotes=['*'])
    root.add('specific_fuel_consumption', IndepVarComp('SFC', SFC), promotes=['*'])
    root.add('design_range', IndepVarComp('R', R), promotes=['*'])

    # Independent variables that are optimization design variables
    root.add('thicknesses', IndepVarComp('t', t_0), promotes=['*'])
    root.add('local_chords', IndepVarComp('chords', chords_0), promotes=['*'])
    root.add('local_sweep_angles', IndepVarComp('sweep', sweep_0), promotes=['*'])
    root.add('wing_span', IndepVarComp('b', b_0), promotes=['*'])
    root.add('wing_twist', IndepVarComp('theta', theta_0), promotes=['*'])
    root.add('angle_of_attack', IndepVarComp('alpha', alpha_0), promotes=['*'])
    root.add('airfoil_tc_ratio', IndepVarComp('tc', tc_0), promotes=['*'])
    root.add('minimum_cabin_height', IndepVarComp('h_min', h_min), promotes=['*'])
    root.add('minimum_cabin_area', IndepVarComp('area_min', area_min), promotes=['*'])

    #Interpolation Components
    root.add('interp_struct_morph', Interpolation(ns_all, na_unique_h, function = function_type, bias = bias_morph))
    
    #Geometry and meshing Components
    root.add('aerodynamic_mesher', PanairMesher(n_sec, na, na_unique, network_info, ref_airfoil_files), promotes=['camc','chords','tc','theta','x_le','y_le','z_le'])
    root.add('aerodynamic_mesher_h', PanairMesher(n_sec, na_h, na_unique_h, network_info_h, ref_airfoil_files), promotes=['camc','chords','tc','theta','x_le','y_le','z_le','apoints_coord','apoints_coord_unique'])
    root.add('structure_mesher', StructureMesher(na_unique_h, node_id, node_id_all), promotes=['*'])
    root.add('x_leading_edge', XLeadingEdge(n_sec), promotes=['*'])

    root.add('y_leading_edge', ExecComp(
        'y_le = b/b_baseline*y_le_baseline', y_le=np.zeros(len(y_le_baseline), dtype=float), y_le_baseline=np.zeros(len(y_le_baseline), dtype=float)), promotes=['*'])

    #Planform area
    root.add('reference_surface',
             ExecComp('Sw = 2.*area_segment.sum()', area_segment=np.zeros(n_sec-1, dtype=float)), promotes=['*'])
    
    #Total wet surface
    root.add('wet_surface', ExecComp(
        'S_wet = 2*Sw'), promotes=['*'])

    root.add('zero_lift_CD', ExecComp('CD0 = Cfe*S_wet/Sw'), promotes=['*'])

    root.add('wing_segment_properties', WingSegmentProps(n_sec), promotes=['*'])

    #Weighted average thickness-to-chord ratio
    root.add('average_tc_ratio',
             ExecComp('tc_avg = (area_segment*tc_segment).sum()/(area_segment).sum()', area_segment=np.zeros(n_sec-1, dtype=float), tc_segment=np.zeros(n_sec-1, dtype=float)), promotes=['*'])

    #Weighted average quarter-chord sweep
    root.add('average_quarter_chord_sweep',
             ExecComp('sweep_avg = (area_segment*sweep_segment).sum()/(area_segment).sum()', area_segment=np.zeros(n_sec-1, dtype=float), sweep_segment=np.zeros(n_sec-1, dtype=float)), promotes=['*'])

    #Critical Mach number for wave drag
    root.add('critical_mach', ExecComp(
        'Mcr = k/cos(radians(sweep_avg)) - tc_avg/cos(radians(sweep_avg))**2 - CL/(10.*cos(radians(sweep_avg))**3) - (0.1/80.)**(1./3.)'), promotes=['*'])

    root.add('wave_drag_coefficient', WaveDrag(), promotes=['*'])

    root.add('drag_coefficient', ExecComp('CD = CD0 + CDi + CDw'), promotes=['*'])

    #Drag force
    root.add('drag_force', ExecComp('D = 0.5*rho_a*V**2*Sw*CD'), promotes=['*'])

    #Fuel burn (Breguet equation)
    root.add('fuel_burn', ExecComp('FB = (W_airframe+2.*1.25*mass)*(exp(R*9.81*SFC/(V*(CL/CD))) - 1)'), promotes=['*'])

    # Aeroelastic MDA components
    mda_l = Group()
    
    #Add disciplines to the low fidelity group 
    mda_l.add('mult_filter_l', Filter(ns, fidelity)) #This component allows to recover result from HiFi 
    mda_l.add('displacement_transfer', DisplacementTransfer(na, ns)) 
    mda_l.add('aerodynamics', Panair(na, network_info, case_name, aero_template_l, sym_plane_index=sym_plane_index), promotes=['V','Sw','alpha','rho_a']) 
    mda_l.add('load_transfer', LoadTransfer(na, ns))
    mda_l.add('structures', NastranStatic(node_id, node_id_all, n_stress, tn, mn, sn, case_name, an=an), promotes=['n','m','t','s','Ix','Iy','a'])
    
    #Inner interpolation methods 
    mda_l.add('inter', Interpolation(na, ns, function = function_type, bias = bias_inter), promotes=['node_coord'])
    
    #Hi-Fi Group
    mda_h = Group()
    
    #Add disciplines to the high-fidelity group 
    mda_h.add('mult_filter_h', Filter(ns, fidelity))
    mda_h.add('displacement_transfer_h', DisplacementTransfer(na_h, ns))
    mda_h.add('aerodynamics_h', Panair(na_h, network_info_h, case_name_h, aero_template_h, sym_plane_index=sym_plane_index), promotes=['V','Sw','alpha','rho_a','CL','CDi','apoints_coord'])    
    mda_h.add('load_transfer_h', LoadTransfer(na_h, ns))
    mda_h.add('structures_h', NastranStatic(node_id, node_id_all, n_stress, tn, mn, sn, case_name_h, an=an), promotes=['mass','VMStress','n','m','t','s','Ix','Iy','node_coord_all','a'])
    
    #Inner interpolation method
    mda_h.add('inter_h', Interpolation(na_h, ns, function = function_type, bias = bias_inter), promotes=['apoints_coord','node_coord'])

    #Define solver type and tolerance for MDA Lo-Fi
    mda_l.nl_solver = NLGaussSeidel()
    #The solver execution limit is used to control fidelity levels
    if fidelity == 'high':
        mda_l.nl_solver.options['maxiter'] = 0 #No Lo-Fi iterations
              
    mda_l.nl_solver.options['rutol'] = 1.e-1 
    mda_l.nl_solver.options['use_aitken'] = True
    mda_l.nl_solver.options['aitken_alpha_min'] = 0.1
    mda_l.nl_solver.options['aitken_alpha_max'] = 1.5

    mda_l.ln_solver = ScipyGMRES()
    
    #Define solver type and tolerance for MDA Hi-Fi
    mda_h.nl_solver = NLGaussSeidel()
    #The solver execution limit is used to control fidelity levels
    if fidelity == 'low':
        mda_h.nl_solver.options['maxiter'] = 0
        
    mda_h.nl_solver.options['rutol'] = 1.e-1
    mda_h.nl_solver.options['use_aitken'] = True
    mda_h.nl_solver.options['aitken_alpha_min'] = 0.1
    mda_h.nl_solver.options['aitken_alpha_max'] = 1.5

    mda_h.ln_solver = ScipyGMRES()

    root.add('mda_group_l', mda_l, promotes=['*'])

    #Explicit connection Lo-Fi
    root.mda_group_l.connect('displacement_transfer.delta','aerodynamics.delta')
    root.mda_group_l.connect('inter.H','displacement_transfer.H')
    root.mda_group_l.connect('mult_filter_l.us','displacement_transfer.u')
    root.mda_group_l.connect('aerodynamics.f_a','load_transfer.f_a')
    root.mda_group_l.connect('load_transfer.f_node','structures.f_node')
    root.mda_group_l.connect('inter.H','load_transfer.H')
    root.mda_group_l.connect('structures.u','mult_filter_l.u')
    root.mda_group_l.connect('aerodynamics.apoints_coord','inter.apoints_coord')
    root.connect('aerodynamic_mesher.apoints_coord', 'aerodynamics.apoints_coord')
    root.connect('aerodynamic_mesher.apoints_coord','inter.apoints_coord')
    #Connect Indep Variables
    root.connect('Mach', 'aerodynamics.Mach')
    root.connect('b_baseline', 'aerodynamics.b')
    root.connect('c', 'aerodynamics.c')
    root.connect('nu', 'structures.nu')
    root.connect('E', 'structures.E')
    root.connect('rho_s', 'structures.rho_s')
    root.connect('xs_b', 'structures.node_coord_all')
    
    root.add('mda_group_h', mda_h, promotes=['*'])
    
    #Explicit connection Hi-Fi
    root.mda_group_h.connect('displacement_transfer_h.delta','aerodynamics_h.delta')
    root.mda_group_h.connect('inter_h.H','displacement_transfer_h.H')
    root.mda_group_h.connect('mult_filter_h.us','displacement_transfer_h.u')
    root.mda_group_h.connect('aerodynamics_h.f_a','load_transfer_h.f_a')
    root.mda_group_h.connect('load_transfer_h.f_node','structures_h.f_node')
    root.mda_group_h.connect('inter_h.H','load_transfer_h.H')
    root.mda_group_h.connect('structures_h.u','mult_filter_h.u')
    
    #This order guarantees that the filters are always privileged in the computation
    root.mda_group_l.set_order(['mult_filter_l', 'inter', 'displacement_transfer', 'aerodynamics', 'load_transfer','structures'])
    root.mda_group_h.set_order(['mult_filter_h', 'inter_h', 'displacement_transfer_h', 'aerodynamics_h', 'load_transfer_h', 'structures_h'])
    #Connect Indep Variables
    root.connect('Mach', 'aerodynamics_h.Mach')
    root.connect('b_baseline', 'aerodynamics_h.b')
    root.connect('c', 'aerodynamics_h.c')
    root.connect('nu', 'structures_h.nu')
    root.connect('E', 'structures_h.E')
    root.connect('rho_s', 'structures_h.rho_s')
        
    #Multifidelity explicit connections
    
    root.connect('structures.u', 'mult_filter_h.ul')
    root.connect('structures_h.u', 'mult_filter_l.ul')
    
    # #Recorder Lo-Fi
    # recorder_l = SqliteRecorder('mda_l.sqlite3')
    # recorder_l.options['record_metadata'] = False
    # #Recorder Hi-Fi
    # recorder_h = SqliteRecorder('mda_h.sqlite3')
    # recorder_h.options['record_metadata'] = False
    # top.root.mda_group_l.nl_solver.add_recorder(recorder_l)
    # top.root.mda_group_h.nl_solver.add_recorder(recorder_h)

    #Constraint components
    #Lift coefficient constraints (two constraints with same value to treat equality constraint as two inequality constraints)
    root.add('con_lift_cruise_upper', ExecComp(
        'con_l_u = CL - n*(W_airframe+2.*1.25*mass+0.2*FB)*9.81/(0.5*rho_a*V**2*Sw)'), promotes=['*'])
    root.add('con_lift_cruise_lower', ExecComp(
        'con_l_l = CL - n*(W_airframe+2.*1.25*mass+0.2*FB)*9.81/(0.5*rho_a*V**2*Sw)'), promotes=['*'])

    #Maximum stress constraint (considering factor of safety)
    root.add('con_stress', ExecComp('con_s = (W_airframe+2.*1.25*mass+FB)/(W_airframe+2.*1.25*mass+0.2*FB)*FS*2.5*max(VMStress) - sigma_y',
                                    VMStress=np.zeros(n_stress, dtype=float)), promotes=['*'])

    #Stress constraints (considering max load factor and factor of safety)
    for i in range(n_stress):
        root.add('con_stress_'+str(i+1), ExecComp('con_s_'+str(i+1)+' = (W_airframe+2.*1.25*mass+FB)/(W_airframe+2.*1.25*mass+0.2*FB)*FS*2.5*VMStress['+str(
            i)+'] - sigma_y', VMStress=np.zeros(n_stress, dtype=float)), promotes=['*'])

    #Minimum cabin height constraint
    for i in range(3):
        root.add('con_height_'+str(i+1), ExecComp('con_h_'+str(i+1)+' = 0.9*tc['+str(i)+']*chords['+str(i)+'] - h_min', tc=np.zeros(n_sec, dtype=float), chords=np.zeros(n_sec, dtype=float)), promotes=['*'])

    root.add('con_cabin_area', ExecComp('con_area = 2.*area_segment[:2].sum() - area_min', area_segment=np.zeros(n_sec-1, dtype=float)), promotes=['*'])
    
    #Add design variable bounds as constraints (COBYLA does not support design variable bounds)
    for i in range(tn):
        root.add('t_lower_bound_'+str(i+1), ExecComp('t_l_'+str(i+1) +
                                                     ' = t['+str(i)+']', t=np.zeros(tn, dtype=float)), promotes=['*'])
        root.add('t_upper_bound_'+str(i+1), ExecComp('t_u_'+str(i+1) +
                                                     ' = t['+str(i)+']', t=np.zeros(tn, dtype=float)), promotes=['*'])

    root.add('b_lower_bound', ExecComp('b_l = b'), promotes=['*'])
    root.add('b_upper_bound', ExecComp('b_u = b'), promotes=['*'])

    root.add('alpha_lower_bound', ExecComp('alpha_l = alpha'), promotes=['*'])
    root.add('alpha_upper_bound', ExecComp('alpha_u = alpha'), promotes=['*'])

    for i in range(n_sec):
        root.add('theta_lower_bound_'+str(i+1), ExecComp('theta_l_'+str(i+1) +
                                                         ' = theta['+str(i)+']', theta=np.zeros(n_sec, dtype=float)), promotes=['*'])
        root.add('theta_upper_bound_'+str(i+1), ExecComp('theta_u_'+str(i+1) +
                                                         ' = theta['+str(i)+']', theta=np.zeros(n_sec, dtype=float)), promotes=['*'])
        root.add('tc_lower_bound_'+str(i+1), ExecComp('tc_l_'+str(i+1) +
                                                      ' = tc['+str(i)+']', tc=np.zeros(n_sec, dtype=float)), promotes=['*'])
        root.add('tc_upper_bound_'+str(i+1), ExecComp('tc_u_'+str(i+1) +
                                                      ' = tc['+str(i)+']', tc=np.zeros(n_sec, dtype=float)), promotes=['*'])
        root.add('chords_lower_bound_'+str(i+1), ExecComp('chords_l_'+str(i+1) +
                                                          ' = chords['+str(i)+']', chords=np.zeros(n_sec, dtype=float)), promotes=['*'])
        root.add('chords_upper_bound_'+str(i+1), ExecComp('chords_u_'+str(i+1) +
                                                          ' = chords['+str(i)+']', chords=np.zeros(n_sec, dtype=float)), promotes=['*'])
    
    for i in range(n_sec-1):
        root.add('sweep_lower_bound_'+str(i+1), ExecComp('sweep_l_'+str(i+1) +
                                                         ' = sweep['+str(i)+']', sweep=np.zeros(n_sec-1, dtype=float)), promotes=['*'])
        root.add('sweep_upper_bound_'+str(i+1), ExecComp('sweep_u_'+str(i+1) +
                                                         ' = sweep['+str(i)+']', sweep=np.zeros(n_sec-1, dtype=float)), promotes=['*'])

    #Explicit connections
    root.connect('interp_struct_morph.H', 'G')
    root.connect('base_aerodynamic_mesh_h.xa_b', 'interp_struct_morph.node_coord')
    root.connect('xs_b', 'interp_struct_morph.apoints_coord')

    #Define the optimizer (Scipy)
    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'COBYLA'
    top.driver.options['disp'] = True
    top.driver.options['tol'] = 1.e-3
    top.driver.options['maxiter'] = 500
    top.driver.opt_settings['rhobeg'] = 0.7

    top.driver.add_desvar('t', lower=t_min, upper=t_max,
                          adder=-t_min, scaler=1./(t_max-t_min))
    
    top.driver.add_desvar('chords', lower=chords_min, upper=chords_max,
                          adder=-chords_min, scaler=1./(chords_max-chords_min))
    top.driver.add_desvar('sweep', lower=sweep_min, upper=sweep_max,
                          adder=-sweep_min, scaler=1./(sweep_max-sweep_min))
    top.driver.add_desvar('b', lower=b_min, upper=b_max,
                          adder=-b_min, scaler=1./(b_max-b_min))
    top.driver.add_desvar('alpha', lower=alpha_min, upper=alpha_max,
                          adder=-alpha_min, scaler=1./(alpha_max-alpha_min))
    top.driver.add_desvar('theta', lower=theta_min, upper=theta_max,
                          adder=-theta_min, scaler=1./(theta_max-theta_min))
    top.driver.add_desvar('tc', lower=tc_min, upper=tc_max,
                          adder=-tc_min, scaler=1./(tc_max-tc_min))

    top.driver.add_objective('FB')

    for i in range(n_stress):
        top.driver.add_constraint('con_s_'+str(i+1), upper=0., scaler=1./sigma_y)

    # top.driver.add_constraint('con_s', upper=0., scaler=1./sigma_y)

    top.driver.add_constraint(
        'con_l_u', upper=0., scaler=1./(n*W_ref*9.81/(0.5*rho_a*V**2*S_ref)))
    top.driver.add_constraint(
        'con_l_l', lower=0., scaler=1./(n*W_ref*9.81/(0.5*rho_a*V**2*S_ref)))

    #Ensure positive FB
    top.driver.add_constraint('FB', lower=0.)

    # Add design variable bounds constraints to the driver
    for i in range(tn):
        top.driver.add_constraint('t_l_'+str(i+1), lower=t_min[i], scaler=1./t_0[i])
        top.driver.add_constraint('t_u_'+str(i+1), upper=t_max[i], scaler=1./t_0[i])

    top.driver.add_constraint('b_l', lower=b_min, scaler=1./b_0)
    top.driver.add_constraint('b_u', upper=b_max, scaler=1./b_0)

    top.driver.add_constraint('alpha_l', lower=alpha_min, scaler=1./alpha_0)
    top.driver.add_constraint('alpha_u', upper=alpha_max, scaler=1./alpha_0)

    for i in range(n_sec):
        top.driver.add_constraint(
            'theta_l_'+str(i+1), lower=theta_min[i], scaler=1.)
        top.driver.add_constraint(
            'theta_u_'+str(i+1), upper=theta_max[i], scaler=1.)
        top.driver.add_constraint(
            'tc_l_'+str(i+1), lower=tc_min[i], scaler=1.)
        top.driver.add_constraint(
            'tc_u_'+str(i+1), upper=tc_max[i], scaler=1.)
        top.driver.add_constraint(
            'chords_l_'+str(i+1), lower=chords_min[i], scaler=1.)
        top.driver.add_constraint(
            'chords_u_'+str(i+1), upper=chords_max[i], scaler=1.)
    
    for i in range(n_sec-1):
        top.driver.add_constraint(
            'sweep_l_'+str(i+1), lower=sweep_min[i], scaler=1.)
        top.driver.add_constraint(
            'sweep_u_'+str(i+1), upper=sweep_max[i], scaler=1.)

    recorder = SqliteRecorder('mdao.sqlite3')
    recorder.options['record_metadata'] = False
    recorder.options['includes'] = ['CDi', 'con_l_u', 'con_s', 't', 'a', 'chords', 'sweep', 'b', 'alpha', 'theta', 'tc', 'CD0', 'CDw', 'D', 'FB', 'con_area', 'con_h_1', 'con_h_2', 'con_h_3']
    
    top.driver.add_recorder(recorder)

    #Define solver type
    root.ln_solver = ScipyGMRES()

    start1 = time.time() #timer for set-up and re-order
    top.setup()
    order = root.list_auto_order() #This is to ensure that the mda_l group is executed always before the mda_h group
    a, b = order[0].index('mda_group_h'), order[0].index('mda_group_l')
    order[0].insert(a, order[0].pop(b))
    root.set_order(order[0])
    end1 = time.time()
    view_model(top, show_browser=False) #generates an N2 diagram to visualize connections

    #Setting initial values for design variables
    top['t'] = t_0
    top['chords'] = chords_0
    top['sweep'] = sweep_0
    top['b'] = b_0
    top['alpha'] = alpha_0
    top['theta'] = theta_0
    top['tc'] = tc_0

    start2 = time.time()
    top.run()
    end2 = time.time()
    top.cleanup()  # this closes all recorders
    print("Set up time = " + str(end1 - start1))
    print("Run time = " + str(end2 - start2))
