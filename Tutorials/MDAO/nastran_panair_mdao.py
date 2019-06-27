# -*- coding: utf-8 -*-
"""
Created on Mon Jul 2 2018

@author: © Joan Mas Colomer
"""

from __future__ import print_function

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyGMRES, SqliteRecorder, view_model, ScipyOptimizer

from aerostructures import NastranStatic, DisplacementTransfer, Panair, LoadTransfer, Interpolation, PanairMesher, StructureMesher, PlanformGeometry, StaticStructureProblemDimensions, AeroProblemDimensions, StaticStructureProblemParams, AeroProblemParams, NLGaussSeidel

if __name__ == "__main__":

    #Interpolation function type and setup
    function_type = 'thin_plate'
    bias_morph = (1.,1.,1.)
    bias_inter = (1.,100.,1.)

    #Symmetry plane index
    sym_plane_index = 2

    #Number of wing sections
    n_sec = 8

    #Position (index) of the wing break
    b_sec = 4

    #Airfoil file
    ref_airfoil_file = 'crm.eta65.unswept31.5deg.sharp.te.txt'

    #Problem parameters
    #Speed of sound
    a = 297.4

    Sw = 383.689555
    V = 252.16168
    Mach = V/a
    rho_a = 0.38058496
    # h = 10500.
    alpha_0 = 1.340
    b_0 = 58.7629
    b_baseline = 58.7629
    c = 7.00532
    E = 6.89e10
    nu = 0.31
    rho_s = 2795.67
    #Reference aircraft weight (mass units)
    W_ref = 226796.185
    #Wing weight (full span) of the reference aircraft (mass units)
    W_ref_wing = 26400.
    #Airframe weight (complete aircraft excluding wing structure, mass units)
    W_airframe = W_ref - W_ref_wing
    #Yield stress (can also be used as ultimate stress if FS = 1.5)
    sigma_y = 450.e6
    #Factor of safety
    FS = 1.
    #Cruise load factor
    n = 1.

    #Sectional properties (that are not design variables)
    y_le_baseline = np.array([0., 2.938145, 7.3453752, 10.8711746, 16.1598356, 20.5670658, 24.974296, 29.3815262])
    z_le = np.array([4.424397971, 4.44511389, 4.476187859, 4.501047142, 4.538335797, 4.569409766, 4.600483735, 4.631557704])
    c_0 = np.array([13.6189974, 11.9001794, 9.3216984,
                    7.2588628, 5.9643264, 4.8855376, 3.8067488, 2.72796])
    tc_0 = np.array([0.1542, 0.138, 0.1137, 0.1052, 0.0988, 0.0962, 0.0953, 0.095])
    th = tc_0*c_0
    camc = np.array([0.0003, 0.0012, 0.0037, 0.0095, 0.0146, 0.0158, 0.0161, 0.0009])

    structure_problem_dimensions = StaticStructureProblemDimensions()

    ns = structure_problem_dimensions.ns
    ns_all = structure_problem_dimensions.ns_all
    node_id = structure_problem_dimensions.node_id
    node_id_all = structure_problem_dimensions.node_id_all
    n_stress = structure_problem_dimensions.n_stress
    tn = structure_problem_dimensions.tn
    #Choose 4 mass design variables
    mn = 0
    sn = 0
    an = structure_problem_dimensions.an

    aero_problem_dimensions = AeroProblemDimensions()

    na = aero_problem_dimensions.na
    na_unique = aero_problem_dimensions.na_unique
    network_info = aero_problem_dimensions.network_info

    structure_problem_params = StaticStructureProblemParams(node_id, node_id_all)
    aero_problem_params = AeroProblemParams()

    #Design variable initial values (and other parameters)
    t_0 = np.array([.00635, .005334, .004572, .003302, .00254,
                  .001651, .01905, .01524, .0127, .009525, .00508, .00254])

    a_0 = np.array([0.0066339, 0.0048852, 0.0034935,
                  0.0021121, 9.14E-04, 3.74E-04])

    theta = np.array([6.691738003, 4.545042708, 2.793550837, 1.673916686,
                      0.754303126, 0.91369482, 1.136056807, 0.272576679])

    cr_0 = 13.6189974

    cb_0 = 7.2588628

    ct_0 = 2.72796

    sweep_0 = 37.16

    #X-position of the leading edge at the root
    xr = 22.9690676

    #Design variable boundaries
    t_max = 3*t_0
    t_min = 0.25*t_0

    a_max = 3*a_0
    a_min = 0.25*a_0

    cr_max = 1.5*cr_0
    cr_min = 0.75*cr_0

    cb_max = 1.5*cb_0
    cb_min = 0.75*cb_0

    ct_max = 1.5*ct_0
    ct_min = 0.75*ct_0

    sweep_max = 50.
    sweep_min = 30.

    b_max = 80.
    b_min = 40.

    alpha_max = 5.
    alpha_min = -2.

    #Coordinates of aerodynamic and structure matching meshes
    xa_b = aero_problem_params.apoints_coord_unique
    xs_b = structure_problem_params.node_coord_all

    top = Problem()
    top.root = root = Group()

    #Add independent variables (parameters)
    root.add('wing_area', IndepVarComp('Sw', Sw), promotes=['*'])
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
    root.add('airfoil_thickness', IndepVarComp('th', th), promotes=['*'])
    root.add('camber_chord_ratio', IndepVarComp('camc', camc), promotes=['*'])
    root.add('base_aerodynamic_mesh', IndepVarComp('xa_b', xa_b), promotes=['*'])
    root.add('base_structure_mesh', IndepVarComp('xs_b', xs_b), promotes=['*'])
    root.add('cruise_load_factor', IndepVarComp('n', n), promotes=['*'])
    root.add('root_leading_edge_x', IndepVarComp('xr', xr), promotes=['*'])
    root.add('wing_twist', IndepVarComp('theta', theta), promotes=['*'])

    # Independent variables that are optimization design variables
    root.add('thicknesses', IndepVarComp('t', t_0), promotes=['*'])
    root.add('rod_sections', IndepVarComp('a', a_0), promotes=['*'])
    root.add('root_chord', IndepVarComp('cr', cr_0), promotes=['*'])
    root.add('break_chord', IndepVarComp('cb', cb_0), promotes=['*'])
    root.add('tip_chord', IndepVarComp('ct', ct_0), promotes=['*'])
    root.add('sweep_angle', IndepVarComp('sweep', sweep_0), promotes=['*'])
    root.add('wing_span', IndepVarComp('b', b_0), promotes=['*'])
    root.add('angle_of_attack', IndepVarComp('alpha', alpha_0), promotes=['*'])

    #Interpolation Components
    root.add('interp_struct_morph', Interpolation(ns_all, na_unique, function = function_type, bias = bias_morph))
    root.add('interp_mda', Interpolation(na, ns, function = function_type, bias = bias_inter), promotes=['*'])

    #Geometry and meshing Components
    root.add('planform_geometry', PlanformGeometry(n_sec, b_sec), promotes=['*'])
    root.add('aerodynamic_mesher', PanairMesher(n_sec, na, na_unique, network_info, ref_airfoil_file), promotes=['*'])
    root.add('structure_mesher', StructureMesher(na_unique, node_id, node_id_all), promotes=['*'])

    root.add('y_leading_edge', ExecComp(
        'y_le = b/b_baseline*y_le_baseline', y_le=np.zeros(len(y_le_baseline), dtype=float), y_le_baseline=np.zeros(len(y_le_baseline), dtype=float)), promotes=['*'])

    root.add('tc_ratio', ExecComp(
        'tc = th/chords', tc=np.zeros(n_sec, dtype=float), th=np.zeros(n_sec, dtype=float), chords=np.zeros(n_sec, dtype=float)), promotes=['*'])

    #Aeroelastic MDA components
    mda = Group()

    #Add disciplines to the group
    mda.add('aerodynamics', Panair(na, network_info, 'alpha', sym_plane_index), promotes=['*'])
    mda.add('load_transfer', LoadTransfer(na, ns), promotes=['*'])
    mda.add('structures', NastranStatic(node_id, node_id_all, n_stress, tn, mn, sn, 'alpha', an=an), promotes=['*'])
    mda.add('displacement_transfer', DisplacementTransfer(na, ns), promotes=['*'])

    #Define solver type and tolerance for MDA
    mda.nl_solver = NLGaussSeidel()
    # mda.nl_solver.options['maxiter'] = 0
    mda.nl_solver.options['rutol'] = 1.e-1
    mda.nl_solver.options['use_aitken'] = True
    mda.nl_solver.options['aitken_alpha_min'] = 0.1
    mda.nl_solver.options['aitken_alpha_max'] = 1.5

    mda.ln_solver = ScipyGMRES()

    root.add('mda_group', mda, promotes=['*'])

    #Constraint components
    #Lift coefficient constraints (two constraints with same value to treat equality constraint as two inequality constraints)
    root.add('con_lift_cruise_upper', ExecComp(
        'con_l_u = CL - n*(W_airframe+2*1.25*mass)*9.81/(0.5*rho_a*V**2*Sw)'), promotes=['*'])
    root.add('con_lift_cruise_lower', ExecComp(
        'con_l_l = CL - n*(W_airframe+2*1.25*mass)*9.81/(0.5*rho_a*V**2*Sw)'), promotes=['*'])

    #Maximum stress constraint (considering factor of safety)
    root.add('con_stress', ExecComp('con_s = FS*2.5*max(VMStress) - sigma_y', VMStress=np.zeros(n_stress,dtype=float)), promotes=['*'])

    #Stress constraints (considering max load factor and factor of safety)
    for i in range(n_stress):
        root.add('con_stress_'+str(i+1), ExecComp('con_s_'+str(i+1)+' = FS*2.5*VMStress['+str(
            i)+'] - sigma_y', VMStress=np.zeros(n_stress, dtype=float)), promotes=['*'])

    #Add design variable bounds as constraints (COBYLA does not support design variable bounds)
    for i in range(tn):
        root.add('t_lower_bound_'+str(i+1), ExecComp('t_l_'+str(i+1) +
                                                     ' = t['+str(i)+']', t=np.zeros(tn, dtype=float)), promotes=['*'])
        root.add('t_upper_bound_'+str(i+1), ExecComp('t_u_'+str(i+1) +
                                                     ' = t['+str(i)+']', t=np.zeros(tn, dtype=float)), promotes=['*'])
    for i in range(an):
        root.add('a_lower_bound_'+str(i+1), ExecComp('a_l_'+str(i+1) +
                                                     ' = a['+str(i)+']', a=np.zeros(an, dtype=float)), promotes=['*'])
        root.add('a_upper_bound_'+str(i+1), ExecComp('a_u_'+str(i+1) +
                                                     ' = a['+str(i)+']', a=np.zeros(an, dtype=float)), promotes=['*'])

    root.add('cr_lower_bound', ExecComp('cr_l = cr'), promotes=['*'])
    root.add('cr_upper_bound', ExecComp('cr_u = cr'), promotes=['*'])

    root.add('cb_lower_bound', ExecComp('cb_l = cb'), promotes=['*'])
    root.add('cb_upper_bound', ExecComp('cb_u = cb'), promotes=['*'])

    root.add('ct_lower_bound', ExecComp('ct_l = ct'), promotes=['*'])
    root.add('ct_upper_bound', ExecComp('ct_u = ct'), promotes=['*'])

    root.add('sweep_lower_bound', ExecComp('sweep_l = sweep'), promotes=['*'])
    root.add('sweep_upper_bound', ExecComp('sweep_u = sweep'), promotes=['*'])

    root.add('b_lower_bound', ExecComp('b_l = b'), promotes=['*'])
    root.add('b_upper_bound', ExecComp('b_u = b'), promotes=['*'])

    root.add('alpha_lower_bound', ExecComp('alpha_l = alpha'), promotes=['*'])
    root.add('alpha_upper_bound', ExecComp('alpha_u = alpha'), promotes=['*'])

    #Explicit connections
    root.connect('interp_struct_morph.H', 'G')
    root.connect('xa_b', 'interp_struct_morph.node_coord')
    root.connect('xs_b', 'interp_struct_morph.apoints_coord')

    #Define the optimizer (Scipy)
    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'COBYLA'
    top.driver.options['disp'] = True
    top.driver.options['tol'] = 1.e-3
    top.driver.options['maxiter'] = 500
    top.driver.opt_settings['rhobeg'] = 0.4

    top.driver.add_desvar('t', lower=t_min, upper=t_max,
                          adder=-t_min, scaler=1./(t_max-t_min))
    top.driver.add_desvar('a', lower=a_min, upper=a_max,
                          adder=-a_min, scaler=1./(a_max-a_min))
    top.driver.add_desvar('cr', lower=cr_min, upper=cr_max,
                          adder=-cr_min, scaler=1./(cr_max-cr_min))
    top.driver.add_desvar('cb', lower=cb_min, upper=cb_max,
                          adder=-cb_min, scaler=1./(cb_max-cb_min))
    top.driver.add_desvar('ct', lower=ct_min, upper=ct_max,
                          adder=-ct_min, scaler=1./(ct_max-ct_min))
    top.driver.add_desvar('sweep', lower=sweep_min, upper=sweep_max,
                          adder=-sweep_min, scaler=1./(sweep_max-sweep_min))
    top.driver.add_desvar('b', lower=b_min, upper=b_max,
                          adder=-b_min, scaler=1./(b_max-b_min))
    top.driver.add_desvar('alpha', lower=alpha_min, upper=alpha_max,
                          adder=-alpha_min, scaler=1./(alpha_max-alpha_min))

    top.driver.add_objective('CDi')

    for i in range(n_stress):
        top.driver.add_constraint('con_s_'+str(i+1), upper=0., scaler=1./sigma_y)

    # top.driver.add_constraint('con_s', upper=0., scaler=1./sigma_y)

    top.driver.add_constraint(
        'con_l_u', upper=0., scaler=1./(n*W_ref*9.81/(0.5*rho_a*V**2*Sw)))
    top.driver.add_constraint(
        'con_l_l', lower=0., scaler=1./(n*W_ref*9.81/(0.5*rho_a*V**2*Sw)))

    #Add design variable bounds constraints to the driver
    for i in range(tn):
        top.driver.add_constraint('t_l_'+str(i+1), lower=t_min[i], scaler=1./t_0[i])
        top.driver.add_constraint('t_u_'+str(i+1), upper=t_max[i], scaler=1./t_0[i])

    for i in range(an):
        top.driver.add_constraint('a_l_'+str(i+1), lower=a_min[i], scaler=1./a_0[i])
        top.driver.add_constraint('a_u_'+str(i+1), upper=a_max[i], scaler=1./a_0[i])

    top.driver.add_constraint('cr_l', lower=cr_min, scaler=1./cr_0)
    top.driver.add_constraint('cr_u', upper=cr_max, scaler=1./cr_0)

    top.driver.add_constraint('cb_l', lower=cb_min, scaler=1./cb_0)
    top.driver.add_constraint('cb_u', upper=cb_max, scaler=1./cb_0)

    top.driver.add_constraint('ct_l', lower=ct_min, scaler=1./ct_0)
    top.driver.add_constraint('ct_u', upper=ct_max, scaler=1./ct_0)

    top.driver.add_constraint('sweep_l', lower=sweep_min, scaler=1./sweep_0)
    top.driver.add_constraint('sweep_u', upper=sweep_max, scaler=1./sweep_0)

    top.driver.add_constraint('b_l', lower=b_min, scaler=1./b_0)
    top.driver.add_constraint('b_u', upper=b_max, scaler=1./b_0)

    top.driver.add_constraint('alpha_l', lower=alpha_min, scaler=1./alpha_0)
    top.driver.add_constraint('alpha_u', upper=alpha_max, scaler=1./alpha_0)

    recorder = SqliteRecorder('mdao.sqlite3')
    recorder.options['record_metadata'] = False
    recorder.options['includes'] = ['CDi', 'con_l_u', 'con_s', 't', 'a', 'cr',
                                    'cb', 'ct', 'sweep', 'b', 'alpha']
    
    top.driver.add_recorder(recorder)

    #Define solver type
    root.ln_solver = ScipyGMRES()

    top.setup()

    view_model(top, show_browser=False)

    #Setting initial values for design variables
    top['t'] = t_0
    top['a'] = a_0
    top['cr'] = cr_0
    top['cb'] = cb_0
    top['ct'] = ct_0
    top['sweep'] = sweep_0
    top['b'] = b_0
    top['alpha'] = alpha_0

    top.run()

    top.cleanup()  # this closes all recorders
