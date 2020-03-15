# -*- coding: utf-8 -*-
"""


@author: © Gilberto Ruiz 
"""

from __future__ import print_function
import numpy as np
from openmdao.api import Group, IndepVarComp, ExecComp, ScipyGMRES, SqliteRecorder, ScipyOptimizer, view_model, Problem
#from openmdao.api import NonlinearBlockGS, ScipyKrylov
import pywarpustruct
from aerostructures import NastranStatic, DisplacementTransfer, LoadTransfer, Interpolation, SurfaceGenerator, StructureMesher, AeroHiFiMesher, PlanformGeometry, StaticStructureProblemDimensions, AeroProblemDimensions, StaticStructureProblemParams, AeroProblemParams, NLGaussSeidel
from aerostructures import ADflow, Panair, Filter, PanairMesher
from bending_inertias import BendingInertias



import time

if __name__ == "__main__":

    #Interpolation function type and setup
    function_type = 'thin_plate'
    bias_morph = (1.,1.,1.)
    bias_inter = (1.,100.,1.)

    #Symmetry plane index
    sym_plane_index = 2

    #Number of wing sections
    n_sec = 8

    #Aerodynamic template 
    #Hi-Fi
    aero_template_l = 'aero_template_l.wgs'
    #Lo-Fi
    aero_template_h = 'aero_template_h.wgs'
    # Multi-fidelity options 'low', for low-fidelity; 'high', for high-fidelity; 'multi', for multi-fidelity
    #fidelity = raw_input('Please enter the fidelity level: low, high or multi: ') 
    fidelity = 'multi'

    
    #Position (index) of the wing break
    b_sec = 4

    #Airfoil file
    ref_airfoil_file = 'crm.eta65.unswept31.5deg.sharp.te.txt'

    #CGNS base mesh
    base_gridFile = 'ucrm_wing_definition_21_L2.cgns'
                 
    
    #Problem parameters
    #Speed of sound
    a = 297.4
    Sw = 383.689555
    V = 252.16168
    Mach = V/a
    rho_a = 0.38058496
    h = 10500.
    alpha = 1.340
    b_0 = 58.7629
    b_baseline = 58.7629
    c = 7.00532
    E = 6.89e10
    nu = 0.31
    rho_s = 2795.67
    h_b = 1.578
    t_b = 0.14557
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
    #y_le = b_0/b_baseline*np.array([0., 2.938145, 7.3453752, 10.8711746, 16.1598356, 20.5670658, 24.974296, 29.3815262])
    z_le = np.array([4.424397971, 4.44511389, 4.476187859, 4.501047142, 4.538335797, 4.569409766, 4.600483735, 4.631557704])
    tc = np.array([0.1542, 0.138, 0.1137, 0.1052, 0.0988, 0.0962, 0.0953, 0.095])
    camc = np.array([0.0003, 0.0012, 0.0037, 0.0095, 0.0146, 0.0158, 0.0161, 0.0009])                                                         
    
    structure_problem_dimensions = StaticStructureProblemDimensions()
    #High fidelity instance -- aero_template_h.wgs
    aero_problem_dimensions_h = AeroProblemDimensions(aero_template_h)                                                 

    ns = structure_problem_dimensions.ns
    ns_all = structure_problem_dimensions.ns_all
    node_id = structure_problem_dimensions.node_id
    node_id_all = structure_problem_dimensions.node_id_all
    n_stress = structure_problem_dimensions.n_stress
    tn = structure_problem_dimensions.tn
    u = np.zeros((ns, 3))
    ul = np.zeros((ns, 3)) #Auxiliary variable to transfer the displacement field between fidelities
    #Choose 4 mass design variables
    mn = 0
    sn = structure_problem_dimensions.sn
    ngeom = aero_problem_dimensions_h.na
    ngeom_unique = aero_problem_dimensions_h.na_unique
    network_info_h = aero_problem_dimensions_h.network_info
    an = structure_problem_dimensions.an                  

    case_name = 'alpha_low'
    case_name_h = 'alpha_high'

    #Low fidelity instance -- aero_template_l.wgs
    aero_problem_dimensions = AeroProblemDimensions(aero_template_l)                                         
    na = aero_problem_dimensions.na
    na_unique = aero_problem_dimensions.na_unique
    network_info = aero_problem_dimensions.network_info

    structure_problem_params = StaticStructureProblemParams(node_id, node_id_all)
    aeroOptions_base = {'gridFile':base_gridFile}
    
    outputDirectory = './'
    MGCycle = 'sg'
    
    aeroOptions = {
        # Common Parameters
        'gridFile':base_gridFile,
        'outputDirectory':outputDirectory,
    
        # Physics Parameters
        'equationType':'rans',
    
        # Common Parameters
        'CFL':1.5,
        'CFLCoarse':1.1,
        'MGCycle':MGCycle,
        'MGStartLevel':-1,
        'nCyclesCoarse':250,
        'nCycles':1000,
        'monitorvariables':['resrho','cl','cd'],
        'surfaceVariables':['cp','vx', 'vy', 'vz', 'mach'],
        'useNKSolver':True,
        'liftIndex': 3,
    
        # Convergence Parametersaerodynamics', A
        'L2Convergence':1e-6,
        'L2ConvergenceCoarse':1e-2,
    
        #Solver
        'useANKsolver':True,
        #Only used for viscous flow (turbulence)
        'nsubiterturb':5,
    
        # Multidisciplinary Coupling Parameters:
        'forcesastractions':False,
        }

    #Instantiate mesh component just to get mesh dimensions and base coordinates
    base_mesh = pywarpustruct.USMesh(options=aeroOptions_base)
    xa_b_h = base_mesh.getSurfaceCoordinates()
    na_h = len(xa_b_h)
    na_unique_h = len(np.unique(xa_b_h, axis=0))
    
    #Low fidelity instance -- aero_template_l.wgs
    aero_problem_params = AeroProblemParams(aero_template_l)
    
    #High fidelity instance -- aero_template_h.wgs
    aero_problem_params_h = AeroProblemParams(aero_template_h)

    #Design variable values
    t = np.array([.00635,
                .005334,
                .004572,
                .003302,
                .00254,
                .001651,
                .01905,
                .01524,
                .0127,
                .009525,
                .00508,
                .00254])
                
    s = np.array([.0001636,
                .0003403,
                .0003097,
                .0005161,
                .0011355,
                .0001064])
    
    theta = np.array([6.691738003,
                      4.545042708,
                      2.793550837,
                      1.673916686,
                      0.754303126,
                      0.91369482,
                      1.136056807,
                      0.272576679])                  

    cr = 13.6189974

    cb = 7.2588628

    ct = 2.72796

    sweep = 37.16

    #X-position of the leading edge at the root
    xr = 22.9690676

	#Coordinates of aerodynamic and structure matching meshes
    xgeom_b = aero_problem_params_h.apoints_coord_unique
    
    #Design variable boundaries
    t_max = 3*t
    t_min = 0.25*t

    s_max = 3*s
    s_min = 0.25*s

    cr_max = 1.5*cr
    cr_min = 0.75*cr

    cb_max = 1.5*cb
    cb_min = 0.75*cb

    ct_max = 1.5*ct
    ct_min = 0.75*ct

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
    root.add('altitude', IndepVarComp('h', h), promotes=['*'])
    root.add('Mach_number', IndepVarComp('Mach', Mach))
    root.add('angle_of_attack', IndepVarComp('alpha', alpha), promotes=['*'])
    root.add('wing_chord', IndepVarComp('c', c))
    root.add('Youngs_modulus', IndepVarComp('E', E))
    root.add('Poissons_ratio', IndepVarComp('nu', nu))
    root.add('material_density', IndepVarComp('rho_s', rho_s))
    root.add('h_b_ratio', IndepVarComp('h_b', h_b), promotes=['*'])
    root.add('t_b_ratio', IndepVarComp('t_b', t_b), promotes=['*'])
    root.add('airframe_mass', IndepVarComp('W_airframe', W_airframe), promotes=['*'])
    root.add('Tensile_Yield_Strength', IndepVarComp('sigma_y', sigma_y), promotes=['*'])
    root.add('factor_safety', IndepVarComp('FS', FS), promotes=['*'])
    root.add('y_leading_edge_baseline', IndepVarComp('y_le_baseline', y_le_baseline), promotes=['*'])
    #root.add('y_leading_edge', IndepVarComp('y_le', y_le), promotes=['*'])
    root.add('z_leading_edge', IndepVarComp('z_le', z_le), promotes=['*'])
    root.add('thickness_chord_ratio', IndepVarComp('tc', tc), promotes=['*'])
    root.add('camber_chord_ratio', IndepVarComp('camc', camc), promotes=['*'])
    root.add('base_aerodynamic_mesh', IndepVarComp('xa_b', xa_b))
    root.add('base_aerodynamic_mesh_h', IndepVarComp('xa_b', xa_b_h))
    root.add('base_geometric_mesh', IndepVarComp('xgeom_b', xgeom_b), promotes=['*'])
    root.add('base_structure_mesh', IndepVarComp('xs_b', xs_b), promotes=['*'])
    root.add('cruise_load_factor', IndepVarComp('n', n), promotes=['*'])

    root.add('thicknesses', IndepVarComp('t', t), promotes=['*'])
    root.add('sections', IndepVarComp('s', s), promotes=['*'])
    root.add('wing_twist', IndepVarComp('theta', theta), promotes=['*'])
    root.add('root_chord', IndepVarComp('cr', cr), promotes=['*'])
    root.add('break_chord', IndepVarComp('cb', cb), promotes=['*'])
    root.add('tip_chord', IndepVarComp('ct', ct), promotes=['*'])
    root.add('sweep_angle', IndepVarComp('sweep', sweep), promotes=['*'])
    root.add('root_leading_edge_x', IndepVarComp('xr', xr), promotes=['*'])                                                                         
    root.add('baseline_wing_span', IndepVarComp('b_baseline', b_baseline), promotes=['*'])
    # Independent variables that are optimization design variables
    root.add('wing_span', IndepVarComp('b', b_0), promotes=['*'])
    

    #Interpolation Components
    root.add('interp_struct_morph', Interpolation(ns_all, ngeom_unique, function = function_type, bias = bias_morph))
    root.add('interp_aero_hifi_morph', Interpolation(na_h, ngeom_unique, function = function_type, bias = bias_morph))
    
    
    #Geometry and meshing Components
    root.add('planform_geometry', PlanformGeometry(n_sec, b_sec), promotes=['*'])
    root.add('surface_generator', SurfaceGenerator(n_sec, ngeom, ngeom_unique, network_info_h, ref_airfoil_file), promotes=['*'])
    root.add('aero_hifi_mesher', AeroHiFiMesher(ngeom_unique, na_h, na_unique_h), promotes=['*'])
    root.add('aerodynamic_mesher', PanairMesher(n_sec, na, na_unique, network_info, ref_airfoil_file), promotes=['camc','chords','tc','theta','x_le','y_le','z_le'])
    root.add('bending_inertias', BendingInertias(sn), promotes=['*'])
    root.add('structure_mesher', StructureMesher(ngeom_unique, node_id, node_id_all), promotes=['*'])

    root.add('y_leading_edge', ExecComp(
        'y_le = b/b_baseline*y_le_baseline', y_le=np.zeros(len(y_le_baseline), dtype=float), y_le_baseline=np.zeros(len(y_le_baseline), dtype=float)), promotes=['*'])

    #root.add('tc_ratio', ExecComp(
        #'tc = th/chords', tc=np.zeros(n_sec, dtype=float), th=np.zeros(n_sec, dtype=float), chords=np.zeros(n_sec, dtype=float)), promotes=['*'])
    
    #Aeroelastic MDA components
       
    #Lo-Fi Group
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
    mda_h.add('aerodynamics_h', ADflow(na_h, aeroOptions, sym_plane_index), promotes=['*'])    
    mda_h.add('load_transfer_h', LoadTransfer(na_h, ns))
    mda_h.add('structures_h', NastranStatic(node_id, node_id_all, n_stress, tn, mn, sn, case_name_h, an=an), promotes=['mass','VMStress','n','m','t','s','Ix','Iy','node_coord_all','a'])
    
    #Inner interpolation method
    mda_h.add('inter_h', Interpolation(na_h, ns, function = function_type, bias = bias_inter), promotes=['jig_surface_coord','node_coord'])
                                                      
    
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
    root.mda_group_l.connect('aerodynamics.apoints_coord','inter.jig_surface_coord')
    root.connect('aerodynamic_mesher.apoints_coord', 'aerodynamics.apoints_coord')
    root.connect('aerodynamic_mesher.apoints_coord','inter.jig_surface_coord')
    #Connect Indep Variables
    root.connect('Mach_number.Mach', 'aerodynamics.Mach')
    root.connect('b_baseline', 'aerodynamics.b')
    root.connect('wing_chord.c', 'aerodynamics.c')
    root.connect('Poissons_ratio.nu', 'structures.nu')
    root.connect('Youngs_modulus.E', 'structures.E')
    root.connect('material_density.rho_s', 'structures.rho_s')
    root.connect('xs_b', 'structures.node_coord_all')
    
    root.add('mda_group_h', mda_h, promotes=['*'])
    
    #Explicit connection Hi-Fi
    root.mda_group_h.connect('displacement_transfer_h.delta','delta')
    root.mda_group_h.connect('inter_h.H','displacement_transfer_h.H')
    root.mda_group_h.connect('mult_filter_h.us','displacement_transfer_h.u')
    root.mda_group_h.connect('f_a','load_transfer_h.f_a')
    root.mda_group_h.connect('load_transfer_h.f_node','structures_h.f_node')
    root.mda_group_h.connect('inter_h.H','load_transfer_h.H')
    root.mda_group_h.connect('structures_h.u','mult_filter_h.u')
    
    #This order guarantees that the filters are always privileged in the computation
    root.mda_group_l.set_order(['mult_filter_l', 'inter', 'displacement_transfer', 'aerodynamics', 'load_transfer','structures'])
    root.mda_group_h.set_order(['mult_filter_h', 'inter_h', 'displacement_transfer_h', 'aerodynamics_h', 'load_transfer_h', 'structures_h'])
    #Connect Indep Variables
    root.connect('Mach_number.Mach', 'Mach')
    #root.connect('b_baseline', 'b')
    root.connect('wing_chord.c', 'c')
    root.connect('Poissons_ratio.nu', 'structures_h.nu')
    root.connect('Youngs_modulus.E', 'structures_h.E')
    root.connect('material_density.rho_s', 'structures_h.rho_s')
        
    #Multifidelity explicit connections
    
    root.connect('structures.u', 'mult_filter_h.ul')
    root.connect('structures_h.u', 'mult_filter_l.ul')
    
    #Recorder Lo-Fi
    recorder_l = SqliteRecorder('mda_l.sqlite3')
    recorder_l.options['record_metadata'] = False
    #Recorder Hi-Fi
    recorder_h = SqliteRecorder('mda_h.sqlite3')
    recorder_h.options['record_metadata'] = False
    # recorder.options['includes'] =
    top.root.mda_group_l.nl_solver.add_recorder(recorder_l)
    top.root.mda_group_h.nl_solver.add_recorder(recorder_h)
    
    #Constraint components
    #Lift coefficient constraints (two constraints with same value to treat equality constraint as two inequality constraints)
    root.add('con_lift_cruise_upper', ExecComp(
        'con_l_u = CL - n*(W_airframe+2*1.25*mass)*9.81/(0.5*rho_a*V**2*Sw)'), promotes=['*']) 
    root.add('con_lift_cruise_lower', ExecComp(
        'con_l_l = CL - n*(W_airframe+2*1.25*mass)*9.81/(0.5*rho_a*V**2*Sw)'), promotes=['*'])
    
    #Maximum stress constraint (considering factor of safety)
    root.add('con_stress', ExecComp('con_s = FS*2.5*max(VMStress) - sigma_y', VMStress=np.zeros(n_stress,dtype=float)), promotes=['*'])

    #Fuel burned
    root.add('fuel_burned', ExecComp('fuel_burned = (W_airframe+2.*1.25*mass)*(exp(14306.7*1000/(V*(CL/CD))) - 1)'), promotes=['*'])
    
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
        root.add('s_lower_bound_'+str(i+1), ExecComp('s_l_'+str(i+1) +
                                                     ' = s['+str(i)+']', s=np.zeros(an, dtype=float)), promotes=['*'])
        root.add('s_upper_bound_'+str(i+1), ExecComp('s_u_'+str(i+1) +
                                                     ' = s['+str(i)+']', s=np.zeros(an, dtype=float)), promotes=['*'])

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
    root.connect('xgeom_b', 'interp_struct_morph.node_coord') #base_aerodynamic_mesh_h.xa_b
    root.connect('xs_b', 'interp_struct_morph.jig_surface_coord')
      
    root.connect('interp_aero_hifi_morph.H', 'D')
    root.connect('xgeom_b', 'interp_aero_hifi_morph.node_coord')
    root.connect('base_aerodynamic_mesh_h.xa_b', 'interp_aero_hifi_morph.jig_surface_coord')
    
    #Define the optimizer (Scipy)
    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'COBYLA'
    top.driver.options['disp'] = True
    top.driver.options['tol'] = 1.e-3
    top.driver.options['maxiter'] = 500
    top.driver.opt_settings['rhobeg'] = 0.4

    top.driver.add_desvar('t', lower=t_min, upper=t_max,
                          adder=-t_min, scaler=1./(t_max-t_min))
    top.driver.add_desvar('s', lower=s_min, upper=s_max,
                          adder=-s_min, scaler=1./(s_max-s_min))
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

    top.driver.add_objective('fuel_burned')

    for i in range(n_stress):
        top.driver.add_constraint('con_s_'+str(i+1), upper=0., scaler=1./sigma_y)

    top.driver.add_constraint(
        'con_l_u', upper=0., scaler=1./(n*W_ref*9.81/(0.5*rho_a*V**2*Sw)))
        
    top.driver.add_constraint(
        'con_l_l', lower=0., scaler=1./(n*W_ref*9.81/(0.5*rho_a*V**2*Sw)))

    #Add design variable bounds constraints to the driver
    for i in range(tn):
        top.driver.add_constraint('t_l_'+str(i+1), lower=t_min[i], scaler=1./t[i])
        top.driver.add_constraint('t_u_'+str(i+1), upper=t_max[i], scaler=1./t[i])

    for i in range(an):
        top.driver.add_constraint('s_l_'+str(i+1), lower=s_min[i], scaler=1./s[i])
        top.driver.add_constraint('s_u_'+str(i+1), upper=s_max[i], scaler=1./s[i])

    top.driver.add_constraint('cr_l', lower=cr_min, scaler=1./cr)
    top.driver.add_constraint('cr_u', upper=cr_max, scaler=1./cr)

    top.driver.add_constraint('cb_l', lower=cb_min, scaler=1./cb)
    top.driver.add_constraint('cb_u', upper=cb_max, scaler=1./cb)

    top.driver.add_constraint('ct_l', lower=ct_min, scaler=1./ct)
    top.driver.add_constraint('ct_u', upper=ct_max, scaler=1./ct)

    top.driver.add_constraint('sweep_l', lower=sweep_min, scaler=1./sweep)
    top.driver.add_constraint('sweep_u', upper=sweep_max, scaler=1./sweep)

    top.driver.add_constraint('b_l', lower=b_min, scaler=1./b_0)
    top.driver.add_constraint('b_u', upper=b_max, scaler=1./b_0)

    top.driver.add_constraint('alpha_l', lower=alpha_min, scaler=1./alpha)
    top.driver.add_constraint('alpha_u', upper=alpha_max, scaler=1./alpha)

    #Recorder
    recorder = SqliteRecorder('mdao_panair_adflow.sqlite3')
    recorder.options['record_metadata'] = False
    recorder.options['includes'] = ['CD', 'con_l_u', 'con_s', 't', 'a', 'cr',
                                    'cb', 'ct', 'sweep', 'b', 'alpha']
    
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
    top['t'] = t
    top['s'] = s
    top['cr'] = cr
    top['cb'] = cb
    top['ct'] = ct
    top['sweep'] = sweep
    top['b'] = b_0
    top['alpha'] = alpha
    start2 = time.time()
    top.run()
    end2 = time.time()
    top.cleanup()  # this closes all recorders
    print("Set up time = " + str(end1 - start1))
    print("Run time = " + str(end2 - start2))
