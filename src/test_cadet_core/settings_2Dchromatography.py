# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:21:20 2024

This script defines model settings considered for the verification of the
2D flow chromatography models in CADET-Core

@author: jmbr
"""


from utility import convergence
from cadet import Cadet
import numpy as np
from matplotlib import pyplot as plt
from addict import Dict
import h5py
import json
import re
import copy

# =============================================================================
# Definition of helper functions
# =============================================================================

from scipy.special import legendre


def q_and_L(poly_deg, x):
    P = legendre(poly_deg)
    dP = P.deriv()
    L = P(x)
    q = P.deriv()(x)
    q_der = P.deriv(2)(x)
    return L, q, q_der


def lgl_nodes_weights(poly_deg):
    if poly_deg < 1:
        raise ValueError("Polynomial degree must be at least 1!")

    nodes = np.zeros(poly_deg + 1)
    weights = np.zeros(poly_deg + 1)
    pi = np.pi
    tolerance = 1e-15
    n_iterations = 10

    if poly_deg == 1:
        nodes[0] = -1
        nodes[1] = 1
        weights[0] = 1
        weights[1] = 1
    else:
        nodes[0] = -1
        nodes[poly_deg] = 1
        weights[0] = 2.0 / (poly_deg * (poly_deg + 1.0))
        weights[poly_deg] = weights[0]

        for j in range(1, (poly_deg + 1) // 2):
            x = -np.cos(pi * (j + 0.25) / poly_deg - 3 /
                        (8.0 * poly_deg * pi * (j + 0.25)))
            for k in range(n_iterations):
                L, q, q_der = q_and_L(poly_deg, x)
                dx = q / q_der
                x -= dx
                if abs(dx) <= tolerance * abs(x):
                    break
            nodes[j] = x
            nodes[poly_deg - j] = -x
            L, q, q_der = q_and_L(poly_deg, x)
            weights[j] = 2.0 / (poly_deg * (poly_deg + 1.0) * L**2)
            weights[poly_deg - j] = weights[j]

        if poly_deg % 2 == 0:
            L, q, q_der = q_and_L(poly_deg, 0.0)
            nodes[poly_deg // 2] = 0
            weights[poly_deg // 2] = 2.0 / (poly_deg * (poly_deg + 1.0) * L**2)

    return nodes, weights
# # Example usage:
# poly_deg = 5  # Change this to the desired order
# nodes, weights = lgl_nodes_weights(poly_deg)

# # Display the results
# print("nodes:", nodes)
# print("Weights:", weights)
# print("wiki node", np.sqrt(1/3 - 2*np.sqrt(7) / 21))
# print("wiki weight", (14 + np.sqrt(7))/30)
# print("Sum of weights:", sum(weights))
# print("Inv weights:", np.reciprocal(weights))


def generate_connections_matrix(rad_method, rad_cells,
                                velocity, porosity, col_radius,
                                add_inlet_per_port=True, add_outlet=False):
    """Computes the connections matrix with const. velocity flow rates, and radial coordinates.
    Equidistant cell/element spacing is assumed.
    
    Parameters
    ----------
    rad_method : int
        radial method / polynomial degree
    rad_cells : int
        radial number of cells
    velocity : float
        column velocity (constant)
    porosity : float
        column porosity (constant)
    col_radius : float
        column radius (constant)
    add_inlet_per_port : int | bool
        specifies how many radial zones are used either by number or by true to specify one per port
    add_outlet : bool
        specifies whetehr or not an outlet is connected per radial zone
    
    Returns
    -------
    List of float, List of float
        Connections matrix, radial coordinates.
    """

    nRadPoints = (rad_method + 1) * rad_cells

    # we want the same velocity within each radial zone and use an equidistant radial grid, ie we adjust the volumetric flow rate accordingly in each port
    # 1. compute cross sections

    subcellCrossSectionAreas = []
    rad_coords = []

    if rad_method > 0:

        nodes, weights = lgl_nodes_weights(rad_method)
        # scale the weights to radial element spacing
        # note that weights need to be scaled to 1 later, to give us the size of the corresponding subcells
        # print(sum(weights) / 2.0 - 1.0 < 1E-15)
        deltaR = col_radius / rad_cells
        for rIdx in range(rad_cells):
            jojoL = rIdx * deltaR
            for node in range(rad_method + 1):
                jojoR = jojoL + weights[node] / 2.0 * deltaR
                # print("Left boundary: ", jojoL)
                # print("Right boundary: ", jojoR)
                subcellCrossSectionAreas.append(
                    np.pi * (jojoR ** 2 - jojoL ** 2))
                rad_coords.append(
                    rIdx * deltaR + (nodes[node] + 1) / 2.0 * deltaR)
                jojoL = jojoR
    else:
        deltaR = col_radius / nRadPoints
        jojoL = 0.0
        for rIdx in range(nRadPoints):
            rad_coords.append(rIdx * deltaR + deltaR / 2.0)
            jojoR = jojoL + deltaR
            subcellCrossSectionAreas.append(np.pi * (jojoR ** 2 - jojoL ** 2))
            jojoL = jojoR

    # print("subcellCrossSectionAreas: ", subcellCrossSectionAreas)
    # print(len(subcellCrossSectionAreas) == nRadPoints)

    # create flow rates for each zone
    flowRates = []
    columnIdx = 0  # always needs to be the first unit
    
    for rad in range(nRadPoints):
        flowRates.append(subcellCrossSectionAreas[rad] * porosity * velocity)
    # create connections matrix
    connections = []
    # add inlet connections
    if add_inlet_per_port:

        nRadialZones = rad_cells if add_inlet_per_port is True else add_inlet_per_port

        if not rad_cells % nRadialZones == 0:
            raise Exception(
                f"Number of rad_cells {rad_cells} is not a multiple of radial zones {nRadialZones}")

        for rad in range(nRadPoints):
            zone = int(rad / (nRadPoints / nRadialZones))
            connections += [zone + 1, columnIdx,
                            0, rad, -1, -1, flowRates[rad]]
            if add_outlet:
                connections += [columnIdx, nRadialZones + 1 + zone,
                                rad, 0, -1, -1, flowRates[rad]]
    else:
        for rad in range(nRadPoints):
            connections += [1, columnIdx, 0, rad, -1, -1, flowRates[rad]]
            if add_outlet:
                connections += [columnIdx, nRadPoints + 1 + rad,
                                rad, 0, -1, -1, flowRates[rad]]
                
    return connections, rad_coords

# =============================================================================
# Setting
# =============================================================================


# %% Verification settings


def model2Dflow_linBnd_benchmark1(
        axMethod=0, axNElem=8,
        radMethod=0, radNElem=3,
        parMethod=0, parNElem=2,
        nRadialZones=1,  # discontinuous radial zones (equidistant)
        tolerance=1e-12,
        plot=False, run=False,
        save_path="C:/Users/jmbr/JupyterNotebooks/",
        file_name=None,
        export_json_config=False,
        particle_models=["0D"],
        **kwargs
):

    nRadPoints = (radMethod + 1) * radNElem
    nInlets = nRadialZones if kwargs.get('rad_inlet_profile', None) is None else nRadPoints
    nOutlets = nRadialZones

    column = Dict()

    column.UNIT_TYPE = "COLUMN_MODEL_2D"
    nComp = 1
    column.NCOMP = nComp

    column.COL_LENGTH = 0.014
    column.COL_RADIUS = 0.0035
    column.CROSS_SECTION_AREA = np.pi * column.COL_RADIUS**2
    column.NPARTYPE = len(particle_models)
    column.COL_POROSITY = 0.37
    column.PAR_TYPE_VOLFRAC = kwargs.get('par_type_volfrac', 1.0)

    column.VELOCITY = 3.45 / (100.0 * 60.0)  # 3.45 cm/min
    column.COL_DISPERSION = 5.75e-8
    column.COL_DISPERSION_RADIAL = kwargs.get('col_dispersion_radial', 5e-8)
    
    for parType in range(column.NPARTYPE):
    
        groupName = 'particle_type_' + str(parType).zfill(3)
        
        column[groupName].FILM_DIFFUSION = kwargs.get('film_diffusion', 6.9e-6)
        column[groupName].PAR_RADIUS = kwargs.get('par_radius', 45E-6)
        column[groupName].PAR_POROSITY = kwargs.get('par_porosity', 0.75)
        
        if particle_models[parType] == "1D":
            column[groupName].PARTICLE_TYPE = "GENERAL_RATE_PARTICLE"
            column[groupName].PAR_DIFFUSION = kwargs.get('par_diffusion', 6.07e-11)
            column[groupName].PAR_SURFDIFFUSION = kwargs.get('par_surfdiffusion', 0.0)
        else:
            column[groupName].PARTICLE_TYPE = "HOMOGENEOUS_PARTICLE"

        # binding parameters
        column[groupName].nbound = 1
        column[groupName].adsorption_model = 'LINEAR'
        if 'adsorption' in kwargs:
            column[groupName].adsorption.is_kinetic = kwargs['adsorption.is_kinetic'][parType]
            column[groupName].adsorption.lin_ka = kwargs['adsorption.lin_ka'][parType]
            column[groupName].adsorption.lin_kd = kwargs['adsorption.lin_kd'][parType]
        else:
            column[groupName].adsorption.is_kinetic = 0
            column[groupName].adsorption.lin_ka = 35.5
            column[groupName].adsorption.lin_kd = 1.0
            
        
        if axMethod > 0 and particle_models[parType] == "1D":
            
            column[groupName].discretization.PAR_DISC_TYPE = ['EQUIDISTANT_PAR']
            column[groupName].discretization.PAR_POLYDEG = parMethod
            column[groupName].discretization.PAR_NELEM = parNElem

    if 'INIT_C' in kwargs:

        rad_coords = np.zeros(nRadPoints)
        ax_coords = np.zeros((axMethod+1)*axNElem)
        ax_delta = column.COL_LENGTH / axNElem
        rad_delta = column.COL_RADIUS / radNElem

        if axMethod > 0:
            ax_nodes, _ = lgl_nodes_weights(axMethod)
            rad_nodes, _ = lgl_nodes_weights(radMethod)

            for idx in range(axNElem):
                ax_coords[idx * (axMethod+1): (idx + 1) * (axMethod+1)
                          ] = convergence.map_xi_to_z(ax_nodes, idx, ax_delta)
            for idx in range(radNElem):
                rad_coords[idx * (radMethod+1): (idx + 1) * (radMethod+1)
                           ] = convergence.map_xi_to_z(rad_nodes, idx, rad_delta)
        else:
            ax_coords = np.array([idx * ax_delta for idx in range(axNElem)])
            rad_coords = np.array([rad_delta / 2.0 + idx * rad_delta for idx in range(radNElem)])

        column.init_c = kwargs['INIT_C'](ax_coords, rad_coords)
    else:
        column.init_c = [0] * nComp
    column.init_cp = kwargs.get('init_cp', [0] * nComp)
    column.init_q = kwargs.get('init_cs', [0] * nComp)

    if axMethod > 0:
        column.discretization.SPATIAL_METHOD = "DG"
        column.discretization.AX_POLYDEG = axMethod
        column.discretization.AX_NELEM = axNElem
        column.discretization.RAD_POLYDEG = radMethod
        column.discretization.RAD_NELEM = radNElem
    else:
        if particle_models[0] == "1D":
            column.discretization.NPAR = parNElem
            column.discretization.PAR_DISC_TYPE = ['EQUIDISTANT_PAR']
        column.discretization.SPATIAL_METHOD = "FV"
        column.discretization.NCOL = axNElem
        column.discretization.NRAD = radNElem
        column.discretization.SCHUR_SAFETY = 1.0e-8
        column.discretization.weno.BOUNDARY_MODEL = 0
        column.discretization.weno.WENO_EPS = 1e-10
        column.discretization.weno.WENO_ORDER = 3
        column.discretization.GS_TYPE = 1
        column.discretization.MAX_KRYLOV = 0
        column.discretization.MAX_RESTARTS = 10

    column.discretization.USE_ANALYTIC_JACOBIAN = True
    column.discretization.RADIAL_DISC_TYPE = 'EQUIDISTANT'
    column.PORTS = nRadPoints

    inletUnit = Dict()

    inletUnit.INLET_TYPE = 'PIECEWISE_CUBIC_POLY'
    inletUnit.UNIT_TYPE = 'INLET'
    inletUnit.NCOMP = nComp
    inletUnit.sec_000.CONST_COEFF = [kwargs.get('INLET_CONST', 1.0)] * nComp
    inletUnit.sec_001.CONST_COEFF = [0.0] * nComp
    inletUnit.ports = 1
    
    # define cadet model using the unit-dicts above
    model = Dict()

    model.model.nunits = 1 + nInlets + nOutlets

    # Store solution
    model['return'].split_components_data = 0
    model['return'].split_ports_data = kwargs.get('SPLIT_PORTS_DATA', 1)
    model['return']['unit_000'].WRITE_SOLUTION_INLET = kwargs.get(
        'WRITE_SOLUTION_INLET', 0)
    model['return']['unit_000'].WRITE_SOLUTION_FLUX = kwargs.get(
        'WRITE_SOLUTION_FLUX', 0)
    model['return']['unit_000'].WRITE_SOLUTION_OUTLET = kwargs.get(
        'WRITE_SOLUTION_OUTLET', 1)
    model['return']['unit_000'].WRITE_SOLUTION_BULK = kwargs.get(
        'WRITE_SOLUTION_BULK', 0)
    model['return']['unit_000'].WRITE_SOLUTION_PARTICLE = kwargs.get(
        'WRITE_SOLUTION_PARTICLE', 0)
    model['return']['unit_000'].WRITE_SOLUTION_SOLID = kwargs.get(
        'WRITE_SOLUTION_SOLID', 0)
    model['return']['unit_000'].WRITE_COORDINATES = 1
    model['return']['unit_000'].WRITE_SENS_OUTLET = kwargs.get(
        'WRITE_SENS_OUTLET', 0)

    # Tolerances for the time integrator
    model.solver.time_integrator.USE_MODIFIED_NEWTON = kwargs.get(
        'USE_MODIFIED_NEWTON', 0)
    model.solver.time_integrator.ABSTOL = 1e-6
    model.solver.time_integrator.ALGTOL = 1e-10
    model.solver.time_integrator.RELTOL = 1e-6
    model.solver.time_integrator.INIT_STEP_SIZE = 1e-6
    model.solver.time_integrator.MAX_STEPS = 1000000

    # Solver settings
    model.model.solver.GS_TYPE = 1
    model.model.solver.MAX_KRYLOV = 0
    model.model.solver.MAX_RESTARTS = 10
    model.model.solver.SCHUR_SAFETY = 1e-8

    # Run the simulation on single thread
    model.solver.NTHREADS = 1
    model.solver.CONSISTENT_INIT_MODE = 3
    
    # Sections
    model.solver.sections.NSEC = 2
    model.solver.sections.SECTION_TIMES = [0.0, 10.0, 1500.0]

    # get connections matrix
    if re.search("2D", column.UNIT_TYPE):
        connections, rad_coords = generate_connections_matrix(
            rad_method=radMethod, rad_cells=radNElem,
            velocity=column.VELOCITY, porosity=column.COL_POROSITY, col_radius=column.COL_RADIUS,
            add_inlet_per_port=nInlets, add_outlet=True
        )

    else:
        Q = np.pi * column.COL_RADIUS**2 * column.VELOCITY
        connections = [1, 0, -1, -1, Q]
        rad_coords = [column.COL_RADIUS / 2.0]

    outletUnit = Dict()
    outletUnit.UNIT_TYPE = 'OUTLET'
    outletUnit.NCOMP = nComp
            
    # Set units
    model.model['unit_000'] = column
    model.model['unit_001'] = copy.deepcopy(inletUnit)

    if kwargs.get('rad_inlet_profile', None) is None:
        for rad in range(max(1, nRadialZones)):

            model.model['unit_' + str(rad + 1).zfill(3)
                        ] = copy.deepcopy(inletUnit)

            model.model['unit_' + str(rad + 1).zfill(
                3)].sec_000.CONST_COEFF = float(rad + 1) if nRadialZones > 0 else 0.0

            model.model['unit_' + str(nRadialZones + 1 + rad).zfill(3)] = copy.deepcopy(outletUnit)
            model['return']['unit_' + str(nRadialZones + 1 + rad).zfill(3)] = model['return']['unit_000']

    else:
        for rad in range(nRadPoints):

            model.model['unit_' + str(rad + 1).zfill(3)
                        ] = copy.deepcopy(inletUnit)

            model.model['unit_' + str(rad + 1).zfill(
                3)].sec_000.CONST_COEFF = [kwargs['rad_inlet_profile'](rad_coords[rad], column.COL_RADIUS)] * nComp

            model.model['unit_' + str(nRadPoints + 1 + rad).zfill(3)] = copy.deepcopy(outletUnit)
            model['return']['unit_' + str(nRadPoints + 1 + rad).zfill(3)] = model['return']['unit_000']

    model.model.connections.NSWITCHES = 1
    model.model.connections.switch_000.SECTION = 0
    model.model.connections.switch_000.connections = connections

    model.solver.sections.SECTION_CONTINUITY = [0,]
    model.solver.USER_SOLUTION_TIMES = np.linspace(1, 1500, 1500)

    if not run:
        return {'input': model}
    else:
        cadet_model = Cadet()
        if 'cadet_path' in kwargs:
            Cadet.cadet_path = kwargs['cadet_path']
        cadet_model.root.input = model

        if column.discretization.SPATIAL_METHOD == "FV":
            cadet_model.filename = save_path + \
                file_name if file_name is not None else save_path + "FV_grm2d_debug.h5"
        else:
            cadet_model.filename = save_path + \
                file_name if file_name is not None else save_path + "lrmp2d_debug.h5"
        cadet_model.save()

        data = cadet_model.run()
        if data.return_code == 0:
            print(cadet_model.filename + " simulation completed successfully")
            cadet_model.load()
        else:
            print(data)
            raise Exception(cadet_model.filename + " simulation failed")

        if plot:

            plt.figure()
            zeitpunkt = int(plot)
            time = cadet_model.root.output.solution.solution_times
            ax_coords = cadet_model.root.output.coordinates['unit_000'].axial_coordinates
            c = cadet_model.root.output.solution['unit_000'].solution_bulk

            if re.search("2D", column.UNIT_TYPE):
                for rad in range(nRadPoints):
                    plt.plot(ax_coords, c[zeitpunkt, :, rad, 0],
                             linestyle='dashed', label='c' + str(rad))
            else:
                plt.plot(ax_coords, c[zeitpunkt, :, 0],
                         linestyle='dashed', label='c' + str(rad))

            if nRadPoints <= 12:
                plt.legend()
            plt.title(f'Column bulk at t = {zeitpunkt}')
            plt.xlabel('$time~/~s$')
            plt.ylabel(r'$concentration~/~mol \cdot L^{-1} $')
            plt.show()

        return cadet_model
    
    
       
#%%
 
polyDeg = 3

kwargs = {
    'cadet_path': r'C:\Users\jmbr\software\CADET-Core\out\install\aRELEASE\bin\cadet-cli.exe',
    'par_surfdiffusion': 1e-11,
    'particle_models': ["1D", "0D"],
    'par_type_volfrac': [0.4, 0.6],
    'adsorption.is_kinetic': [1, 0], 'adsorption.lin_ka': [35.5], 'adsorption.lin_kd': [1.0, 1.0]
    }

save_path=r"C:\Users\jmbr\software\Verify-DG2D/"
file_name = 'ref_COL2D_GRMsd3Zone2ParType_dynLin_1Comp_benchmark1_DG_axP3Z8_radP3Z3_parP3Z1.h5'

model = model2Dflow_linBnd_benchmark1(
    axMethod=polyDeg, axNElem=8,
    radMethod=polyDeg, radNElem=3,
    parMethod=polyDeg, parNElem=1,
    nRadialZones=3,  # discontinuous radial zones (equidistant)
    tolerance=1e-10,
    plot=False, run=1,
    save_path=save_path,
    file_name=file_name,
    export_json_config=False,
    **kwargs
    )
    
#%%

# save_path=r"C:\Users\jmbr\software\Verify-DG2D/"
# file_name = 'ref_2DGRMsd3Zone_dynLin_1Comp_benchmark1_DG_axP3Z8_radP3Z3_parP3Z1.h5'

# model.save_as_python_script(filename="jojo.py")







