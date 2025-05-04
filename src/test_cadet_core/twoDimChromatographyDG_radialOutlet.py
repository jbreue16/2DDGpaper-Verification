# %% import packages and files
import utility.convergence as convergence
import os
import numpy as np
import json
import shutil
import json
    
from cadet import Cadet

from utility import convergence
import bench_func
import bench_configs
import settings_2Dchromatography


# %% We define multiple settings convering binding modes, surface diffusion and
# multiple particle types. All settings consider three radial zones.
# An analytical solution can be provided and the EOC is computed for three
# radial zones. Ultimately, the discrete maximum norm of the zonal errors is
# considered to compute the EOC.
def LRMP2D_linBnd_tests(
        polyDeg,
        n_jobs, database_path, small_test,
        output_path, cadet_path, reference_data_path=None,
        use_CASEMA_reference=True, zonal_ref_file_names=None,
        rerun_sims=True):

    os.makedirs(output_path, exist_ok=True)

    nRadialZones = 3
    n_settings = 1
    numRef_kwargs = {  # only filled when numerical reference is being used
    'domain_end': 0.0035,
    'ref_coords': convergence.get_radial_coordinates(
        reference_data_path + '/' + zonal_ref_file_names[0], unit='000'
        )
    }
        
    if zonal_ref_file_names is None:
        
        references = [None] * n_settings
        
    elif use_CASEMA_reference:
        raise ValueError("Ff zonal_ref_file_names are provided dynamically, you cannot specify the use of CASEMA references (which are fixed)")
    
    else:
        
        references = []
        
        for idx in range(n_settings):
            # note that we consider the outlet for radial zone 0
            references.extend(
                [convergence.get_solution(
                    reference_data_path + '/' + zonal_ref_file_names[idx], unit='unit_' + str(nRadialZones + 1).zfill(3), which='outlet'
                    )]
                )

    if use_CASEMA_reference:

        references = []
        zonal_ref_file_names = ['CASEMA_reference/ref_2DGRM3Zone_noFilmDiff_1Comp_radZ3.h5']

        # Note: All zones will be considered when use_CASEMA_reference is true.
        # We start with the first and compute the other two in a second step.
        # Finally, we compute a discrete norm of the zonal errors to compute the EOC.
        for idx in range(n_settings):
            # note that we consider radial zone 0
            references.extend(
                [convergence.get_solution(
                    reference_data_path + '/' + zonal_ref_file_names[idx], unit='unit_' + str(nRadialZones + 1).zfill(3), which='outlet'
                )]
            )

    def get_settings():
        return [
            {  # PURE COLUMN TRANSPORT CASE
                'film_diffusion': 0.0,
                # 'col_dispersion_radial' : 0.0,
                'analytical_reference': use_CASEMA_reference,
                'nRadialZones': 3,
                'name': '2DLRMP3Zone_noFilmDiff_1Comp',
                'adsorption_model': 'NONE',
                'par_surfdiffusion': 0.0,
                'reference': references[0]
            }
        ][0:n_settings]

    # %% Define benchmarks

    settings = get_settings()

    cadet_configs = []
    config_names = []
    include_sens = []
    ref_files = []
    unit_IDs = []
    which = []
    idas_abstol = []
    ax_methods = []
    ax_discs = []
    rad_methods = []
    rad_discs = []
    refinement_IDs = []

    def LRMP2D_DG_Benchmark(small_test=False, polyDeg=3, **kwargs):

        nDisc = 4 if small_test else 6
        nRadialZones = kwargs.get('nRadialZones', 3)

        benchmark_config = {
            'cadet_config_jsons': [
                settings_2Dchromatography.model2Dflow_linBnd_benchmark1(
                    transport_model='LUMPED_RATE_MODEL_WITH_PORES_2D',
                    radNElem=nRadialZones,
                    rad_inlet_profile=None,
                    USE_MODIFIED_NEWTON=1, axMethod=3, **kwargs)
            ],
            'include_sens': [
                False
            ],
            'ref_files': [
                [kwargs.get('reference', None)]
            ],
            'refinement_ID': [
                '000'
            ],
            'unit_IDs': [
                '000'
            ],
            'which': [
                'radial_outlet'
            ],
            'idas_abstol': [
                [1e-10]
            ],
            'ax_methods': [
                [polyDeg]
            ],
            'ax_discs': [
                [bench_func.disc_list(4, nDisc)]
            ],
            'rad_methods': [
                [polyDeg]
            ],
            'rad_discs': [
                [bench_func.disc_list(nRadialZones, nDisc)]
            ]
        }

        return benchmark_config

    # %% create benchmark configurations

    for setting in settings:
        addition = LRMP2D_DG_Benchmark(small_test=small_test, polyDeg=polyDeg, **setting)

        bench_configs.add_benchmark(
            cadet_configs, include_sens, ref_files, unit_IDs, which,
            idas_abstol,
            ax_methods, ax_discs, rad_methods=rad_methods, rad_discs=rad_discs,
            refinement_IDs=refinement_IDs,
            addition=addition)

        config_names.extend([setting['name']])

    # %% Run convergence analysis

    Cadet.cadet_path = cadet_path

    bench_func.run_convergence_analysis(
        database_path=database_path, output_path=output_path,
        cadet_path=cadet_path,
        cadet_configs=cadet_configs,
        cadet_config_names=config_names,
        include_sens=include_sens,
        ref_files=ref_files,
        unit_IDs=unit_IDs,
        which=which,
        ax_methods=ax_methods, ax_discs=ax_discs,
        rad_methods=rad_methods, rad_discs=rad_discs,
        idas_abstol=idas_abstol,
        n_jobs=n_jobs,
        rad_inlet_profile=None,
        rerun_sims=rerun_sims,
        refinement_IDs=refinement_IDs,
        zonal_reference=False,
        **numRef_kwargs
    )