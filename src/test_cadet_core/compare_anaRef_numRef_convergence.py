# -*- coding: utf-8 -*-
"""

Test script to investigate a difference betwenn the convergence behaviour towards the analytical and numerical reference.
An error in the analytical comparison is suspected

@author: jmbr
""" 
  
#%% Include packages
import os
import sys
from pathlib import Path
import re
from joblib import Parallel, delayed
import numpy as np

from cadet import Cadet
from cadetrdm import ProjectRepo

import utility.convergence as convergence
import bench_func
import bench_configs

import twoDimChromatography
import twoDimChromatographyDG
import twoDimChromatographyDG_radialOutlet
import settings_2Dchromatography

#%% User Input

commit_message = f"Compare FV and DG convergence to analytical and numerical reference"

rdm_debug_mode = 1 # Run CADET-RDM in debug mode to test if the script works

delete_h5_files = 0 # delete h5 files (but keep convergence tables and plots)
exclude_files = None # ["file1", "file2"] # specify h5 files that should not be deleted


database_path = "https://jugit.fz-juelich.de/IBG-1/ModSim/cadet/cadet-database" + \
    "/-/raw/core_tests/cadet_config/test_cadet-core/"

sys.path.append(str(Path(".")))
project_repo = ProjectRepo()
output_path = project_repo.output_path / "test_cadet-core"

# %% Run with CADET-RDM

with project_repo.track_results(results_commit_message=commit_message, debug=rdm_debug_mode):

    rerun_sims = 0
    small_test = 0 # Defines a smaller test set (less numerical refinement steps)
    n_jobs = -1 # For parallelization on the number of simulations
    cadet_path = r"C:\Users\jmbr\OneDrive\Desktop\CADET_compiled\master4_crysPartII_d0888cb\aRELEASE\bin\cadet-cli.exe"
    Cadet.cadet_path = cadet_path # convergence.get_cadet_path()


    # twoDimChromatography.GRM2D_linBnd_tests(
    #         n_jobs=n_jobs, database_path=None, small_test=small_test,
    #         output_path=str(output_path) + "/anaRef_2D_chromatography", cadet_path=cadet_path,
    #         reference_data_path=str(project_repo.output_path.parent / 'data'),
    #         use_CASEMA_reference=True, rerun_sims=rerun_sims)


    # twoDimChromatographyDG.LRMP2D_linBnd_tests(
    #     polyDeg=3, n_jobs=n_jobs, database_path=None, small_test=small_test,
    #     output_path=str(output_path) + "/anaRef_2D_chromatography", cadet_path=cadet_path,
    #     reference_data_path=str(project_repo.output_path.parent / 'data'),
    #     use_CASEMA_reference=True, rerun_sims=rerun_sims
    #     )

    if delete_h5_files:
        convergence.delete_h5_files(str(output_path) + "/anaRef_2Dchromatography", exclude_files=exclude_files)


    # twoDimChromatography.GRM2D_linBnd_tests(
    #         n_jobs=n_jobs, database_path=None, small_test=small_test,
    #         output_path=str(output_path) + "/numRef_2D_chromatography", cadet_path=cadet_path,
    #         reference_data_path=str(project_repo.output_path.parent / 'data'),
    #         use_CASEMA_reference=False, zonal_ref_file_names=[r"2DLRMP3Zone_noFilmDiff_1Comp_DG_axP3Z128_radP3Z96.h5"],
    #         rerun_sims=rerun_sims)


    # twoDimChromatographyDG.LRMP2D_linBnd_tests(
    #     polyDeg=3, n_jobs=n_jobs, database_path=None, small_test=small_test,
    #     output_path=str(output_path) + "/anaRef_2D_chromatography", cadet_path=cadet_path,
    #     reference_data_path=str(project_repo.output_path.parent / 'data'),
    #     comparison_mode=0,
    #     rerun_sims=rerun_sims
    #     )


    # twoDimChromatographyDG.LRMP2D_linBnd_tests(
    #     polyDeg=3, n_jobs=n_jobs, database_path=None, small_test=small_test,
    #     output_path=str(output_path) + "/numRef_2D_chromatography", cadet_path=cadet_path,
    #     reference_data_path=str(project_repo.output_path.parent / 'data'),
    #     comparison_mode=1, ref_file_names=[r"2DLRMP3Zone_noFilmDiff_1Comp_DG_axP3Z128_radP3Z96.h5"],
    #     rerun_sims=rerun_sims
    #     )


    twoDimChromatographyDG.LRMP2D_linBnd_tests(
        polyDeg=3, n_jobs=n_jobs, database_path=None, small_test=small_test,
        output_path=str(output_path) + "/2D_chromatography", cadet_path=cadet_path,
        reference_data_path=str(project_repo.output_path.parent / 'data'),
        comparison_mode=2, #ref_file_names=[r"2DLRMP3Zone_noFilmDiff_1Comp_DG_axP3Z128_radP3Z96.h5"],
        rerun_sims=rerun_sims
        )


    if delete_h5_files:
        convergence.delete_h5_files(str(output_path) + "/numRef_2Dchromatography", exclude_files=exclude_files)
        
