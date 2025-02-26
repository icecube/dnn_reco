#!/bin/bash

# Gather keys
FINAL_OUT=/data/user/mhuennefeld/software/repositories/dnn_recoV2/tests_manual/test_data/dnn_reco_test_01/NuGen/NuE/medium_energy/IC86_2013_holeice_30_v4/l5/1/DNN_l5_00000001
I3_ENDING=i3.bz2
KEEP_CRASHED_FILES=False
WRITE_HDF5=True
WRITE_I3=False
CUDA_HOME=/data/user/mhuennefeld/software/cuda/cuda-11.8
LD_LIBRARY_PATH_PREPENDS={ld_library_path_prepends}
CVMFS_PYTHON=py3-v4.3.0
PYTHON_PACKAGE_IMPORTS={python_package_imports}


# load environment
echo 'Starting job on Host: '$HOSTNAME
echo 'Loading: ' ${CVMFS_PYTHON}
eval `/cvmfs/icecube.opensciencegrid.org/${CVMFS_PYTHON}/setup.sh`
export PYTHONUSERBASE=/data/user/mhuennefeld/DNN_reco/virtualenvs/tensorflow_gpu_py3-v4.3.0
echo 'Using PYTHONUSERBASE: '${PYTHONUSERBASE}

export ENV_SITE_PACKAGES=$(find ${PYTHONUSERBASE}/lib* -maxdepth 2 -type d -name "site-packages")
export PYTHONPATH=$ENV_SITE_PACKAGES:$PYTHONPATH
export PATH=$PYTHONUSERBASE/bin:$PATH
echo 'Using PYTHONPATH: '${PYTHONPATH}

# set MPL backend for Matplotlib
export MPLBACKEND=agg

# add cuda directory
if [ "$(echo "$CUDA_HOME" | sed 's/^.\(.*\).$/\1/')" = "cuda_home" ]; then
  echo 'No cuda home provided. Not adding cuda to path.'
else
  echo 'Adding cuda dir: '$CUDA_HOME
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi

# add additional LD_LIBRARY_PATH additions if we have them
if [ "$(echo "$LD_LIBRARY_PATH_PREPENDS" | sed 's/^.\(.*\).$/\1/')" != "ld_library_path_prepends" ]; then
  echo 'Prepending to LD_LIBRARY_PATH: '$LD_LIBRARY_PATH_PREPENDS
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_PREPENDS:$LD_LIBRARY_PATH
fi

# add additional python package imports if we have them
if [ "$(echo "$PYTHON_PACKAGE_IMPORTS" | sed 's/^.\(.*\).$/\1/')" != "python_package_imports" ]; then
  echo 'Importing additional python packages: '$PYTHON_PACKAGE_IMPORTS
  export PYTHON_PACKAGE_IMPORTS
fi

# start python script
echo 'Starting process for output file: '$FINAL_OUT
if [ -z ${PBS_JOBID} ] && [ -z ${_CONDOR_SCRATCH_DIR} ]
then
    echo 'Running Script w/o temporary scratch'
    /data/user/mhuennefeld/software/repositories/dnn_recoV2/tests_manual/test_data/dnn_reco_test_01/processing/NuGen/NuE/medium_energy/IC86_2013_holeice_30_v4/l5/processing_steps_0001/general_i3_processing_step_0001.py /data/user/mhuennefeld/software/repositories/dnn_recoV2/tests_manual/test_data/dnn_reco_test_01/processing/NuGen/NuE/medium_energy/IC86_2013_holeice_30_v4/l5/processing_steps_0001/test_data_cfg_step_0001.yaml 1 --no-scratch
    JOB_RC=$?
    echo 'Job finished with Exit Code: ' $JOB_RC
    if [ $JOB_RC -ne 0 ] && [ "$KEEP_CRASHED_FILES" = "False" ] ; then
        echo 'Deleting partially processed file! ' $FINAL_OUT

        # Clean Up
        if [ "$WRITE_HDF5" = "True" ]; then
            rm ${FINAL_OUT}.hdf5
        fi
        if [ "$WRITE_I3" = "True" ]; then
            rm ${FINAL_OUT}.${I3_ENDING}
        fi
        if [ -f "$FINAL_OUT" ]; then
            rm $FINAL_OUT
        fi

    fi
else
    echo 'Running Script w/ temporary scratch'
    if [ -z ${_CONDOR_SCRATCH_DIR} ]
    then
        cd /scratch/${USER}
    else
        cd ${_CONDOR_SCRATCH_DIR}
    fi
    /data/user/mhuennefeld/software/repositories/dnn_recoV2/tests_manual/test_data/dnn_reco_test_01/processing/NuGen/NuE/medium_energy/IC86_2013_holeice_30_v4/l5/processing_steps_0001/general_i3_processing_step_0001.py /data/user/mhuennefeld/software/repositories/dnn_recoV2/tests_manual/test_data/dnn_reco_test_01/processing/NuGen/NuE/medium_energy/IC86_2013_holeice_30_v4/l5/processing_steps_0001/test_data_cfg_step_0001.yaml 1 --scratch
    JOB_RC=$?
    echo 'Job finished with Exit Code: ' $JOB_RC
    if [ $JOB_RC -eq 0 ] || [ "$KEEP_CRASHED_FILES" = "True" ]; then

        # create output folder
        mkdir -p /data/user/mhuennefeld/software/repositories/dnn_recoV2/tests_manual/test_data/dnn_reco_test_01/NuGen/NuE/medium_energy/IC86_2013_holeice_30_v4/l5/1

        if [ "$WRITE_HDF5" = "True" ]; then
            cp DNN_l5_00000001.hdf5 /data/user/mhuennefeld/software/repositories/dnn_recoV2/tests_manual/test_data/dnn_reco_test_01/NuGen/NuE/medium_energy/IC86_2013_holeice_30_v4/l5/1
        fi
        if [ "$WRITE_I3" = "True" ]; then
            cp DNN_l5_00000001.${I3_ENDING} /data/user/mhuennefeld/software/repositories/dnn_recoV2/tests_manual/test_data/dnn_reco_test_01/NuGen/NuE/medium_energy/IC86_2013_holeice_30_v4/l5/1
        fi
        if [ -f "$FINAL_OUT" ]; then
            cp DNN_l5_00000001 /data/user/mhuennefeld/software/repositories/dnn_recoV2/tests_manual/test_data/dnn_reco_test_01/NuGen/NuE/medium_energy/IC86_2013_holeice_30_v4/l5/1
        fi
    fi

    # Clean Up
    if [ "$WRITE_HDF5" = "True" ]; then
        rm DNN_l5_00000001.hdf5
    fi
    if [ "$WRITE_I3" = "True" ]; then
        rm DNN_l5_00000001.${I3_ENDING}
    fi
    if [ -f "$FINAL_OUT" ]; then
        rm DNN_l5_00000001
    fi
fi
exit $JOB_RC
