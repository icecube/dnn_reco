#!/bin/bash
#PBS -l nodes=1:ppn={cpus}
#PBS -l pmem={memory}
#PBS -l mem={memory}
#PBS -l vmem={memory}
#PBS -l pvmem={memory}
#PBS -l walltime={walltime}
#PBS -o /data/user/mhuennefeld/to_delete/unit_tests/dnn_reco_test_01_base/processing/NuGen/NuTau/medium_energy/IC86_2013_holeice_30_v4/l5/logs/1/run_1_${PBS_JOBID}.out
#PBS -e /data/user/mhuennefeld/to_delete/unit_tests/dnn_reco_test_01_base/processing/NuGen/NuTau/medium_energy/IC86_2013_holeice_30_v4/l5/logs/1/run_1_${PBS_JOBID}.err
#PBS -q long
#PBS -S /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/icetray-start
FINAL_OUT=/data/user/mhuennefeld/to_delete/unit_tests/dnn_reco_test_01_base/NuGen/NuTau/medium_energy/IC86_2013_holeice_30_v4/l5/1/DNN_l5_00000001
KEEP_CRASHED_FILES=False
WRITE_HDF5=True
WRITE_I3=False

echo 'Loading py2-v3.0.1'
eval `/cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/setup.sh`
export PYTHONUSERBASE=/mnt/lfs7/user/mhuennefeld/DNN_reco/virtualenvs/tensorflow_gpu_py2-v3.0.1
echo 'Using PYTHONUSERBASE: '${PYTHONUSERBASE}


export MPLBACKEND=agg
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python2.7/site-packages:$PYTHONPATH

#-----------------------------------------------------------
# Work-around for nodes which do not have correct paths set
#-----------------------------------------------------------
export CUDA_HOME=/mnt/lfs7/user/mhuennefeld/software/condor_cuda3/cuda-8.0;
export PATH=$PATH:/mnt/lfs7/user/mhuennefeld/software/condor_cuda3/cuda-8.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/lfs7/user/mhuennefeld/software/condor_cuda3/cuda-8.0/lib64

# check if CUDA_HOME is set
if [ -z ${CUDA_HOME+x} ]; then
    export CUDA_HOME=/usr/local/cuda-8.0;
    export PATH=$PATH:/usr/local/cuda-8.0/bin:/usr/local/cuda/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
fi
#-----------------------------------------------------------


echo $FINAL_OUT
if [ -z ${PBS_JOBID} ] && [ -z ${_CONDOR_SCRATCH_DIR} ]
then
    echo 'Running Script w/o temporary scratch'
    /mnt/lfs7/user/mhuennefeld/scripts/processing_scripts/processing/scripts/dnn_reco_create_data.py /data/user/mhuennefeld/to_delete/unit_tests/dnn_reco_test_01_base/processing/NuGen/NuTau/medium_energy/IC86_2013_holeice_30_v4/l5/dnn_reco_test_01_base.yaml_0001 1 --no-scratch
    ICETRAY_RC=$?
    echo 'IceTray finished with Exit Code: ' $ICETRAY_RC
    if [ $ICETRAY_RC -ne 0 ] && [ "$KEEP_CRASHED_FILES" = "False" ] ; then
        echo 'Deleting partially processed file! ' $FINAL_OUT
        rm ${FINAL_OUT}.i3.bz2
        rm ${FINAL_OUT}.hdf5
    fi
else
    echo 'Running Script w/ temporary scratch'
    if [ -z ${_CONDOR_SCRATCH_DIR} ]
    then
        cd /scratch/${USER}
    else
        cd ${_CONDOR_SCRATCH_DIR}
    fi
    /mnt/lfs7/user/mhuennefeld/scripts/processing_scripts/processing/scripts/dnn_reco_create_data.py /data/user/mhuennefeld/to_delete/unit_tests/dnn_reco_test_01_base/processing/NuGen/NuTau/medium_energy/IC86_2013_holeice_30_v4/l5/dnn_reco_test_01_base.yaml_0001 1 --scratch
    ICETRAY_RC=$?
    echo 'IceTray finished with Exit Code: ' $ICETRAY_RC
    if [ $ICETRAY_RC -eq 0 ] || [ $KEEP_CRASHED_FILES -eq 1 ]; then
        if [ "$WRITE_HDF5" = "True" ]; then
            cp *.hdf5 /data/user/mhuennefeld/to_delete/unit_tests/dnn_reco_test_01_base/NuGen/NuTau/medium_energy/IC86_2013_holeice_30_v4/l5/1
        fi
        if [ "$WRITE_I3" = "True" ]; then
            cp *.i3.bz2 /data/user/mhuennefeld/to_delete/unit_tests/dnn_reco_test_01_base/NuGen/NuTau/medium_energy/IC86_2013_holeice_30_v4/l5/1
        fi
    fi

    # Clean Up
    if [ "$WRITE_HDF5" = "True" ]; then
        rm *.hdf5
    fi
    if [ "$WRITE_I3" = "True" ]; then
        rm *.i3.bz2
    fi
fi
exit $ICETRAY_RC

