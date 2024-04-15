

# ---------------------------------------
# Test:
#       - pulse summary values data
#       - DNN reco based on summary
# Note: the tests here use the processing framework `ic3_processing` from
https://github.com/mhuen/ic3-processing
# and the config located in this directory:
./test_data_cfg.yaml
# ---------------------------------------

# Define directory for output, which we will directly place into the
# test_data directory of dnn_reco
export dnn_reco_dir=/INSERT/PATH/TO/DNN_RECO/DIRECTORY
export output_dir=${dnn_reco_dir}/tests_manual/test_data/dnn_reco_test_01

# Create jobs [adjust the python env in config prior to job creation]
ic3_create_job_files ${dnn_reco_dir}/tests_manual/test_data_cfg.yaml -d ${output_dir}

# Run scripts on NPX GPU/CPU
${output_dir}/processing/NuGen/NuE/low_energy/IC86_2013_holeice_30_v4/l5/jobs/1/job_DNN_l5_00000002.sh
${output_dir}/processing/NuGen/NuMu/low_energy/IC86_2013_holeice_30_v4/l5/jobs/1/job_DNN_l5_00000002.sh
${output_dir}/processing/NuGen/NuTau/low_energy/IC86_2013_holeice_30_v4/l5/jobs/1/job_DNN_l5_00000002.sh
${output_dir}/processing/NuGen/NuE/medium_energy/IC86_2013_holeice_30_v4/l5/jobs/1/job_DNN_l5_00000001.sh
${output_dir}/processing/NuGen/NuMu/medium_energy/IC86_2013_holeice_30_v4/l5/jobs/1/job_DNN_l5_00000001.sh
${output_dir}/processing/NuGen/NuTau/medium_energy/IC86_2013_holeice_30_v4/l5/jobs/1/job_DNN_l5_00000001.sh

# run test
python ${dnn_reco_dir}/tests_manual/test.py
