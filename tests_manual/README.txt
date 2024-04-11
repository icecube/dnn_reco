

# ---------------------------------------
# Test:
#       - pulse summary values data
#       - DNN reco based on summary
# Note: the tests here use the processing scripts from
# https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/sandbox/mhuennefeld/processing_scripts/trunk/
# and more specifically this config:
# https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/sandbox/mhuennefeld/processing_scripts/trunk/processing/configs/unit_tests/dnn_reco_test_01.yaml
# ---------------------------------------

# Define directory for output
export output_dir=/data/user/mhuennefeld/to_delete/unit_tests/dnn_reco_test_01

# Create jobs [adjust the python env in config prior to job creation]
python create_job_files.py configs/unit_tests/dnn_reco_test_01.yaml -d ${output_dir}

# Run scripts on NPX GPU/CPU
${output_dir}/processing/NuGen/NuE/low_energy/IC86_2013_holeice_30_v4/l5/jobs/1/job_DNN_l5_00000002.sh
${output_dir}/processing/NuGen/NuMu/low_energy/IC86_2013_holeice_30_v4/l5/jobs/1/job_DNN_l5_00000002.sh
${output_dir}/processing/NuGen/NuTau/low_energy/IC86_2013_holeice_30_v4/l5/jobs/1/job_DNN_l5_00000002.sh
${output_dir}/processing/NuGen/NuE/medium_energy/IC86_2013_holeice_30_v4/l5/jobs/1/job_DNN_l5_00000001.sh
${output_dir}/processing/NuGen/NuMu/medium_energy/IC86_2013_holeice_30_v4/l5/jobs/1/job_DNN_l5_00000001.sh
${output_dir}/processing/NuGen/NuTau/medium_energy/IC86_2013_holeice_30_v4/l5/jobs/1/job_DNN_l5_00000001.sh


# copy files over to location where test is being run
rsync -avP madisonData:/data/user/mhuennefeld/to_delete/unit_tests/dnn_reco_test_01 ./test_data/

# run test
python test.py
