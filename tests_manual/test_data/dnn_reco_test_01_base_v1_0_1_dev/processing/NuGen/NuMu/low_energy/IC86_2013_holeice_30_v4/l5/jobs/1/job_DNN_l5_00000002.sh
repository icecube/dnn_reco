#!/bin/bash
# Shell-script wrapper to execute indidividual steps

# Start Timer
# Note: SECONDS is a bash special variable that returns the seconds
# since it was set.
SECONDS=0

# Create array and index counter to keep track of time per-step
declare -a times
times[0]=$SECONDS
step_counter=1
OUT_DIR=/data/user/mhuennefeld/software/repositories/dnn_recoV2compatV1/tests_manual/test_data/dnn_reco_test_01/processing/NuGen/NuMu/low_energy/IC86_2013_holeice_30_v4/l5/jobs/1/steps_DNN_l5_00000002

echo &&
echo '=========================' &&
echo '==> Starting step 000 <==' &&
echo '=========================' &&
echo &&
eval "${OUT_DIR}/job_DNN_l5_00000002_step_0000.sh" && 
times[$step_counter]=$SECONDS &&
((step_counter++)) &&

echo &&
echo '=========================' &&
echo '==> Starting step 001 <==' &&
echo '=========================' &&
echo &&
eval "${OUT_DIR}/job_DNN_l5_00000002_step_0001.sh" && 
times[$step_counter]=$SECONDS &&
((step_counter++)) &&


        echo &&
        echo '=========================================' &&
        echo '==> Successfully processed all steps! <==' &&
        echo '========================================='

        RET=$?
        if [ $RET -ne 0 ] ; then
           echo
           echo '======================='
           echo '==> Error occurred! <=='
           echo '======================='
           echo
        fi

        echo
        echo 'Runtime for each processed step:'

        for (( i=1; i<$step_counter; i++ ))
        do
          printf "    %-13s --> " "Step $i"
          printf "%+8u s
" $(( ${times[$i]} - ${times[$i-1]}))
        done
        printf "    %13s --> %+8u s
" "Total runtime" $SECONDS
        echo

        echo Cleaning up intermediate files ...
echo '   ... removing ./temp_step_files/NuGen/NuMu/low_energy/IC86_2013_holeice_30_v4/l5/1/DNN_l5_00000002_step0000*'
rm ./temp_step_files/NuGen/NuMu/low_energy/IC86_2013_holeice_30_v4/l5/1/DNN_l5_00000002_step0000*
echo '   ... checking if temp_step_files exist'
if [ -d temp_step_files ]; then
   echo '   ... checking if temp_step_files is empty'
   lines=$(find temp_step_files/ -type f | wc -l)
   if [ $lines -eq 0 ]; then
      echo '   ... removing directory temp_step_files'
      rm -r temp_step_files
   else
      echo '   ...    Not deleting directory as it still    contains files.'
   fi
else
   echo '   ...    temp_step_files does not exist'
fi
exit $RET
