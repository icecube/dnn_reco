from __future__ import print_function, division
import os
import glob
import pandas as pd
import numpy as np


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def warning(msg):
    print(bcolors.WARNING + msg + bcolors.ENDC)


def error(msg):
    print(bcolors.FAIL + msg + bcolors.ENDC)


files = [
    "NuGen/NuE/low_energy/IC86_2013_holeice_30_v4/l5/1/DNN_l5_00000002.hdf5",
    "NuGen/NuMu/low_energy/IC86_2013_holeice_30_v4/l5/1/DNN_l5_00000002.hdf5",
    "NuGen/NuTau/low_energy/IC86_2013_holeice_30_v4/l5/1/DNN_l5_00000002.hdf5",
    "NuGen/NuE/medium_energy/IC86_2013_holeice_30_v4/l5/1/DNN_l5_00000001.hdf5",
    "NuGen/NuMu/medium_energy/IC86_2013_holeice_30_v4/l5/1/DNN_l5_00000001.hdf5",
    "NuGen/NuTau/medium_energy/IC86_2013_holeice_30_v4/l5/1/DNN_l5_00000001.hdf5",
]

keys = [
    # sanity checks
    "I3EventHeader",
    # ic3-data input data to dnn_reco
    "dnn_data__charge_bins_bin_values",
    "dnn_data__charge_bins_bin_indices",
    "dnn_data__charge_bins_bin_exclusions",
    "dnn_data__charge_bins_global_time_offset",
    "dnn_data_inputs3_InIceDSTPulses_bin_values",
    "dnn_data_inputs3_InIceDSTPulses_bin_indices",
    "dnn_data_inputs3_InIceDSTPulses_global_time_offset",
    "dnn_data_inputs9_InIceDSTPulses_bin_values",
    "dnn_data_inputs9_InIceDSTPulses_bin_indices",
    "dnn_data_inputs9_InIceDSTPulses_global_time_offset",
    # ic3-labels labels
    "LabelsDeepLearning",
    "LabelsMCCascade",
    "MCCascade",
    # dnn_reco results
    "DeepLearningReco_event_selection_cscdl3_300m_01",
    "DeepLearningReco_event_selection_cascade_monopod_starting_events_big_kernel_02",
    "DeepLearningReco_mese_v2__all_gl_both2",
    "DeepLearningReco_dnn_reco_paper_hese__m7_after_sys",
]

dir_original = "test_data/dnn_reco_test_01_base_v1_0_1_dev"
test_dirs = glob.glob("test_data/*")
test_dirs.remove(dir_original)

if len(test_dirs) == 0:
    raise ValueError("No test directories found!")

got_warning = False
passed_test = True
for dir_test in test_dirs:
    print("\nNow testing {!r} against {!r}".format(dir_test, dir_original))
    for file_name in files:
        print("\n\tNow testing {!r}".format(file_name))
        for key in keys:
            try:
                df_original = pd.read_hdf(
                    os.path.join(dir_original, file_name), key=key
                )
                df_test = pd.read_hdf(
                    os.path.join(dir_test, file_name), key=key
                )
            except Exception as e:
                warning("\t\tProblem with key {!r}".format(key))
                warning("\t\t", e)
                got_warning = True

            assert (df_original.columns == df_test.columns).all()
            for k in df_original.columns:
                if "runtime" not in k:
                    if not np.allclose(
                        df_original[k].values,
                        df_test[k].values,
                        atol=5e-6,
                        rtol=5e-4,
                    ):
                        if key == "LabelsDeepLearning":
                            warning("\t\tWarning: mismatch for {}".format(k))
                            got_warning = True
                        else:
                            error("\t\tError: mismatch for {}".format(k))
                            passed_test = False
                        print(
                            "\t\t",
                            key,
                            k,
                            (df_original[k].values - df_test[k].values),
                        )
                        print("\t\t", df_original[k].values)
                        print("\t\t", df_test[k].values)
                else:
                    runtime_orig = np.mean(df_original[k].values) * 1000.0
                    runtime_orig_std = np.std(df_original[k].values) * 1000.0
                    runtime_test = np.mean(df_test[k].values) * 1000.0
                    runtime_test_std = np.std(df_test[k].values) * 1000.0
                    max_dev = max(2 * runtime_orig_std, 0.5 * runtime_orig)
                    if np.abs(runtime_orig - runtime_test) > max_dev:
                        msg = "\t\t Runtimes: {:3.3f} +- {:3.3f}ms [base] "
                        msg += "{:3.3f} +- {:3.3f}ms [test]"
                        print(
                            msg.format(
                                runtime_orig,
                                runtime_orig_std,
                                runtime_test,
                                runtime_test_std,
                            )
                        )

print("\n====================")
print("=== Summary ========")
print("====================")
if got_warning:
    print(
        "=== "
        + bcolors.WARNING
        + "Warnings: {}".format(got_warning)
        + bcolors.ENDC
    )
else:
    print(
        "==="
        + bcolors.OKGREEN
        + " Warnings: {}".format(got_warning)
        + bcolors.ENDC
    )
if passed_test:
    print(
        "==="
        + bcolors.OKGREEN
        + " Passed:   {}".format(passed_test)
        + bcolors.ENDC
    )
else:
    print(
        "==="
        + bcolors.FAIL
        + " Passed:   {}".format(passed_test)
        + bcolors.ENDC
    )
print("====================\n")
