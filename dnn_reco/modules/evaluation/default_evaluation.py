"""
All defined evaluation functions must have the following signature:

    Parameters
    ----------
    feed_dict_train : dict
        The feed_dict used to feed tensorflow placeholder for the evaluation
        on training data.
    feed_dict_val : dict
        The feed_dict used to feed tensorflow placeholder for the evaluation
        on validation data.
    results_train : dict
        A dictionary with the results of the tensorflow operations for the
        training data.
    results_val : dict
        A dictionary with the results of the tensorflow operations for the
        validation data.
    config : dict
        Dictionary containing all settings as read in from config file.
    data_handler : :obj: of class DataHandler
        An instance of the DataHandler class. The object is used to obtain
        meta data.
    data_transformer : :obj: of class DataTransformer
        An instance of the DataTransformer class. The object is used to
        transform data.
    shared_objects : dict
        A dictionary containing settings and objects that are shared and passed
        on to sub modules.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
"""

from __future__ import division, print_function
import numpy as np

from dnn_reco.utils.angles import get_angle_deviation, get_angle


def eval_direction(
    feed_dict_train,
    feed_dict_val,
    results_train,
    results_val,
    config,
    data_handler,
    data_transformer,
    shared_objects,
    *args,
    **kwargs
):
    """Evaluation method to compute angular resolution.

    Parameters
    ----------
    feed_dict_train : dict
        The feed_dict used to feed tensorflow placeholder for the evaluation
        on training data.
    feed_dict_val : dict
        The feed_dict used to feed tensorflow placeholder for the evaluation
        on validation data.
    results_train : dict
        A dictionary with the results of the tensorflow operations for the
        training data.
    results_val : dict
        A dictionary with the results of the tensorflow operations for the
        validation data.
    config : dict
        Dictionary containing all settings as read in from config file.
    data_handler : :obj: of class DataHandler
        An instance of the DataHandler class. The object is used to obtain
        meta data.
    data_transformer : :obj: of class DataTransformer
        An instance of the DataTransformer class. The object is used to
        transform data.
    shared_objects : dict
        A dictionary containing settings and objects that are shared and passed
        on to sub modules.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    """
    y_true_train = feed_dict_train[shared_objects["y_true"]]
    y_true_val = feed_dict_val[shared_objects["y_true"]]

    y_pred_train = results_train["y_pred"]
    y_pred_val = results_val["y_pred"]
    y_unc_train = results_train["y_unc"]
    y_unc_val = results_val["y_unc"]

    index_azimuth = data_handler.get_label_index(config["label_azimuth_key"])
    index_zenith = data_handler.get_label_index(config["label_zenith_key"])

    angle_train = get_angle_deviation(
        azimuth1=y_true_train[:, index_azimuth],
        zenith1=y_true_train[:, index_zenith],
        azimuth2=y_pred_train[:, index_azimuth],
        zenith2=y_pred_train[:, index_zenith],
    )

    angle_val = get_angle_deviation(
        azimuth1=y_true_val[:, index_azimuth],
        zenith1=y_true_val[:, index_zenith],
        azimuth2=y_pred_val[:, index_azimuth],
        zenith2=y_pred_val[:, index_zenith],
    )

    print(
        "\t[Train]      Opening Angle: mean {:3.1f}, median {:3.1f}".format(
            np.mean(np.rad2deg(angle_train)),
            np.median(np.rad2deg(angle_train)),
        )
    )
    print(
        "\t[Validation] Opening Angle: mean {:3.1f}, median {:3.1f}".format(
            np.mean(np.rad2deg(angle_val)), np.median(np.rad2deg(angle_val))
        )
    )

    if "label_dir_x_key" in config and config["label_dir_x_key"] is not None:

        index_dir_x = data_handler.get_label_index(config["label_dir_x_key"])
        index_dir_y = data_handler.get_label_index(config["label_dir_y_key"])
        index_dir_z = data_handler.get_label_index(config["label_dir_z_key"])

        vec1_train = np.stack(
            [
                y_true_train[:, index_dir_x],
                y_true_train[:, index_dir_y],
                y_true_train[:, index_dir_z],
            ],
            axis=1,
        )
        vec2_train = np.stack(
            [
                y_pred_train[:, index_dir_x],
                y_pred_train[:, index_dir_y],
                y_pred_train[:, index_dir_z],
            ],
            axis=1,
        )
        vec1_val = np.stack(
            [
                y_true_val[:, index_dir_x],
                y_true_val[:, index_dir_y],
                y_true_val[:, index_dir_z],
            ],
            axis=1,
        )
        vec2_val = np.stack(
            [
                y_pred_val[:, index_dir_x],
                y_pred_val[:, index_dir_y],
                y_pred_val[:, index_dir_z],
            ],
            axis=1,
        )
        angle_dir_train = get_angle(vec1_train, vec2_train)
        angle_dir_val = get_angle(vec1_val, vec2_val)

        print(
            "\t[Train]      {}: mean {:3.1f}, median {:3.1f} [{}]".format(
                "Opening Angle",
                np.mean(np.rad2deg(angle_dir_train)),
                np.median(np.rad2deg(angle_dir_train)),
                "Over Direction Vector",
            )
        )
        print(
            "\t[Validation] {}: mean {:3.1f}, median {:3.1f} [{}]".format(
                "Opening Angle",
                np.mean(np.rad2deg(angle_dir_val)),
                np.median(np.rad2deg(angle_dir_val)),
                "Over Direction Vector",
            )
        )

        # Test weighted correction: [very much beta version]
        def get_weighted_normed_dir_vector(
            dir_x, dir_y, dir_z, dir_x_unc, dir_y_unc, dir_z_unc
        ):

            # get direction of normalization scaling
            d1 = dir_x * dir_x_unc
            d2 = dir_y * dir_y_unc
            d3 = dir_z * dir_z_unc
            norm = np.sqrt(d1**2 + d2**2 + d3**2)
            d1 /= norm
            d2 /= norm
            d3 /= norm

            # calculate scaling length
            p = d1 * dir_x + d2 * dir_y + d3 * dir_z
            q = dir_x**2 + dir_y**2 + dir_z**2 - 1
            L = -p + np.sqrt(p**2 - q)

            # calculate normed direction vector
            dir_x_new = L * d1 + dir_x
            dir_y_new = L * d2 + dir_y
            dir_z_new = L * d3 + dir_z

            # print(np.sqrt(dir_x_new**2 + dir_y_new**2 + dir_z_new**2))

            return dir_x_new, dir_y_new, dir_z_new

        test_dir_vector_scaling = False
        if test_dir_vector_scaling:

            vec2_train = np.stack(
                get_weighted_normed_dir_vector(
                    dir_x=y_pred_train[:, index_dir_x],
                    dir_y=y_pred_train[:, index_dir_y],
                    dir_z=y_pred_train[:, index_dir_z],
                    dir_x_unc=y_unc_train[:, index_dir_x],
                    dir_y_unc=y_unc_train[:, index_dir_y],
                    dir_z_unc=y_unc_train[:, index_dir_z],
                ),
                axis=1,
            )
            vec2_val = np.stack(
                get_weighted_normed_dir_vector(
                    dir_x=y_pred_val[:, index_dir_x],
                    dir_y=y_pred_val[:, index_dir_y],
                    dir_z=y_pred_val[:, index_dir_z],
                    dir_x_unc=y_unc_val[:, index_dir_x],
                    dir_y_unc=y_unc_val[:, index_dir_y],
                    dir_z_unc=y_unc_val[:, index_dir_z],
                ),
                axis=1,
            )

            angle_dir_train = get_angle(vec1_train, vec2_train)
            angle_dir_val = get_angle(vec1_val, vec2_val)

            print(
                "\t[Train]      {}: mean {:3.1f}, median {:3.1f} [{}]".format(
                    "Opening Angle",
                    np.mean(np.rad2deg(angle_dir_train)),
                    np.median(np.rad2deg(angle_dir_train)),
                    "Weighted dir vector",
                )
            )
            print(
                "\t[Validation] {}: mean {:3.1f}, median {:3.1f} [{}]".format(
                    "Opening Angle",
                    np.mean(np.rad2deg(angle_dir_val)),
                    np.median(np.rad2deg(angle_dir_val)),
                    "Weighted dir vector",
                )
            )
