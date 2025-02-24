"""The YAML loader and dumper for the dnn_reco package.

This module defines the YAML loader and dumper for the dnn_reco package. It
also registers all classes that can be loaded from YAML files.
"""

from copy import deepcopy
from ruamel.yaml import YAML

# from ruamel.yaml.representer import RoundTripRepresenter
# from tensorflow.python.trackable.data_structures import ListWrapper

from dnn_reco.misc import load_class


def convert_nested_list_wrapper(data):
    """Converts nested ListWrapper objects to lists.

    Parameters
    ----------
    data : list | dict
        The (possibly nested) data structure for which to convert
        any ListWrapper objects to lists.

    Returns
    -------
    list | dict
        The data structure with any ListWrapper objects converted to lists.
    """
    data = deepcopy(data)

    # recursion stop condition
    if not isinstance(data, (list, tuple, dict)):
        return data

    if isinstance(data, (list, tuple)):
        return [convert_nested_list_wrapper(item) for item in data]
    else:
        return {
            key: convert_nested_list_wrapper(value)
            for key, value in data.items()
        }


REGISTERTED_CLASSES = []

# # Create a subclass of RoundTripRepresenter to handle ListWrapper objects
# class ListWrapperRepresenter(RoundTripRepresenter):
#     pass

# def represent_list_wrapper(dumper, data):
#     return dumper.represent_list(convert_nested_list_wrapper(data))

# # ListWrapperRepresenter.add_representer(ListWrapper, represent_list_wrapper)
# ListWrapperRepresenter.add_representer(list, represent_list_wrapper)
# ListWrapperRepresenter.add_representer(tuple, represent_list_wrapper)
# ListWrapperRepresenter.add_representer(dict, represent_list_wrapper)


# define yaml dumper
yaml_dumper = YAML(typ="safe")
yaml_dumper.default_flow_style = False
# yaml_dumper.Representer = ListWrapperRepresenter

# define yaml loader and register all classes
yaml_loader = YAML(typ="safe", pure=True)
# yaml_loader.add_constructor(
#     "tag:yaml.org,2002:python/unicode", lambda _, node: node.value
# )

for class_name in REGISTERTED_CLASSES:
    yaml_loader.register_class(load_class(class_name))
    yaml_dumper.register_class(load_class(class_name))
