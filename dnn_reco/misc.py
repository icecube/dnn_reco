import importlib


def print_warning(msg):
    """Print Warning in yellow color.

    Parameters
    ----------
    msg : str
        String to print.
    """
    print("\033[93m" + msg + "\033[0m")


def load_class(full_class_string):
    """
    dynamically load a class from a string

    Parameters
    ----------
    full_class_string : str
        The full class string to the given python clas.
        Example:
            my_project.my_module.my_class

    Returns
    -------
    python class
        PYthon class defined by the 'full_class_string'
    """

    class_data = full_class_string.split(".")
    module_path = ".".join(class_data[:-1])
    class_str = class_data[-1]

    module = importlib.import_module(module_path)
    # Finally, we retrieve the Class
    return getattr(module, class_str)


def get_full_class_string_of_object(object_instance):
    """Get full class string of an object's class.

    o.__module__ + "." + o.__class__.__qualname__ is an example in
    this context of H.L. Mencken's "neat, plausible, and wrong."
    Python makes no guarantees as to whether the __module__ special
    attribute is defined, so we take a more circumspect approach.
    Alas, the module name is explicitly excluded from __qualname__
    in Python 3.

    Adopted from:
        https://stackoverflow.com/questions/2020014/
        get-fully-qualified-class-name-of-an-object-in-python

    Parameters
    ----------
    object_instance : object
        The object of which to obtain the full class string.

    Returns
    -------
    str
        The full class string of the object's class
    """
    module = object_instance.__class__.__module__
    if module is None or module == str.__class__.__module__:
        # Avoid reporting __builtin__
        return object_instance.__class__.__name__
    else:
        return module + "." + object_instance.__class__.__name__
