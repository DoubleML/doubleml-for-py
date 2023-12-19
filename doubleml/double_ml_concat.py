from .double_ml_framework import DoubleMLFramework


def concat(objs):
    """
    Concatenate DoubleMLFramework objects.
    """
    if len(objs) == 0:
        raise ValueError('Need at least one object to concatenate.')

    if not all(isinstance(obj, DoubleMLFramework) for obj in objs):
        raise ValueError('All objects must be of type DoubleMLFramework.')

    # TODO: Add more Input checks
    new_obj = DoubleMLFramework(

    )
    return new_obj
