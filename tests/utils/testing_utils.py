from __future__ import annotations

import inspect
import sys


def get_scenes_to_test(module_name: str):
    """Get all Test classes of the module from which it is called. Used to fetch all the SceneTest of the module.

    Parameters
    ----------
    module_name
        The name of the module tested.

    Returns
    -------
    List[:class:`type`]
        The list of all the classes of the module.
    """
    return inspect.getmembers(
        sys.modules[module_name],
        lambda m: inspect.isclass(m) and m.__module__ == module_name,
    )
