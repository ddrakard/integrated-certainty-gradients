"""
    Utilities relating to the Python language.
"""

from typing import List, Dict, Callable


def call_method(
        method_name: str, positional_arguments: List = [],
        keyword_arguments: Dict = {}) -> Callable:
    """
        Return a function that calls the given method on its argument, with
        the given arguments.
    """
    return lambda object: getattr(object, method_name)(
        *positional_arguments, **keyword_arguments)
