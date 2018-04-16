"""This module provides additional utilities, such as common functions

those utilities do not have any logical depencencies on other parts
of the project.

"""


def truncate(arg, lower=0, upper=1):
    """This function returns value truncated within range <lower, upper>

    Args:
        arg (number) - value to be truncated
        lower (number) - lower truncating bound, default to 0
        upper (number) - upper truncating bound, default to 1

    Returns:
        arg (number) - truncated function argument
    """

    if arg > upper:
        return upper
    if arg < lower:
        return lower
    return arg
