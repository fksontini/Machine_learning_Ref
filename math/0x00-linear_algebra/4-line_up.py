#!/usr/bin/env python3
"""Sum arrays"""


def add_arrays(arr1, arr2):
    """ sum two arrays"""
    if len(arr1) != len(arr2):
        return None
    return [x + y for x, y in zip(arr1, arr2)]
