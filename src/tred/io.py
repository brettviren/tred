#!/usr/bin/env python
'''
Functions for file I/O
'''

from .blocking import Block

import numpy
def write_npz(path, **tensors_or_blocks):
    '''
    Write tensors or blocks to .npz file at path.

    Tensors are saved with keyword arg name.  Blocks are saved as individual
    <name>_location and <name>_data arrays.

    All tensors are converted to numpy arrays in memory prior to writing.
    '''
    arrays = dict()
    for key, val in tensors_or_blocks.items():
        if isinstance(val, Block):
            arrays[f'{key}_location'] = val.location
            arrays[f'{key}_data'] = val.data
        else:
            arrays[key] = val
    numpy.savez_compressed(path, **arrays)

