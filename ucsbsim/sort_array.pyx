cimport cython
cimport numpy as np
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def sort(array, sorting_index):
    return np.take_along_axis(array, sorting_index, axis=0)
