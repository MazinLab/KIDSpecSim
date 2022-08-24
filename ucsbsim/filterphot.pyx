cimport cython
cimport numpy as np
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def mask_deadtime(times, deadtime):
    n = times.shape[0]
    keep = np.ones(n, bool)
    lim = times[0] + deadtime
    i=1
    while i<n:
        if times[i]<lim:
            keep[i] = False
        else:
            lim = times[i] + deadtime
        i+=1
    return keep
