import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime as dt
import logging

"""
Bringing it all together. This is the stage where we bring together:
- The observation spectrum separated into virtual pixels and the errors on each
- The wavecal solution from the emission spectrum
to recover a final spectrum.
This script may also:
- fit an empiric blaze function to be divided out

"""