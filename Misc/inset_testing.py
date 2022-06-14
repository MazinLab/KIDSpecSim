# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:46:34 2020

@author: BVH
"""

from parameters import object_file
from useful_funcs import data_extractor
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


spec_wavelengths,data_cube = data_extractor(object_file,XShooter=False,II_D=True,fits_f=True,plotting=False)

extent = (0,500,0,250)

fig, ax = plt.subplots(figsize=[5,4])

ax.imshow(data_cube[650,:,:],extent=extent,interpolation=None)

axins = zoomed_inset_axes(ax, 6, loc=1)
axins.imshow(data_cube[650,:,:],extent=extent,interpolation=None)

x1, x2, y1, y2 = 240,260,120,130
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

plt.xticks(visible=False)
plt.yticks(visible=False)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

axins = zoomed_inset_axes(ax, 6, loc=3)
axins.imshow(data_cube[650,:,:],extent=extent,interpolation=None)

x1, x2, y1, y2 = 240,260,120,130
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

plt.xticks(visible=False)
plt.yticks(visible=False)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
