from matplotlib import pyplot
import matplotlib as mpl

# Make a figure and axes with dimensions as desired.
fig = pyplot.figure()
# vertical
# ax2 = fig.add_axes([0.1, 0.1, 0.02, 0.8])
# horizontal
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.04])



# The second example illustrates the use of a ListedColormap, a
# BoundaryNorm, and extended ends to show the "over" and "under"
# value colors.
cmap = mpl.colors.ListedColormap(['#4575b4','#74add1','#abd9e9','#e0f3f8','#ffffbf','#fee090','#fdae61','#f46d43','#d73027'])
cmap.set_over('#a50026')
cmap.set_under('#313695')

# If a ListedColormap is used, the length of the bounds array must be
# one greater than the length of the color list.  The bounds must be
# monotonically increasing.

# # soybean
# bounds = [15,20,25,30,35,40,45,50,55,60]
# corn
bounds = [20,40,60,80,100,120,140,160,180,200]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                norm=norm,
                                # to use 'extend', you must
                                # specify two extra boundaries:
                                boundaries=[0] + bounds + [220],
                                extend='both',
                                ticks=bounds,  # optional
                                spacing='proportional',
                                orientation='horizontal')



pyplot.show()