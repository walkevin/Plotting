#!/usr/bin/env python
'''
Created on 25.02.2015

@author: kevin
'''

import matplotlib.pyplot as plt
import numpy as np
# Text
_xlabel = "n"
_ylabel = "[Gflop / s]"
_title = "DFT $2^n$ (single precision) on Pentium 4, 2.53 GHz"

# Load some fake data.
data = np.loadtxt('sampledata.txt', skiprows=1)
n = data[:,0]
a = data[:,1]
b = data[:,2]
c = data[:,3]
d = data[:,4]
labels = ['Spiral SSE', 'Spiral vectorized', 'Spiral scalar', 'Intel MKL']
# Define colors
colors = ['#ff8080', '#8080ff', '#408040', '#ff80ff', '#60b0b0', '#b0b060']

# Define default values
plt.rc('lines', linewidth=2)
#plt.rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica']})

# Create figure
fig = plt.figure(facecolor = 'white')
ax1 = plt.axes(axisbg = '#f0f0f0')

# Create plots with pre-defined labels.
ax1.plot(n, a, colors[0], marker='o', markeredgecolor=colors[0], label=labels[0])
ax1.plot(n, b, colors[1], marker='o', markeredgecolor=colors[1], label=labels[1])
ax1.plot(n, c, colors[2], marker='o', markeredgecolor=colors[2], label=labels[2])
ax1.plot(n, d, colors[3], marker='o', markeredgecolor=colors[3], label=labels[3])

# Set title
ax1.set_title(label = _title)

# Set labels
ax1.set_ylabel(_ylabel, rotation = 0, horizontalalignment = 'right')
ax1.set_xlabel(_xlabel)

# Set legend
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax1.legend(loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))

# Alternatively, set annotations (by hand)
ax1.annotate(labels[0], xy=(n[5], a[5]), xytext=(n[5], a[5]+0.3), color=colors[0])
ax1.annotate(labels[1], xy=(n[6], b[6]), xytext=(n[6], b[6]+0.3), color=colors[1])
ax1.annotate(labels[2], xy=(n[6], c[6]), xytext=(n[6], c[6]-0.4), color=colors[2])
ax1.annotate(labels[3], xy=(n[4], d[4]), xytext=(n[4]-2, d[4]), color=colors[3])

# Turn off spines except for x-axis
ax1.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)

# Insert horizontal grid
ax1.grid(True, which = 'both', axis = 'y', color = 'w', linewidth=2, linestyle = '-')
ax1.set_axisbelow(True)

# Hide ticks for y-axis
ax1.tick_params(
    axis = 'y',
    which = 'both',
    left = 'off',
    right = 'off'            
    )

# Hide top ticks for x-axis
ax1.tick_params(
    axis = 'x',
    which = 'both',
    top = 'off',
    bottom = 'on',
    direction = 'out'         
    )

# Add 2% margin on x axis
ax1.margins(x = 0.02)

# Save the plot
plt.savefig('sampleplot.eps')

plt.show()
