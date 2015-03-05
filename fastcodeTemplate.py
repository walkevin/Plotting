#!/usr/bin/env python
'''
Created on 25.02.2015
Modified on 05.03.2015

@author: kevin
'''

import argparse
import math
import random
import re

import matplotlib.pyplot as plt
import numpy as np

# Text
#_xlabel = "n"
#_ylabel = "[Gflop / s]"
#_title = "DFT $2^n$ (single precision) on Pentium 4, 2.53 GHz"

# Define available line colors
colors = [
          '#ff8080', # Red
          '#8080ff', # Blue
          '#408040', # Green
          '#ff80ff', # Pink
          '#60b0b0', # Cyan
          '#b0b060', # Olive
         ]

def generate_graph(ifpath, ofpath, title, ylabel):
    # Load some fake data.
    data = np.loadtxt(ifpath, skiprows=1)

    X = data[:, 0]
    Xmin = np.min(X)
    Xmax = np.max(X)
    print 'x in [%f, %f]' % (Xmin, Xmax)

    Y = data[:, 1:]
    Ymin = np.min(Y)
    Ymax = np.max(Y)
    print 'y in [%f, %f]' % (Ymin, Ymax)
    (M, N) = Y.shape

    #n = data[:,0]
    #a = data[:,1]
    #b = data[:,2]
    #c = data[:,3]
    #d = data[:,4]

    # Parse labels
    with open(ifpath, 'r') as f:
      labels = f.readline().strip().split("\t")
      xlabel = labels[0]
      ylabels = labels[1:]

    # Define default values
    plt.rc('lines', linewidth=2)
    plt.rc('font', family=['Helvetica', 'sans-serif'])
    #plt.rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica']})

    # Create figure
    fig = plt.figure(facecolor = 'white')
    ax1 = plt.axes(axisbg = '#f0f0f0')
    plt.subplots_adjust(left=0.07, right=0.96, top=0.89, bottom=0.1)

    # Create plots with pre-defined labels.
    for i in range(N):
        ax1.plot(X, Y[:, i], colors[i],
            marker='o',
            markeredgecolor=colors[i],
            label=ylabels[i],
        )

    # Set title
    ax1.set_title(label = title,
        size = 'large',
        weight = 'heavy',
        fontstretch = 'semi-condensed',
        horizontalalignment = 'left',
        verticalalignment = 'bottom',
        position = (0.06, 1.03),
    )

    # Set labels
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, rotation = 0,
        color = '#555555',
        size = 'small',
        horizontalalignment='left',
        verticalalignment='bottom',
    )
    ax1.yaxis.set_label_coords(0.06, 1.01)

    ## Set legend
    #box = ax1.get_position()
    #ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #ax1.legend(loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))

    ## Add annotations automatically. A 2-pass procedure.
    # Pass 1: Invalidate areas where data points are
    M_g = 10
    N_g = 9
    chk_grid = np.zeros((M_g + 1, N_g + 1)) # Grid
    dY = (Ymax - Ymin) / M_g
    _Ymin = Ymin - dY
    _Ymax = Ymax + dY
    plt.ylim((_Ymin, _Ymax)) # Set Y limits manually for boring or single-data plots
    dY = (_Ymax - _Ymin) / M_g
    dX = (Xmax - Xmin) / N_g
    for (j, _), y in np.ndenumerate(Y):
        x = X[j]
        chk_grid[round((y-_Ymin)/dY), round((x-Xmin)/dX)] += 1

    # Pass 2: Pick valid area data point at random
    excl_Nx = int(M * .20) # Exclude 20% of boundary points to avoid label
                           # cutoff due to being on graph edge
    for i in range(N):
        ys = Y[excl_Nx:-excl_Nx, i]
        grads = np.gradient(ys)

        # Stochastic selection of data point to label
        while True:
            #print chk_grid

            j = random.randint(0, len(ys) - 1)  # Pick random point
            y = ys[j]
            x = X[j + excl_Nx]                  # We excluded boundary points,
                                                # so fix indexing

            d = grads[j]                    # Gradient at point
            s = 1 if d == 0 else np.sign(d) # Sign of gradient at point

            d_ = math.sqrt(abs(d)) * .3
            dsx = -s * (.2*dX)
            dsy =  s * (.35*dY + d_)
            #print d, x, y, dsx, dsy, dX, dY

            gj1 = round((y - _Ymin) / dY)
            gi1 = round((x - Xmin) / dX)
            gj2 = round((y + dsy - _Ymin) / dY)
            gi2 = round((x + dsx - Xmin) / dX)
            gj3 = round((y - dsy - _Ymin) / dY)
            gi3 = round((x - dsx - Xmin) / dX)
            try: # Skip in case out-of-bounds error for indexing grid
                if gj1 >= 0 and gi1 >= 0 and gj2 >= 0 and gi2 >= 0 and \
                   (chk_grid[gj1, gi1] < 2) and (chk_grid[gj2, gj2] < 1):
                    # Inc occupation count for self and 8 neighbours
                    chk_grid[gj2, gi2] += 1
                    chk_grid[gj2 + 1, gi2] += 1
                    chk_grid[gj2 - 1, gi2] += 1
                    chk_grid[gj2, gi2 + 1] += 1
                    chk_grid[gj2, gi2 - 1] += 1
                    chk_grid[gj2 + 1, gi2 + 1] += 1
                    chk_grid[gj2 - 1, gi2 + 1] += 1
                    chk_grid[gj2 + 1, gi2 - 1] += 1
                    chk_grid[gj2 - 1, gi2 - 1] += 1
                    y += dsy
                    x += dsx
                    break
            except: pass

            try:
                if gj1 >= 0 and gi1 >= 0 and gj3 >= 0 and gi3 >= 0 and \
                   (chk_grid[gj1, gi1] < 2) and (chk_grid[gj3, gi3] < 1):
                    chk_grid[gj3, gi3] += 1
                    chk_grid[gj3 + 1, gi3] += 1
                    chk_grid[gj3 - 1, gi3] += 1
                    chk_grid[gj3, gi3 + 1] += 1
                    chk_grid[gj3, gi3 - 1] += 1
                    chk_grid[gj3 + 1, gi3 + 1] += 1
                    chk_grid[gj3 - 1, gi3 + 1] += 1
                    chk_grid[gj3 + 1, gi3 - 1] += 1
                    chk_grid[gj3 - 1, gi3 - 1] += 1
                    y -= dsy
                    x -= dsx
                    break
            except: pass


        print 'Placing label at (%f, %f) for "%s"' % (x, y, ylabels[i])
        ax1.annotate(ylabels[i],
            xy = (x, y),
            xytext = (x, y),
            color = colors[i],
            horizontalalignment = 'center',
            verticalalignment = 'center',
        )

    ## Alternatively, set annotations (by hand)
    #ax1.annotate(ylabels[0], xy=(n[5], a[5]), xytext=(n[5], a[5]+0.3), color=colors[0])
    #ax1.annotate(ylabels[1], xy=(n[6], b[6]), xytext=(n[6], b[6]+0.3), color=colors[1])
    #ax1.annotate(ylabels[2], xy=(n[6], c[6]), xytext=(n[6], c[6]-0.4), color=colors[2])
    #ax1.annotate(ylabels[3], xy=(n[4], d[4]), xytext=(n[4]-2, d[4]), color=colors[3])

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
        right = 'off',
    )

    # Hide top ticks for x-axis
    ax1.tick_params(
        axis = 'x',
        which = 'both',
        top = 'off',
        bottom = 'on',
        direction = 'out',
    )

    # Add 2% margin on x axis
    ax1.margins(x = 0.02)

    # Save the plot
    plt.savefig(ofpath)

    #plt.show()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate graph formatted for HTWFNC')

    parser.add_argument('input-file',
                       help='Input data file, header (first row) expected. ' +
                       'Entries on a row are delimited by tabstops, and the ' +
                       'first column is X-axis data')
    parser.add_argument('-t', '--title', help='Title label')
    parser.add_argument('-y', '--ylabel', help='Y-axis label')
    parser.add_argument('-o', '--output', help='Output file path')

    args = vars(parser.parse_args())

    # Set default output file path and title
    ifpath = args['input-file']
    ofpath = args['output'] if args['output'] else re.sub(r'\.[^\.]*$', '.eps', ifpath)
    title = args['title'] if args['title'] else re.sub(r'\..*$', '', ifpath)
    ylabel = args['ylabel'] if args['ylabel'] else '[Gflop / s]'

    # Plot graph and save output
    generate_graph(ifpath, ofpath, title, ylabel)


# Run main function if run as main program
if __name__ == '__main__':
    main()

