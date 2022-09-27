#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z Boson: Intro To Programming Assingment

This code reads in and validates measurements of the Z boson. It performs 2D
analysis to find the minimum chi squared which, in turn, finds the mass and
width of the boson. These values are then used to caluclate the lifetime of the
particle. It also creates plots a graph of the data and a contour of the chi
squared.

@author: ellishuntley
"""




import sys
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt
import scipy.constants as pc


def validate_data_1():
    """
    Removes invalid data from the first data file and formats it.

    Returns
    -------
    Array of floats

    """
    while True:
        try:
            input_file_1 = np.genfromtxt('z_boson_data_1.csv', delimiter=',')
            input_file_1 = (input_file_1[~np.isnan(input_file_1).any(axis=1),\
                                         :]) #removes any rows with nans
            input_file_1 = input_file_1[np.where(input_file_1[:, 2] > 0)]\
                #removes any incorrect errors
            return input_file_1[np.where(input_file_1[:, 1] > 0)]\
                #returns\ data after removing any unphysical data
        except IOError:
            print('File 1 Not Found')
            sys.exit() #prints the statement and exits the full code

def validate_data_2():
    """
    Removes invalid data from the second data file and formats it.

    Returns
    -------
    Array of floats

    """
    while True:
        try:
            input_file_2 = np.genfromtxt('z_boson_data_2.csv', delimiter=',')
            input_file_2 = (input_file_2[~np.isnan(input_file_2).any(axis=1),\
                                         :]) #removes any rows with nans
            input_file_2 = input_file_2[np.where(input_file_2[:, 2] > 0)]\
                #removes any incorrect errors
            return input_file_2[np.where(input_file_2[:, 1] > 0)]\
                #removes any unphysical data
        except IOError:
            print('File 2 Not Found')
            sys.exit()

def data_combine(data_1, data_2):
    """
    Combines the validated data into one array

    Parameters
    ----------
    data_1 : validated boson data 1, array
    data_2 : valdated boson data 2, array

    Returns
    -------
    Array

    """
    data = np.vstack((data_1, data_2))
    return data

def remove_outliers(data):
    """
    Removes any rows of data containing outliers

    Parameters
    ----------
    data : 2d numpy array

    Returns
    -------
    2d numpy array

    """
    mean = np.mean(data[:, 1])
    standard_deviation = np.std(data[:, 1])
    data = (data[np.where(np.abs(data[:, 1] - mean) < (3 * \
                                                       standard_deviation))])
    return data

def sort_data(data):
    """
    Sorts the data

    Returns
    -------
    data

    """
    data = data[data[:, 0].argsort()]
    return data

def plot_data(data, values):
    """
    Creates a plot of Energy against Cross section

    Parameters
    ----------
    data : 2d array of nmupy floats


    Returns
    -------
    None.

    """


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Plot of Energy against Cross Section')
    ax.set_xlabel('Energy (GeV)')
    ax.set_ylabel('Cross Section (nb)')

    ax.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], c='mediumseagreen',\
                fmt='.', label='Value Uncertainity', zorder=1)

    ax.scatter(data[:, 0], data[:, 1], marker='x', c='royalblue', label=\
               'Measured Values', zorder=2)

    energies_for_curve = np.linspace(min(data[:, 0]), max(data[:, 0]), 1000)

    ax.plot(energies_for_curve, cross_section_curve(energies_for_curve, \
                                                    values),\
            label='Function Curve', zorder=3, c='black')
    ax.grid()
    ax.legend()
    plt.savefig('cross_section_energy_graph.png', dpi=300, bbox_inches='tight')
    plt.show()



def cross_section_curve(energy, mass_z_and_gamma_z):
    """
    Calculates the cross section using given values

    Returns
    -------
    Array of floats

    """

    gamma_ee = 83.91e-3
    mass_z = mass_z_and_gamma_z[0]
    gamma_z = mass_z_and_gamma_z[1]

    cross_section = 12*np.pi / mass_z**2 * energy**2 / \
        ((energy**2 - mass_z**2)**2 + mass_z**2 * gamma_z**2) * gamma_ee**2

    cross_section = cross_section * (0.3894 * 10**6) #converts to correct units

    return cross_section

def chi_square(data, prediction):
    """
    calculates the chi squared using given value

    Returns
    -------
    chi square, float

    """
    observation = data[:, 1]
    error = data[:, 2]

    return np.sum((observation - prediction)**2 / error**2)


def minimisation(data):
    """

    Uses fmin to find the minimum chi squared


    Returns
    -------
    min chi squared value, Z boson mass and width (floats)

    """
    values_0 = fmin(lambda mass_z_and_gamma_z: \
                    chi_square(data, cross_section_curve(data[:, 0],\
                                                         mass_z_and_gamma_z)),\
                        (90, 3))

    data = data[np.where(np.abs(data[:, 1] - \
                                cross_section_curve(data[:, 0], \
                                                    values_0)) <\
                         (3 * data[:, 2]))]\
        #filters the data again using the new mass and width values

    values = fmin(lambda mass_z_and_gamma_z\
                  : chi_square(data, cross_section_curve(data[:, 0],\
                                                         mass_z_and_gamma_z)),\
                      values_0) #run fmin again using new data just filtered



    new_prediction = cross_section_curve(data[:, 0], values)

    min_chi_squared = chi_square(data, new_prediction)

    reduced_chi_squared = min_chi_squared / (len(data) - 2)

    print('The minimised chi squared is {0:2.3f} and the reduced chi square \
is {1:2.3f}.'.format(min_chi_squared, reduced_chi_squared))

    return data, values

def lifetime_calculator(values):
    """
    Calculates the lifetime of the Z boson.

    Parameters
    ----------
    values: tuple containing boson mass and width


    Returns
    -------
    lifetime, float

    """
    width = values[1]
    lifetime = (pc.hbar  / (1.602e-19*1e9) / width) \
        #calculates lifetime in correct units

    return lifetime

def array_mesh():
    """
    generates value used for contour and creates meshes

    Returns
    -------
    arrays

    """
    mass_values = np.linspace(91.145, 91.21, 500)
    width_values = np.linspace(2.4786, 2.54, 500)



    mass_array_mesh = np.empty((0, len(mass_values)))

    for _ in width_values:
        mass_array_mesh = np.vstack((mass_array_mesh, mass_values))

    width_array_mesh = np.empty((0, len(width_values)))

    for dummy_element in mass_values:
        width_array_mesh = np.vstack((width_array_mesh, width_values))

    width_array_mesh = np.transpose(width_array_mesh)

    return mass_values, width_values, mass_array_mesh, width_array_mesh

def contour_plot(mass_values, width_values, \
                 mass_array_mesh, width_array_mesh, data, values):
    """
    Plots a contour of chi squared for varying mass and width. Uses this
    contour plot to calculate the uncertainty in each value.

    Returns
    -------
    mass and width uncertainity, floats

    """
    figure = plt.figure()
    ax = figure.add_subplot(111)
    mass = values[0]
    width = values[1]
    min_chi_square = chi_square(data, cross_section_curve(data[:, 0], values))

    function_values = np.empty((0, len(mass_values)))
    for width_value in width_values:
        row = np.array([])
        for mass_value in mass_values:
            row = np.append(row,\
                            chi_square(data,\
                                       cross_section_curve(data[:, 0],\
                                                           (mass_value,\
                                                            width_value))))
        function_values = np.vstack((function_values, row))

    plot = ax.contour(mass_array_mesh, width_array_mesh,
                      function_values, levels=[min_chi_square + 1,\
                                           min_chi_square + 2,\
                                               min_chi_square + 3,\
                                                   min_chi_square + 4,\
                                                       min_chi_square + 5, \
                                                           min_chi_square + 6,\
                                                        min_chi_square + 7, \
                                                        min_chi_square + 8])

    ax.clabel(plot)
    ax.scatter(mass, width, label='Mass = {0:.5}\nWidth ={1:.5}'\
               .format(values[0], values[1]))
    ax.set_title('Chi Squared Varying Mass and Width')
    ax.set_xlabel('Mass (GeV/c^2)')
    ax.set_ylabel('Width (GeV)')
    plt.savefig('chi_squared_contour_plot.png', dpi=300, bbox_inches='tight')
    ax.legend(loc='center left')
    plt.show()

    coords = plot.collections[0].get_paths()[0].vertices \
        #finds coordinates of contours

    x = coords[:, 0]
    y = coords[:, 0]

    x_min = min(x)
    y_min = min(y)
    x_max = max(x)
    y_max = max(y)

    sigma_x = (x_max - x_min) / 2
    sigma_y = (y_max - y_min) / 2



    return sigma_x, sigma_y

def lifetime_error(lifetime, width_error, width):
    """
    Calculates the uncertainity of the lifetime using basic error propagation

    Parameters
    ----------
    lifetime : float

    width_error : float

    Returns
    -------
    uncertainity of the lifetime, float

    """
    lifetime_uncertainty = lifetime * (width_error/width)

    return lifetime_uncertainty

def main():
    """
    Calls the other defined functions and prints out results.

    Returns
    -------
    None.

    """
    validated_data_1 = validate_data_1()
    validated_data_2 = validate_data_2()
    data = data_combine(validated_data_1, validated_data_2)
    data = remove_outliers(data)


    cross_section_prediction = cross_section_curve(data[:, 0], (90, 3))

    chi_square(data, cross_section_prediction)
    data, values = minimisation(data)

    plot_data(data, values)
    lifetime = lifetime_calculator(values)
    mass_values, width_values, mass_array_mesh, width_array_mesh = array_mesh()
    mass_error, width_error = contour_plot(mass_values,\
                                           width_values,\
                                               mass_array_mesh, \
                                                   width_array_mesh, \
                                                       data, values)
    print('The mass of the Z boson is {0:3.2f} +/- {2:.3} GeV/c^2 and \
the width is {1:3.3f} +/- {3:.3} GeV.'.format(values[0],\
              values[1], mass_error, width_error))
    lifetime_uncertainty = lifetime_error(lifetime, width_error, values[1])
    print('The lifetime of the Z boson is {0:.3} +/- {1:.2} seconds.'\
          .format(lifetime, lifetime_uncertainty))


if __name__ == "__main__": #main code
    main()
