import constants
from datetime import datetime, timedelta
import PoC4_Vallado
import plotly.io as pio
import functions
import numpy as np
import dynamics
import spiceypy as spice
#template = "plotly_dark"
#pio.templates.default = template

def run_probability_of_collision(primary_initial_keplerian_elements, primary_mass, primary_radius, primary_drag_coefficient, primary_reflectivity, secondary_initial_keplerian_elements, secondary_mass, secondary_radius, secondary_drag_coefficient, secondary_reflectivity, safety_distance, start_time, propagation_time, number_of_evaluations):
    
    udp = PoC4_Vallado.probability_of_collision(primary_initial_keplerian_elements,     
                       primary_mass,
                       primary_radius, 
                       primary_drag_coefficient,
                       primary_reflectivity,
                       secondary_initial_keplerian_elements, 
                       secondary_mass, 
                       secondary_radius, 
                       secondary_drag_coefficient, 
                       secondary_reflectivity,
                       safety_distance,
                       start_time,
                       propagation_time,
                       number_of_evaluations,
                       max_harmonics_degree = 2,
                       max_harmonics_order = 0,
                       rtol = 1e-12,
                       atol = 1e-12)
    
    fig = udp.calculate_probability()
    #fig = udp.plotter()
    pio.show(fig)

def initial_propagation(keplerian_elements, mass, radius, drag_coeff, reflect_coeff, start_time, propag_time):
    initial_state = functions.keplerian_elements_to_state(keplerian_elements, constants.MU_EARTH)
    surf = np.pi*radius**2
    number_of_evaluations = propag_time/50
    object_dynamics = dynamics.dynamics(initial_state, mass, surf, drag_coeff, reflect_coeff, start_time, propag_time, rtol = 1e-12, atol=1e-12, max_degree = 0, max_order = 0, number_of_evaluations = number_of_evaluations)
    state = object_dynamics.propagate()
    new_initial_state = state.y[:,-1]
    #new_initial_keplerian_elements = functions.state_to_keplerian_elements(new_initial_state, constants.MU_EARTH)
    #print(new_initial_keplerian_elements)
    #return new_initial_keplerian_elements
    return new_initial_state


def main():

    epoch = datetime(2024, 6, 3, 00, 00)

    #*** PRIMARY OBJECT ***#
    primary_initial_keplerian_elements = [6378+300, 0, 45, 1, 0, 1] #constants.R_EARTH+300.0115, 0, 45, 1, 0, 0.96 #  6378+300, 0, 45, 1, 0, 1
    primary_mass = 150
    primary_radius = 1
    primary_drag_coefficient = 2.1
    primary_reflectivity = 1
    #primary_days_since_epoch = 3.481

    #*** SECONDARY OBJECT ***#
    secondary_initial_keplerian_elements = [constants.R_EARTH+300.155, 0, -0.01, 0, 0, 0.9999] #constants.R_EARTH+300.17, 0, 0.01, 0, 0, 0.96 # constants.R_EARTH+300.155, 0, -0.01, 0, 0, 0.9999
    secondary_mass = 150
    secondary_radius = 1
    secondary_drag_coefficient = 2.1
    secondary_reflectivity = 1
    #secondary_days_since_epoch = 3.617

    #days_since_epoch_difference = np.abs(primary_days_since_epoch - secondary_days_since_epoch)
    #epoch_difference = days_since_epoch_difference * constants.DAY2SEC
#
    #if primary_days_since_epoch < secondary_days_since_epoch:
    #    start_time = epoch + timedelta(days = primary_days_since_epoch)
    #    primary_initial_keplerian_elements = initial_propagation(primary_initial_keplerian_elements, primary_mass, primary_radius, primary_drag_coefficient, primary_reflectivity, start_time, epoch_difference)
    #    start_time = epoch + timedelta(days = secondary_days_since_epoch)
    #else: 
    #    start_time = epoch + timedelta(days = secondary_days_since_epoch)
    #    secondary_initial_keplerian_elements = initial_propagation(secondary_initial_keplerian_elements, secondary_mass, secondary_radius, secondary_drag_coefficient, secondary_reflectivity, start_time, epoch_difference)
    #    start_time = epoch + timedelta( days = primary_days_since_epoch)
#
    start_time = epoch
    propagation_time = 7*constants.DAY2SEC
    #start_time = epoch + timedelta(epoch_difference)
    number_of_evaluations = propagation_time / (100)
    safety_distance = 10
    
    run_probability_of_collision(primary_initial_keplerian_elements, primary_mass, primary_radius, primary_drag_coefficient, primary_reflectivity, secondary_initial_keplerian_elements, secondary_mass, secondary_radius, secondary_drag_coefficient, secondary_reflectivity, safety_distance, start_time, propagation_time, number_of_evaluations)


if __name__ == "__main__":
    main()