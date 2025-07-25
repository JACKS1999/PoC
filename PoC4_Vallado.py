import functions
import dynamics
import constants
import numpy as np
import sys
import scipy.special
import time as tm
from scipy.interpolate import interp1d
import plotly.graph_objs as go


class probability_of_collision():
    def __init__(self, primary_initial_keplerian_elements, primary_mass, primary_radius, primary_drag_coefficient, primary_reflectivity, secondary_initial_keplerian_elements, secondary_mass, secondary_radius, secondary_drag_coefficient, secondary_reflectivity, safety_distance, start_time, propagation_time, number_of_evaluations, max_harmonics_degree, max_harmonics_order, rtol = 1e-12, atol = 1e-10):
        self.primary_initial_keplerian_elements = primary_initial_keplerian_elements
        self.primary_mass = primary_mass
        self.primary_radius = primary_radius
        self.primary_reference_surface = 1# np.pi * primary_radius**2
        self.primary_drag_coefficient = primary_drag_coefficient
        self.primary_reflectivity = primary_reflectivity
        
        self.secondary_initial_keplerian_elements = secondary_initial_keplerian_elements
        self.secondary_mass = secondary_mass
        self.secondary_radius = secondary_radius
        self.secondary_reference_surface = 1# np.pi * secondary_radius**2
        self.secondary_drag_coefficient = secondary_drag_coefficient
        self.secondary_reflectivity = secondary_reflectivity
        
        self.safety_distance = safety_distance
        self.start_time = start_time
        self.propagation_time = propagation_time
        self.rtol = rtol
        self.atol = atol
        self.max_degree = max_harmonics_degree
        self.max_order = max_harmonics_order
        self.number_of_evaluations = number_of_evaluations
        #self.safety_ellipse_major_axis
        #self.safety_ellipse_minor_axis
        def plotter(self):
            pass
        
    def initial_filter(self):
        primary_perigee, primary_apogee = functions.keplerian_elements_to_rp_ra(self.primary_initial_keplerian_elements)
        secondary_perigee, secondary_apogee = functions.keplerian_elements_to_rp_ra(self.secondary_initial_keplerian_elements)
        max_perigee = max(primary_perigee, secondary_perigee)
        min_apogee = min(primary_apogee, secondary_apogee)
        if max_perigee - min_apogee > self.safety_distance:
            print('')
            print("No probability of collision detected by initial filter")
            print('')
            sys.exit()

    def value_interpolator(self, matrix, time, interpolation_time):
        interpolated_rows = []
        for i in range(len(matrix[:,0])):
            values_data = matrix[i,:]
            interp_func = interp1d(time, values_data, kind = 'linear', fill_value = 'extrapolate')
            interpolated_values = interp_func(interpolation_time)
            interpolated_rows.append(interpolated_values)
        new_matrix = np.stack(interpolated_rows)
        return new_matrix

    def d_function(self, index):
        relative_distance = self.relative_distance[:, index]
        relative_velocity = self.relative_velocity[:, index] 
        relative_acceleration = self.relative_acceleration[:, index]
        d_distance_function = 2*functions.dot(relative_velocity, relative_distance)
        dd_distance_function = 2* (functions.dot(relative_acceleration, relative_distance) + functions.dot(relative_velocity, relative_velocity))
        return d_distance_function, dd_distance_function

    def cubic_splining(self, t_n, t_np1, index1, index2):
        f1, f1_dot = self.d_function(index1)
        f2, f2_dot = self.d_function(index2)
        self.delta_t = t_np1 - t_n
        delta_t = self.delta_t
        alpha_c0 = f1
        alpha_c1 = f1_dot * delta_t
        alpha_c2 = -3*f1 -2*f1_dot*delta_t + 3*f2 - f2_dot*delta_t
        alpha_c3 = 2*f1 + f1_dot*delta_t - 2*f2 + f2_dot*delta_t
        tau = np.roots([alpha_c3, alpha_c2, alpha_c1, alpha_c0])
        real_tau = np.real(tau)
        #real_tau = real_tau[np.isreal(real_tau)]
        valid_tau = real_tau[(real_tau >= 0) & (real_tau <= 1)]
        return valid_tau
       
    def quintic_splining(self, relative_distance, relative_velocity, relative_acceleration, index1, index2):
        f1 = relative_distance[index1]
        f1_dot = relative_velocity[index1]
        f1_ddot = relative_acceleration[index1]
        f2 = relative_distance[index2]
        f2_dot = relative_velocity[index2]
        f2_ddot = relative_acceleration[index2]
        delta_t = self.delta_t
        tau = self.tau
        alpha_q0 = f1
        alpha_q1 = f1_dot*delta_t
        alpha_q2 = 0.5*f1_ddot*delta_t**2
        alpha_q3 = -10*f1 - 6*f1_dot*delta_t - 1.5*f1_ddot*delta_t**2 + 10*f2 - 4*f2_dot*delta_t + 0.5*f2_ddot*delta_t**2
        alpha_q4 = 15*f1 + 8*f1_dot*delta_t + 1.5*f1_ddot*delta_t**2 - 15*f2 + 7*f2_dot*delta_t - f2_ddot*delta_t**2
        alpha_q5 = -6*f1 - 3*f1_dot*delta_t - 0.5*f1_ddot*delta_t**2 + 6*f2 - 3*f2_dot*delta_t + 0.5*f2_ddot*delta_t**2
        P_q = alpha_q5*tau**5 + alpha_q4*tau**4 + alpha_q3*tau**3 + alpha_q2*tau**2 + alpha_q1*tau + alpha_q0
        return P_q

    def alternate_cubic_splining(self, f_e0, f_e1, f_e2, f_e3):
        p1 = f_e0
        p2 = f_e1
        p3 = f_e2
        p4 = f_e3
        tau_1 = 1/3
        tau_2 = 2/3
        DET = tau_1**3 *tau_2**2 + tau_1**2 *tau_2 + tau_1*tau_2**3 - tau_1**3 *tau_2 - tau_1**2 *tau_2**3 - tau_1*tau_2**2
        alpha_c0 = p1
        alpha_c1 = ((tau_2**3-tau_2**2)*(p2-p1)+(tau_1**2-tau_1**3)*(p3-p1)+(tau_1**3*tau_2**2 - tau_1**2*tau_2**3)*(p4-p1))/DET
        alpha_c2 = ((tau_2-tau_2**3)*(p2-p1)+(tau_1**3-tau_1)*(p3-p1)+(tau_1*tau_2**3 - tau_1**3*tau_2)*(p4-p1))/DET
        alpha_c3 = ((tau_2**2-tau_2)*(p2-p1)+(tau_1**2-tau_1**2)*(p3-p1)+(tau_1**2*tau_2 - tau_1*tau_2**2)*(p4-p1))/DET
        tau = np.roots([alpha_c3, alpha_c2, alpha_c1, alpha_c0])
        real_tau = tau[np.isreal(tau)].real
        return real_tau

    def ellipsoidal_function(self, relative_distance, primary_velocity):
        r_d = relative_distance
        v_p = primary_velocity
        a = self.safety_ellipse_major_axis
        b = self.safety_ellipse_minor_axis
        N = np.dot(r_d, v_p) / v_p
        T = np.dot(r_d, r_d) - N**2
        ellipsoidal_function = N**2/a**2 + T**2/b**2 -1

    def maximum_probability(self):
        r_obj_comb = (self.primary_radius + self.secondary_radius)/1000
        dist = self.minimum_distance
        r = r_obj_comb / dist
        if dist < r_obj_comb: 
            print('Predicted miss distance is less than the combined object size, maneuver needed!')
            print('Predicted miss distance:', dist)
            print('Predicted time of closest approach:', self.time_at_minimum_distance)
            
        else:
            PoC = 0.5*(scipy.special.erf( (r+1)/(2*np.sqrt(r)) * np.sqrt( -np.log( (1-r) / (1+r)) ) ) + scipy.special.erf( (r-1)/(2*np.sqrt(r)) * np.sqrt( -np.log( (1-r) / (1+r)) ) ) )
            if dist < 10:
                f = np.interp(dist, np.array([0.05, 0.1, 0.25, 0.5, 1]), np.array([1e-3, 1e-4, 1e-5, 1e-6, 1e-7])) #1e-2, 1e-4, 1e-6, 1e-7
                f = np.interp(dist, np.array([r_obj_comb+0.001, 0.1, 0.2, 0.3, 0.4]), np.array([1e-5, 1e-6, 1e-7, 1e-8, 1e-9]))
                print(f)
                periodo = 2*np.pi * np.sqrt(self.primary_initial_keplerian_elements[0]**3/constants.MU_EARTH)
                coeff1 = 0.03 + (0.005* self.time_at_minimum_distance/periodo)
                coeff2 = 0.97 - (0.005* self.time_at_minimum_distance/periodo)

                coeff1 = 0.03 + (0.005* self.time_at_minimum_distance/periodo)
                coeff2 = 0.97 - (0.005* self.time_at_minimum_distance/periodo)
                PoC = coeff1*PoC + coeff2*f 
                
                print('Predicted maximum Probability of Collision:', PoC)
                print('Preditcted miss distance:', dist)
                print('Predicted time of closest approach:', self.time_at_minimum_distance)
                print('')

    def calculate_probability(self):
        self.initial_filter()

        primary_initial_state = functions.keplerian_elements_to_state(self.primary_initial_keplerian_elements, constants.MU_EARTH)
        secondary_initial_state = functions.keplerian_elements_to_state(self.secondary_initial_keplerian_elements, constants.MU_EARTH)
        
        self.primary_dynamics = dynamics.dynamics(primary_initial_state, self.primary_mass, self.primary_reference_surface, self.primary_drag_coefficient, self.primary_reflectivity, self.start_time, self.propagation_time, self.rtol, self.atol, self.max_degree, self.max_order, self.number_of_evaluations)
        primary_propagate_start_time = tm.time()
        primary_dynamics = self.primary_dynamics.propagate() 
        evaluation_epochs = primary_dynamics.t 
        primary_ephemerides = primary_dynamics.y[:6,:]
        primary_accelerations = self.primary_dynamics.calculate_accelerations(primary_ephemerides, evaluation_epochs) #primary_dynamics.y[6:,:]
        #primary_ephemerides, primary_accelerations, primary_integrator_times = self.primary_dynamics.propagate() 
        primary_propagate_end_time = tm.time()
        self.primary_positions = primary_ephemerides[:3, :]
        primary_propagate_elapsed_time = primary_propagate_end_time - primary_propagate_start_time
        print('')
        print('Primary object successfully propagated, elapsed time:', primary_propagate_elapsed_time)
        print('')

        self.secondary_dynamics = dynamics.dynamics(secondary_initial_state, self.secondary_mass, self.secondary_reference_surface, self.secondary_drag_coefficient, self.secondary_reflectivity, self.start_time, self.propagation_time, self.rtol, self.atol, self.max_degree, self.max_order, self.number_of_evaluations)
        secondary_propagate_start_time = tm.time()
        secondary_dynamics = self.secondary_dynamics.propagate() 
        secondary_ephemerides = secondary_dynamics.y[:6,:]
        secondary_accelerations = self.secondary_dynamics.calculate_accelerations(secondary_ephemerides, evaluation_epochs)# secondary_dynamics.y[6:,:]
        #secondary_ephemerides, secondary_accelerations, secondary_integrator_times = self.secondary_dynamics.propagate()
        secondary_propagate_end_time = tm.time()
        self.secondary_positions = secondary_ephemerides[:3, :]
        secondary_propagate_elapsed_time = secondary_propagate_end_time - secondary_propagate_start_time
        print('Secondary object successfully propagated, elapsed time:', secondary_propagate_elapsed_time)
        print('')
        
        #primary_accelerations = self.value_interpolator(primary_accelerations[:,:-2], primary_integrator_times, evaluation_epochs)
        #secondary_accelerations = self.value_interpolator(secondary_accelerations[:,:-2], secondary_integrator_times, evaluation_epochs)

        self.relative_distance =  secondary_ephemerides[:3, :] - primary_ephemerides[:3, :] 
        self.relative_velocity =  secondary_ephemerides[3:, :] - primary_ephemerides[3:, :]
        self.relative_acceleration = secondary_accelerations - primary_accelerations 
        
        for n in range(len(evaluation_epochs)):
            if n == (len(evaluation_epochs)-1):
                print('No further conjunctions detected in simulated propagation time')
                print('')
                break
            t_n = evaluation_epochs[n]
            t_np1 = evaluation_epochs[n+1]
            index1 = n
            index2 = n+1
            tau = self.cubic_splining(t_n, t_np1, index1, index2)
            if len(tau) != 0:
                for i in tau:
                    self.tau = i

                    P_q_I = self.quintic_splining(self.relative_distance[0,:], self.relative_velocity[0,:], self.relative_acceleration[0,:], index1, index2)
                    P_q_J = self.quintic_splining(self.relative_distance[1,:], self.relative_velocity[1,:], self.relative_acceleration[1,:], index1, index2)
                    P_q_K = self.quintic_splining(self.relative_distance[2,:], self.relative_velocity[2,:], self.relative_acceleration[2,:], index1, index2)

                    self.minimum_distance = functions.norm(np.array([P_q_I, P_q_J, P_q_K]))
                    self.time_at_minimum_distance = t_n + self.tau*self.delta_t
                    self.maximum_probability()
        fig = self.plotter()
        #f_e0 = self.ellipsoidal_function(relative_distance[:3, t_n], primary_ephemerides.y[4:, t_n])
        #f_e1 = self.ellipsoidal_function(relative_distance[:3, t_n+tau*1/3], primary_ephemerides.y[4:, t_n+tau*1/3])
        #f_e2 = self.ellipsoidal_function(relative_distance[:3, t_n+tau*2/3], primary_ephemerides.y[4:, t_n+tau*2/3])
        #f_e3 = self.ellipsoidal_function(relative_distance[:3, t_n+tau], primary_ephemerides.y[4:, t_n+tau])
        return fig

    def plotter(self):
        Re = constants.R_EARTH
        primary_positions = self.primary_positions
        secondary_positions = self.secondary_positions
        
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)

        # meshgrid of points for the Earth sphere
        phi_sphere, theta_sphere = np.meshgrid(phi, theta)

        # spherical coordinates to Cartesian coordinates
        x_sphere = Re * np.sin(phi_sphere) * np.cos(theta_sphere)
        y_sphere = Re * np.sin(phi_sphere) * np.sin(theta_sphere)
        z_sphere = Re * np.cos(phi_sphere)

        earth_sphere_trace = go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, opacity=1, colorscale=[[0, 'rgb(100, 150, 255)'], [1, 'rgb(100, 150, 255)']],)

        primary_orbit_trace = go.Scatter3d(x = primary_positions[0,:], y = primary_positions[1,:], z = primary_positions[2,:], mode='lines', line=dict(color='blue', width=5), opacity = 0.95)
        secondary_orbit_trace = go.Scatter3d(x = secondary_positions[0,:], y = secondary_positions[1,:], z = secondary_positions[2,:], mode='lines', line=dict(color='orange', width=1.5), opacity = 0.95)

        axis_length = Re + 1500

        x_axis_trace = go.Scatter3d(x=[0, axis_length], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=4), name='X Axis')
        y_axis_trace = go.Scatter3d(x=[0, 0], y=[0, axis_length], z=[0, 0], mode='lines', line=dict(color='red', width=4), name='Y Axis')
        z_axis_trace = go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_length], mode='lines', line=dict(color='red', width=4), name='Z Axis')
        #equator = go.Scatter3d(x =(Re+1)*np.cos(theta), y =(Re+1)* np.sin(theta), z=np.zeros(len(theta)), mode='lines', line=dict(color='black', width=3))

        axis = Re + 2500
        # layout for the 3D plot
        layout = go.Layout(
            title='Orbits and Earth Sphere in 3D',
            scene=dict(
                xaxis=dict(title='X (km)', range=[-axis, axis]),
                yaxis=dict(title='Y (km)', range=[-axis, axis]),
                zaxis=dict(title='Z (km)', range=[-axis, axis]),
                aspectmode='cube',  # Keep the aspect ratio of the plot
            ),
        )
        x_cone = go.Cone(x=[axis_length], y=[0], z=[0], u=[1000], v=[0], w=[0])
        y_cone = go.Cone(x=[0], y=[axis_length], z=[0], u=[0], v=[1000], w=[0])
        z_cone = go.Cone(x=[0], y=[0], z=[axis_length], u=[0], v=[0], w=[1000])
        # Create figure and add traces to it
        fig = go.Figure(data=[earth_sphere_trace, primary_orbit_trace, secondary_orbit_trace, x_axis_trace, y_axis_trace, z_axis_trace, x_cone, y_cone, z_cone], layout=layout) #equator

        return fig

class probability_of_collision2():
    def __init__(self, delta_radius, primary_data, secondary_data, start_time, propagation_time, number_of_evaluations,
                                                 max_harmonics_degree = 2,max_harmonics_order = 0,rtol = 1e-12, atol = 1e-12):
        self.delta_radius = delta_radius
        
        self.primary_initial_state = primary_data[:6]
        self.primary_mass = primary_data[6]
        self.primary_radius = primary_data[7]
        self.primary_reference_surface = 1#self.primary_radius**2 * np.pi
        self.primary_drag_coefficient = primary_data[8]
        self.primary_reflectivity = primary_data[9]

        self.secondary_initial_state = secondary_data[:6]
        self.secondary_mass = secondary_data[6]
        self.secondary_radius = secondary_data[7]
        self.secondary_reference_surface = 1#self.secondary_radius**2 * np.pi
        self.secondary_drag_coefficient = secondary_data[8]
        self.secondary_reflectivity = secondary_data[9]

        self.start_time = start_time
        self.propagation_time = propagation_time
        self.number_of_evaluations = number_of_evaluations
        self.max_degree = max_harmonics_degree
        self.max_order = max_harmonics_order
        self.rtol = rtol
        self.atol = atol

    def d_function(self, index):
        relative_distance = self.relative_distance[:, index]
        relative_velocity = self.relative_velocity[:, index] 
        relative_acceleration = self.relative_acceleration[:, index]
        d_distance_function = 2*functions.dot(relative_velocity, relative_distance)
        dd_distance_function = 2* (functions.dot(relative_acceleration, relative_distance) + functions.dot(relative_velocity, relative_velocity))
        return d_distance_function, dd_distance_function

    def cubic_splining(self, t_n, t_np1, index1, index2):
        f1, f1_dot = self.d_function(index1)
        f2, f2_dot = self.d_function(index2)
        self.delta_t = t_np1 - t_n
        delta_t = self.delta_t
        alpha_c0 = f1
        alpha_c1 = f1_dot * delta_t
        alpha_c2 = -3*f1 -2*f1_dot*delta_t + 3*f2 - f2_dot*delta_t
        alpha_c3 = 2*f1 + f1_dot*delta_t - 2*f2 + f2_dot*delta_t
        tau = np.roots([alpha_c3, alpha_c2, alpha_c1, alpha_c0])
        real_tau = np.real(tau)
        #real_tau = real_tau[np.isreal(real_tau)]
        valid_tau = real_tau[(real_tau >= 0) & (real_tau <= 1)]
        return valid_tau
       
    def quintic_splining(self, relative_distance, relative_velocity, relative_acceleration, index1, index2):
        f1 = relative_distance[index1]
        f1_dot = relative_velocity[index1]
        f1_ddot = relative_acceleration[index1]
        f2 = relative_distance[index2]
        f2_dot = relative_velocity[index2]
        f2_ddot = relative_acceleration[index2]
        delta_t = self.delta_t
        tau = self.tau
        alpha_q0 = f1
        alpha_q1 = f1_dot*delta_t
        alpha_q2 = 0.5*f1_ddot*delta_t**2
        alpha_q3 = -10*f1 - 6*f1_dot*delta_t - 1.5*f1_ddot*delta_t**2 + 10*f2 - 4*f2_dot*delta_t + 0.5*f2_ddot*delta_t**2
        alpha_q4 = 15*f1 + 8*f1_dot*delta_t + 1.5*f1_ddot*delta_t**2 - 15*f2 + 7*f2_dot*delta_t - f2_ddot*delta_t**2
        alpha_q5 = -6*f1 - 3*f1_dot*delta_t - 0.5*f1_ddot*delta_t**2 + 6*f2 - 3*f2_dot*delta_t + 0.5*f2_ddot*delta_t**2
        P_q = alpha_q5*tau**5 + alpha_q4*tau**4 + alpha_q3*tau**3 + alpha_q2*tau**2 + alpha_q1*tau + alpha_q0
        return P_q
    
    def maximum_probability(self):
        r_obj_comb = (self.primary_radius + self.secondary_radius)/1000
        dist = self.minimum_distance
        r = r_obj_comb / dist
        if dist < r_obj_comb: 
            print('Predicted miss distance is less than the combined object size, maneuver needed!')
            print('Predicted miss distance:', dist)
            print('Predicted time of closest approach:', self.time_at_minimum_distance)
            
        else:
            PoC = 0.5*(scipy.special.erf( (r+1)/(2*np.sqrt(r)) * np.sqrt( -np.log( (1-r) / (1+r)) ) ) + scipy.special.erf( (r-1)/(2*np.sqrt(r)) * np.sqrt( -np.log( (1-r) / (1+r)) ) ) )
            if dist < 10:
                f = np.interp(dist, np.array([0.05, 0.1, 0.25, 0.5, 1]), np.array([1e-3, 1e-4, 1e-5, 1e-6, 1e-7])) #1e-2, 1e-4, 1e-6, 1e-7
                f = np.interp(dist, np.array([r_obj_comb+0.001, 0.1, 0.2, 0.3, 0.4]), np.array([1e-5, 1e-6, 1e-7, 1e-8, 1e-9]))
                print(f)
                a = np.sqrt(self.primary_initial_state[0]**2 + self.primary_initial_state[1]**2 +self.primary_initial_state[2]**2)
                periodo = 2*np.pi * np.sqrt(a**3/constants.MU_EARTH)
                coeff1 = 0.03 + (0.005* self.time_at_minimum_distance/periodo)
                coeff2 = 0.97 - (0.005* self.time_at_minimum_distance/periodo)

                coeff1 = 0.03 + (0.005* self.time_at_minimum_distance/periodo)
                coeff2 = 0.97 - (0.005* self.time_at_minimum_distance/periodo)
                PoC = coeff1*PoC + coeff2*f 

                output_text = f'{self.delta_radius} {PoC} {self.minimum_distance} {self.time_at_minimum_distance}'
                file_path = "PoC_looped.txt" 
                with open(file_path, "a") as file:
                    file.write(output_text + "\n")

    def calculate_probability2(self):
        self.primary_dynamics = dynamics.dynamics(self.primary_initial_state, self.primary_mass, self.primary_reference_surface, self.primary_drag_coefficient, self.primary_reflectivity, self.start_time, self.propagation_time, self.rtol, self.atol, self.max_degree, self.max_order, self.number_of_evaluations)
        primary_propagate_start_time = tm.time()
        primary_dynamics = self.primary_dynamics.propagate() 
        evaluation_epochs = primary_dynamics.t 
        primary_ephemerides = primary_dynamics.y[:6,:]
        primary_accelerations = self.primary_dynamics.calculate_accelerations(primary_ephemerides, evaluation_epochs) 
        primary_propagate_end_time = tm.time()
        self.primary_positions = primary_ephemerides[:3, :]
        primary_propagate_elapsed_time = primary_propagate_end_time - primary_propagate_start_time
        print('')
        print('Primary object successfully propagated, elapsed time:', primary_propagate_elapsed_time)
        print('')

        self.secondary_dynamics = dynamics.dynamics(self.secondary_initial_state, self.secondary_mass, self.secondary_reference_surface, self.secondary_drag_coefficient, self.secondary_reflectivity, self.start_time, self.propagation_time, self.rtol, self.atol, self.max_degree, self.max_order, self.number_of_evaluations)
        secondary_propagate_start_time = tm.time()
        secondary_dynamics = self.secondary_dynamics.propagate() 
        secondary_ephemerides = secondary_dynamics.y[:6,:]
        secondary_accelerations = self.secondary_dynamics.calculate_accelerations(secondary_ephemerides, evaluation_epochs)
        
        secondary_propagate_end_time = tm.time()
        self.secondary_positions = secondary_ephemerides[:3, :]
        secondary_propagate_elapsed_time = secondary_propagate_end_time - secondary_propagate_start_time
        print('Secondary object successfully propagated, elapsed time:', secondary_propagate_elapsed_time)
        print('')
        
        self.relative_distance =  secondary_ephemerides[:3, :] - primary_ephemerides[:3, :] 
        self.relative_velocity =  secondary_ephemerides[3:, :] - primary_ephemerides[3:, :]
        self.relative_acceleration = secondary_accelerations - primary_accelerations 
        
        for n in range(len(evaluation_epochs)):
            if n == (len(evaluation_epochs)-1):
                print('No further conjunctions detected in simulated propagation time')
                print('')
                break
            t_n = evaluation_epochs[n]
            t_np1 = evaluation_epochs[n+1]
            index1 = n
            index2 = n+1
            tau = self.cubic_splining(t_n, t_np1, index1, index2)
            if len(tau) != 0:
                for i in tau:
                    self.tau = i

                    P_q_I = self.quintic_splining(self.relative_distance[0,:], self.relative_velocity[0,:], self.relative_acceleration[0,:], index1, index2)
                    P_q_J = self.quintic_splining(self.relative_distance[1,:], self.relative_velocity[1,:], self.relative_acceleration[1,:], index1, index2)
                    P_q_K = self.quintic_splining(self.relative_distance[2,:], self.relative_velocity[2,:], self.relative_acceleration[2,:], index1, index2)

                    self.minimum_distance = functions.norm(np.array([P_q_I, P_q_J, P_q_K]))
                    self.time_at_minimum_distance = t_n + self.tau*self.delta_t
                    self.maximum_probability()