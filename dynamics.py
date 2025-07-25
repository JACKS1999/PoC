from scipy.integrate import solve_ivp
import numpy as np
from numba import njit
import spiceypy as spice
from datetime import datetime, timedelta
import constants
import gsh
from nrlmsise00 import msise_model
spice.furnsh('de440.bsp')
spice.furnsh('naif0012.tls')
spice.furnsh('earth_assoc_itrf93.tf')
spice.furnsh('earth_000101_240704_240410.bpc')

@njit(nopython = True)
def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

@njit(nopython = True)
def norm(v):
    return (v[0]**2 + v[1]**2 + v[2]**2)**0.5

class dynamics():
    def __init__(self, initial_state, mass, reference_surface, drag_coefficient, reflectivity, start_time, propagation_time, rtol, atol, max_degree, max_order, number_of_evaluations):
        self.initial_state = initial_state
        self.mass = mass
        self.reference_surface = reference_surface
        self.drag_coefficient = drag_coefficient
        self.reflectivity = reflectivity
        self.initial_time = start_time
        self.propagation_time = propagation_time
        self.rtol = rtol
        self.atol = atol
        self.number_of_evaluations = int(number_of_evaluations)
        

        self.earth_angular_velocity = constants.W_EARTH
        self.j2000_ref_time = spice.str2et(datetime(2000, 1, 1, 12, 00).strftime('%Y-%m-%dT%H:%M:%S'))
        self.j2000_ref_angle = 280.46061837504 * constants.DEG2RAD
        self.earth_radius = constants.R_EARTH
        self.mu_earth = constants.MU_EARTH
        self.sun_radius = constants.R_SUN
        self.mu_sun = constants.MU_SUN
        self.mu_moon = constants.MU_MOON

        def coeff(max_degree, max_order):
            coefficients = {'C': np.zeros((max_degree + 1, max_order + 1)), 'S': np.zeros((max_degree + 1, max_order + 1))}
            with open('EGM2008_to2190_TideFree', 'r') as file:
                for line in file:
                    parts = line.split()
                    if len(parts) >= 4:
                        # Parse degree, order, Cnm, and Snm from the line
                        n, m = int(parts[0]), int(parts[1])
                        if n <= max_degree and m <= max_order:
                            Cnm, Snm = float(parts[2].replace('D', 'E')), float(parts[3].replace('D', 'E'))
                            coefficients['C'][(n, m)] = Cnm
                            coefficients['S'][(n, m)] = Snm
                        elif n > max_degree:
                            # Stop reading if the degree exceeds the maximum specified
                            break
            return coefficients
        self.max_degree = max_degree # max degree
        self.max_order = max_order # max order
        coeffs = coeff(self.max_degree, self.max_order)
        self.clm = coeffs['C']
        self.slm = coeffs['S']
        self.normal_par = gsh._calculate_normalisation_parameters(self.max_degree)


    def drag(self): 
        x, y, z, vx, vy, vz = self.state
        v_ijk = np.array([vx, vy, vz])
        r_ijk = self.r
        et = self.et
        current_time = self.current_time
        w_e = self.earth_angular_velocity
        t_ref = self.j2000_ref_time
        theta_gref = self.j2000_ref_angle
        Re = self.earth_radius
        r = norm(r_ijk)
        C_d = self.drag_coefficient
        S_s = self.reference_surface
        m_s = self.mass

        v_atm = np.array([-w_e*r_ijk[2], w_e*r_ijk[0], 0])  # np.cross([0,0,w_e],r_ijk) #
        v_rel = v_ijk - v_atm
        itrf93_position = spice.mxv(spice.pxform('J2000', 'ITRF93', et), r_ijk)
        ra, lon, phi = spice.recrad(itrf93_position)
        d, _ = msise_model(current_time, (r - Re), np.rad2deg(phi), np.rad2deg(lon), 150, 150, 4) # tot density = d[5] -- [g/cm**3]
        rho = d[5] *1000 # [kg/m**3]
        a_d = - 0.5 * C_d * rho*S_s*1000 / m_s * norm(v_rel)**2 * v_rel/norm(v_rel)
        return a_d
    
    def eclipse(self, r_sun, r_body): # STK VALIDATED
        Rsun = self.sun_radius
        Re = self.earth_radius
        theta_sun = np.arcsin(Rsun/norm(r_sun)) # apparent size of the solar disk
        theta_body = np.arcsin(Re/norm(r_body)) # apparent size of Earth disk
        gamma = np.arccos(dot(r_sun,r_body) /(norm(r_sun)*norm(r_body))) # angular separation between sun and earth
        if (gamma-theta_sun) > theta_body: # no eclipse
            L = 1
        elif theta_body > (gamma+theta_sun): # umbra (total eclipse)
            L = 0
        elif (theta_sun-theta_body) >= gamma or gamma >= (theta_sun+theta_body): # annular eclipse
            L = 1-theta_body**2/theta_sun**2
        else: # partial eclipse
            A = theta_body**2 * np.arccos((gamma**2 + theta_body**2 - theta_sun**2)/(2*gamma*theta_body))
            B = theta_sun**2 * np.arccos((gamma**2 + theta_sun**2 - theta_body**2)/(2*gamma*theta_sun))
            C = 0.5 * np.sqrt((-gamma+theta_body+theta_sun)*(gamma-theta_body+theta_sun)*(gamma+theta_body-theta_sun)*(gamma+theta_body+theta_sun))
            L = 1- (A+B-C)/(np.pi*theta_sun**2)
        return L

    def solar_pressure(self): # STK VALIDATED
        C_r = self.reflectivity
        S_s = self.reference_surface
        m_s = self.mass
        sun_state, _ = spice.spkgeo(targ=10, et=self.et, ref='J2000', obs=399)
        sun_pos = np.array([sun_state[0], sun_state[1], sun_state[2]])
        
        r_ijk = self.r
        r_body = -1*r_ijk # ijk position of the Earth relative to the spacecraft
        r_sun = sun_pos + r_body # ijk position of the Sun relative to the spacecraft
        L = self.eclipse(r_sun, r_body)
        p_srp = 4.5567 * 1e-6 # [N/m**2] solar radiation pressure
        a_srp_ijk = - L * p_srp * C_r * S_s / (m_s * norm(r_sun)) * r_sun
        return a_srp_ijk

    def g_perturbation(self): # STK VALIDATED
        r_ijk = self.r
        Re = self.earth_radius
        mi = self.mu_earth
        theta_gref = self.j2000_ref_angle
        w_e = self.earth_angular_velocity
        t_ref = self.j2000_ref_time
        clm = self.clm
        slm = self.slm
        max_d = self.max_degree
        max_o = self.max_order
        normal_par = self.normal_par
        et = self.et

        r = norm(r_ijk)
        itrf93_position = spice.mxv(spice.pxform('J2000', 'ITRF93', et), r_ijk)
        itrf93_position = np.array([itrf93_position])
        acc = gsh.gravity_spherical_harmonic(itrf93_position, Re, mi, clm, slm, max_d, max_o, normal_par)
        acc = acc.squeeze()
        acc = spice.mxv(spice.pxform('ITRF93', 'J2000', et), acc)
        return acc

    def g_lunar(self, r_ijk): # STK VALIDATED
        mu_body = self.mu_moon
        body_state, _ = spice.spkgeo(targ=301, et=self.et, ref='J2000', obs=399) # j2000 position of body from earth 
        body_pos = np.array([body_state[0], body_state[1], body_state[2]])
        r_body = r_ijk - body_pos # j2000 position of sc from body
        acc_body = -mu_body * ((r_body/norm(r_body)**3) + (body_pos/norm(body_pos)**3))
        return acc_body # ijk acceleration

    def g_solar(self, r_ijk): # STK VALIDATED
        mu_body = self.mu_sun
        body_state, _ = spice.spkgeo(targ=10, et=self.et, ref='J2000', obs=399) # j2000 position of body from earth 
        body_pos = np.array([body_state[0], body_state[1], body_state[2]])
        r_body = r_ijk - body_pos # j2000 position of sc from body
        acc_body = -mu_body * ((r_body/norm(r_body)**3) + (body_pos/norm(body_pos)**3))
        return acc_body # ijk acceleration
        
    def lunisolar(self): # STK VALIDATED
        r_ijk = self.r
        acc_moon = self.g_lunar(r_ijk)
        acc_sun = self.g_solar(r_ijk)
        acc_tot = acc_moon + acc_sun
        return acc_tot
   
    def perturbations(self):
        if self.drag_coefficient == 0: a_drag = np.array([0,0,0])
        else: a_drag = self.drag()
        if self.reflectivity == 0: a_srp = np.array([0,0,0])
        else: a_srp = self.solar_pressure()
        a_earth_gravitational_field = self.g_perturbation()
        a_lunisolar = self.lunisolar()
        a_tot = a_drag + a_srp + a_earth_gravitational_field + a_lunisolar
        return a_tot
   
    def combined_equations(self, t, state):
        x, y, z, vx, vy, vz = state
        self.state = state
        self.r = np.array([x, y, z])
        self.current_time = self.initial_time + timedelta(seconds=t)
        self.et = spice.str2et(self.current_time.strftime('%Y-%m-%dT%H:%M:%S'))

        a_tot = self.perturbations()

        x_dot, y_dot, z_dot = vx, vy, vz
        vx_dot = a_tot[0]
        vy_dot = a_tot[1]
        vz_dot = a_tot[2]
        
        #if self.i == 0: 
        #    self.accelerations = np.zeros((3, 3))
        #    self.accelerations[:, 0] = np.array([vx_dot, vy_dot, vz_dot])
        #    self.times = t
        #else:
        #    self.accelerations = np.column_stack((self.accelerations, np.array([vx_dot, vy_dot, vz_dot])))
        #    self.times = np.hstack([self.times, t])
        state_dot = np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot])
        #self.i +=1
        return state_dot

    def propagate(self):
        #self.i = 0
        t_span = (0, self.propagation_time)
        t_eval = np.linspace(0, self.propagation_time, self.number_of_evaluations)
        sol = solve_ivp(self.combined_equations, t_span, self.initial_state, method='DOP853', t_eval = t_eval, rtol=self.rtol, atol=self.atol)
        return sol#, self.accelerations, self.times
    
    def calculate_accelerations(self, state, epochs):
        self.i = 0
        for columns in state.T:
            x, y, z, vx, vy, vz = columns
            t = epochs[self.i]
            self.state = columns
            self.r = np.array([x, y, z])
            self.current_time = self.initial_time + timedelta(seconds=t)
            self.et = spice.str2et(self.current_time.strftime('%Y-%m-%dT%H:%M:%S'))

            a_tot = self.perturbations()
            if self.i == 0: 
                self.accelerations = np.zeros((3, 1))
                self.accelerations[:, 0] = np.array([a_tot[0], a_tot[1], a_tot[2]])
            else:
                self.accelerations = np.column_stack((self.accelerations, np.array([a_tot[0], a_tot[1], a_tot[2]])))
            self.i +=1
        return self.accelerations[:,]
        