from scipy.integrate import solve_ivp
import numpy as np
from copy import deepcopy
import os
import matplotlib.pyplot as plt
from numba import njit
import state_from_kepler as sfk
from datetime import datetime, timedelta
from nrlmsise00 import msise_model #pydoc nrlmsise00
import plotly.graph_objs as go
import spiceypy as spice
import math


spice.furnsh('de440.bsp')
spice.furnsh('naif0012.tls')
spice.furnsh('earth_assoc_itrf93.tf')
spice.furnsh('earth_000101_240704_240410.bpc')

mi = 398600 # [km**3/s**2] Earth gravitational constant
Re = 6371 # [km] Earth Radius
J2 = 1.0826 *1e-3
w_e = 7.2921159 *1e-5 # [rad/s] Earth's rotation rate

height = 450
m_s = 500 # [kg] s/c mass
S_s = 1 # [m**2] s/c reference surface
C_d = 2 # drag coefficient (2-2.1 = sphere model, 2.2= flat plate model -- VALLADO)
C_r = 1 # reflectivity (0=transluscent, 1=black body, 2=flat mirror -- VALLADO)

start_time = datetime(2023, 4, 12, 16, 30) #datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])
time = 24*3600*20
nodes = (np.rint(time/60)).astype(int)

def drag(y, current_time, et):
    r, theta, phi, u, v, w = y
    r_ijk = [r*np.cos(theta)*np.cos(phi), r*np.sin(theta)*np.cos(phi), r*np.sin(phi)]
    v_atm = [-w_e*r_ijk[2], w_e*r_ijk[0], 0] #np.cross([0,0,w_e],r_ijk)
    R = sfk.eci_to_zen(theta, phi)
    v_atmos = np.dot(R, v_atm)
    v_sc = [u, v, w]
    v_rel = [v_atmos[0] - v_sc[0], v_atmos[1] - v_sc[1], v_atmos[2] - v_sc[2]]

    itrf93_position = spice.mxv(spice.pxform('J2000', 'ITRF93', et), r_ijk)
    ra, longitude, latitude = spice.recrad(itrf93_position)
    d, tt = msise_model(current_time, height, latitude, longitude, 150, 150, 4) # tot density = d[5] -- [g/cm**3]
    rho = d[5] *1000 # [kg/m**3]
    drag_component = C_d*0.5*rho*S_s*1000 # [kg/km]
    a_d = np.array([drag_component*v_rel[0]**2/m_s, drag_component*v_rel[1]**2/m_s, drag_component*v_rel[2]**2/m_s])
    return a_d

@njit
def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

@njit
def norm(v):
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

@njit
def eclipse(y, sun_pos):
    r, theta, phi, u, v, w = y
    r_ijk = [r*np.cos(theta)*np.cos(phi), r*np.sin(theta)*np.cos(phi), r*np.sin(phi)]
    cone_half_angle = math.atan(Re / norm(sun_pos))
    cos_angle = dot(sun_pos, r_ijk)/(norm(sun_pos)*norm(r_ijk))
    if cos_angle < -1.0:
        cos_angle = -1.0
    elif cos_angle > 1.0:
        cos_angle = 1.0
    Rr_angle = math.acos(cos_angle)
    return Rr_angle < cone_half_angle

def solar_pressure(y, et):
    sun_state, _ = spice.spkgeo(targ=10, et=et, ref='ECLIPJ2000', obs=399)
    sun_pos = [sun_state[0], sun_state[1], sun_state[2]]
    eclipse_state = eclipse(y, sun_pos) 
    if not eclipse_state:
        p_srp = 4.57 * 1e-6 # [N/m**2]
        norm_sun_pos = np.linalg.norm(sun_pos)
        sun_pos_array = np.array(sun_pos)
        F_srp_ijk = np.array([(-p_srp * C_r * S_s) * (sun_pos_array[0] / norm_sun_pos),
                            (-p_srp * C_r * S_s) * (sun_pos_array[1] / norm_sun_pos),
                            (-p_srp * C_r * S_s) * (sun_pos_array[2] / norm_sun_pos)])
        r, theta, phi, _, _, _ = y
        R = sfk.eci_to_zen(theta, phi)
        F_srp_uvw = np.dot(R, F_srp_ijk)
        a_srp = [F_srp_uvw[0]/m_s, F_srp_uvw[1]/m_s, F_srp_uvw[2]/m_s]
    else: a_srp = [0,0,0]
    return a_srp

@njit
def oblateness(y):
    r, theta, phi, u, v, w = y
    acc_u = -mi/r**2 + mi/r**2*(J2*(Re/r)**2*0.5*(3*np.sin(phi)**2-1)) + mi/r*(J2*(Re**2/r**3)*(3*np.sin(phi)**2-1))
    acc_w = -mi/r**2 * (J2*(Re/r)**2 * 3*np.sin(phi)*np.cos(phi))
    return [acc_u, 0, acc_w]

def combined_equations_pert(t, y):
    r, theta, phi, u, v, w = y
    
    current_time = start_time + timedelta(seconds=t)
    et = spice.str2et(current_time.strftime('%Y-%m-%dT%H:%M:%S'))
    a_drag = drag(y, current_time, et) # [0,0,0] # 
    a_srp = solar_pressure(y, et) # [0,0,0] #
    a_g = oblateness(y) # [-mi/r**2,0,0] #
    #a_g = g_perturbation(y)
    
    # State
    [rdot, tdot, pdot] = [u, v/(r*np.cos(phi)), w/r]
    udot = a_g[0] + (v**2 + w**2)/r - a_drag[0] + a_srp[0] #+ a_g[0] -mi/r**2
    vdot = (-(u*v) + (v * w * np.tan(phi)))/r - a_drag[1] + a_srp[1] + a_g[1]
    wdot = (-(u*w) - (v**2 * np.tan(phi)))/r - a_drag[2] + a_srp[2] + a_g[2]
    ydot = [rdot, tdot, pdot, udot, vdot, wdot]
    return ydot

@njit
def combined_equations(t, y):
    r, theta, phi, u, v, w = y
    a_g = oblateness(y)
    # State
    [rdot, tdot, pdot] = [u, v/(r*np.cos(phi)), w/r]
    udot = a_g[0] + (v**2 + w**2)/r 
    vdot = (-(u*v) + (v * w * np.tan(phi)))/r 
    wdot = (-(u*w) - (v**2 * np.tan(phi)))/r + a_g[2]  
    ydot = [rdot, tdot, pdot, udot, vdot, wdot]
    return ydot

def integrate_system(flag, initial_conditions, t_final=time):
    t_span = (0, t_final)
    t_eval = np.linspace(0, t_final, nodes)
    in_cond = deepcopy(initial_conditions)
    if flag == True: #perturbations on
        sol = solve_ivp(combined_equations_pert, t_span, in_cond, method='DOP853', dense_output=True, t_eval = t_eval, rtol=1e-8, atol=1e-8)
        #sol.y[1,:] = (sol.y[1,:]/(np.pi*2) - np.floor(sol.y[1,:]/(np.pi*2)))*np.pi*2
    else: sol = solve_ivp(combined_equations, t_span, in_cond, method='DOP853', dense_output=True, t_eval = t_eval, rtol=1e-8, atol=1e-8)
    return sol

spice.unload('naif0008.tls')
spice.unload('de430.bsp')
spice.unload('earth_assoc_itrf93.tf')
spice.unload('earth_000101_240704_240410.bpc')