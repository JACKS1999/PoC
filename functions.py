import numpy as np
import constants
from numba import njit

def r_conicp(p,e,nu):
    return p/(1+e*np.cos(nu))

def eci_to_perifocal(Omega, inc, omega):
    # Combinazione delle rotazioni: Omega (asse Z), inc (asse X), omega (asse Z)
    Rz1 = euler_rotation_matrix(Omega, 'z')
    Rx = euler_rotation_matrix(inc, 'x')
    Rz2 = euler_rotation_matrix(omega, 'z')
    rot_matrix = np.dot(np.dot(Rz2, Rx), Rz1)
    return rot_matrix

def eci_to_zen(theta, phi):
    Rz = euler_rotation_matrix(theta, 'z')
    Ry = euler_rotation_matrix(phi, 'y')
    rot_matrix = np.dot(Ry,Rz)
    return rot_matrix

def euler_rotation_matrix(angle, axis):
    """
    Genera una matrice di rotazione di Eulero.

    Args:
    angle (float): Angolo di rotazione in radianti.
    axis (str): Asse di rotazione ('x', 'y' o 'z').

    Returns:
    np.array: Matrice di rotazione 3x3.
    """
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        return np.array([
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("L'asse deve essere 'x', 'y' o 'z'.")

def perifocal_to_eci(Omega, inc, omega):
    return np.transpose(eci_to_perifocal(Omega, inc, omega))

def cartesian_to_polar(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arcsin(z/r)
    #theta = np.arccos(x/(r*np.cos(phi)))
    theta = np.arctan2(y, x)
    if theta <0: theta = np.pi*2+theta
    return [r, theta, phi]

def keplerian_elements_to_state(initial_keplerian_elements, mu):
    """
    Calcola i vettori di posizione e velocità da elementi orbitali di Kepler.
    
    Args:
    a (float): Semiasse maggiore in km.
    ecc (float): Eccentricità dell'orbita.
    inc (float): Inclinazione dell'orbita in gradi.
    Omega (float): Longitudine del nodo ascendente in gradi.
    omega (float): Argomento del perigeo in gradi.
    nu (float): Anomalia vera in gradi.

    Returns:
    np.array, np.array: Vettori di posizione e velocità.
    """
    a =  initial_keplerian_elements[0]
    ecc = initial_keplerian_elements[1]
    inc = initial_keplerian_elements[2]
    Omega = initial_keplerian_elements[3]
    omega = initial_keplerian_elements[4]
    nu = initial_keplerian_elements[5]

    inc *= constants.DEG2RAD
    Omega *= constants.DEG2RAD
    omega *= constants.DEG2RAD
    nu *= constants.DEG2RAD

    p = a * (1 - ecc**2)
    r = r_conicp(p,ecc,nu)

    r_perifocal = np.array([(r * np.cos(nu)),
                            (r * np.sin(nu)),
                            0])

    v_perifocal = np.array([-np.sqrt(mu / p) * np.sin(nu),
                            np.sqrt(mu / p) * (ecc + np.cos(nu)),
                            0])
    R1 = perifocal_to_eci(Omega,inc,omega)
   
    r_eci = np.dot(R1, r_perifocal)
    v_eci = np.dot(R1, v_perifocal)
    #r_polar = cartesian_to_polar(r_eci[0],r_eci[1],r_eci[2])
    #R2 = eci_to_zen(r_polar[1],r_polar[2])
    #v_zen = np.dot(R2, v_eci)
    state = np.hstack([r_eci, v_eci])
    return state

def keplerian_elements_to_rp_ra(keplerian_elements):
    a =  keplerian_elements[0]
    ecc = keplerian_elements[1]
    inc = keplerian_elements[2]
    Omega = keplerian_elements[3]
    omega = keplerian_elements[4]
    nu = keplerian_elements[5]

    rp = a*(1-ecc)
    ra = a*(1+ecc)
    return rp, ra

@njit(nopython = True)
def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

@njit(nopython = True)
def norm(v):
    return (v[0]**2 + v[1]**2 + v[2]**2)**0.5

def perigee_eccentricity_to_semimajoraxis(rp, e):
    a = rp / (1-e)
    return a

def rev_to_semilatus(n):
    T = 3600*24/n
    a = ((T/(2*np.pi))**2 * constants.MU_EARTH)**(1/3)
    return a

def energ(r, v, mi):
    return v**2/2 - mi/r

def a_from_energ(E, mi):
    return -mi/(2*E)

def state_to_keplerian_elements(state, mu, printdeg = 'true'):
    
    r_vec = state[0:3]
    v_vec = state[3:]

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    k_hat = np.array([0, 0, 1])
    n_vec = np.cross(k_hat, h_vec)
    n = np.linalg.norm(n_vec)

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    Energ = energ(r,v, mu)
    a = a_from_energ(Energ, mu)

    e_vec = (1 / mu) * ((v**2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec)
    ecc = np.linalg.norm(e_vec) * constants.RAD2DEG

    inc = np.arccos(h_vec[2] / h) * constants.RAD2DEG

    if inc ==0: Omega =0
    else: Omega = np.arccos(n_vec[0] / n) * constants.RAD2DEG
    if n_vec[1] < 0:
        Omega = 360 - Omega

    if n == 0: omega = 0
    else: omega = np.arccos(np.dot(n_vec, e_vec) / (n * ecc)) * constants.RAD2DEG
    if e_vec[2] < 0:
        omega = 360 - omega

    nu = np.arccos(np.dot(e_vec, r_vec) / (ecc * r)) * constants.RAD2DEG
    if np.dot(r_vec, v_vec) < 0:
        nu = 360 - nu

    keplerian_elements = np.array([a, ecc, inc, Omega, omega, nu])
    return keplerian_elements