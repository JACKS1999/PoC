import numpy as np

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

def eci_to_UVW(inc, theta_r):
    Rx = euler_rotation_matrix(inc, 'x')
    Ry = euler_rotation_matrix(theta_r, 'y')
    rot_matrix = np.dot(Ry, Rx)
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



def state_vectors_from_orbital_elements(a, ecc, inc, Omega, omega, nu, mu):
    """
    Calcola i vettori di posizione e velocità da elementi orbitali di Kepler.
    
    Args:
    a (float): Semiasse maggiore in metri.
    ecc (float): Eccentricità dell'orbita.
    inc (float): Inclinazione dell'orbita in radianti.
    Omega (float): Longitudine del nodo ascendente in radianti.
    omega (float): Argomento del perigeo in radianti.
    nu (float): Anomalia vera in radianti.

    Returns:
    np.array, np.array: Vettori di posizione e velocità.
    """

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
    r_vec = cartesian_to_polar(r_eci[0],r_eci[1],r_eci[2])
    R2 = eci_to_zen(r_vec[1],r_vec[2])
    v_vec = np.dot(R2, v_eci)
    
    return r_vec, v_vec

def ijkzen_from_kepler(a, ecc, inc, Omega, omega, nu, mu):
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
    r_vec = cartesian_to_polar(r_eci[0],r_eci[1],r_eci[2])
    R2 = eci_to_zen(r_vec[1],r_vec[2])
    v_vec = np.dot(R2, v_eci)
    return r_eci, v_vec

def ijk_from_kepler(a, ecc, inc, Omega, omega, nu, mu):
    p = a * (1 - ecc**2)
    r = r_conicp(p,ecc,nu)

    r_perifocal = np.array([(r * np.cos(nu)),
                            (r * np.sin(nu)),
                            0])

    v_perifocal = np.array([-np.sqrt(mu / p) * np.sin(nu),
                            np.sqrt(mu / p) * (ecc + np.cos(nu)),
                            0])
    R1 = perifocal_to_eci(Omega,inc,omega)
   
    r_ijk = np.dot(R1, r_perifocal)
    v_ijk = np.dot(R1, v_perifocal)
    return r_ijk, v_ijk

def state_vectors_from_pos(x, y, z, i, Om, om, n, r_eci, mu):
    a = np.sqrt(x**2 + y**2 + z**2)
    ecc = 0
    inc = i + np.arcsin((z-r_eci[2])/a)
    Omega = Om + np.arcsin((y-r_eci[1])/a)
    omega = om 
    nu = n
    r_pol, v_zen = state_vectors_from_orbital_elements(a, ecc, inc, Omega, omega, nu, mu)
    return r_pol, v_zen

'''def state_vectors_from_pos(x, y, z, mu):
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arcsin(z/r)
    theta = np.arctan2(y, x)
    v = np.sqrt(mu/r)
    u = 0
    w = 0
    return [r, theta, phi], [u, v, w]'''

''''''
def energ(r, v, mi):
    return v**2/2 - mi/r

def a_from_energ(E, mi):
    return -mi/(2*E)

def orbital_elements_from_state_vectors(r_vec, v_vec, mu, printdeg = 'true'):
    """
    Calcola gli elementi orbitali di Kepler da vettori di posizione (r_vec) e velocità (v_vec).
    
    Args:
    r_vec (np.array): Vettore di posizione [x, y, z] in metri.
    v_vec (np.array): Vettore di velocità [vx, vy, vz] in metri al secondo.

    Returns:
    dict: Dizionario contenente gli elementi orbitali (a, ecc, inc, Omega, omega, nu).
    """
    
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
    ecc = np.linalg.norm(e_vec)

    inc = np.arccos(h_vec[2] / h)

    Omega = np.arccos(n_vec[0] / n)
    if n_vec[1] < 0:
        Omega = 2 * np.pi - Omega

    omega = np.arccos(np.dot(n_vec, e_vec) / (n * ecc))
    if e_vec[2] < 0:
        omega = 2 * np.pi - omega

    nu = np.arccos(np.dot(e_vec, r_vec) / (ecc * r))
    if np.dot(r_vec, v_vec) < 0:
        nu = 2 * np.pi - nu

    return a, ecc, inc, Omega, omega, nu