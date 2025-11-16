import numpy as np
import scipy as scp
import copy
from scipy.sparse.linalg import eigsh

#loop calculation

def f_kn(mN, metR, metI):
    f_kn=(mN/(16*(np.pi)**2))*((metR**2/(-metR**2+mN**2))*np.log(mN**2/metR**2) - (metI**2/(-metI**2+mN**2))*np.log(mN**2/metI**2))
    return f_kn

#loop matrix function

def lam_function(mN,metR,metI):
    lam=np.array([f_kn(mN[0], metR, metI), 
                  f_kn(mN[1], metR, metI), 
                  f_kn(mN[2], metR, metI)])
    return lam

#implementation of CI 

def casas_ibarra(parameters):

    try:
        minpar=parameters["MINPAR"]
        maj_masses=parameters["MNIN"]
        other= parameters["OTHER"]

        mEt2=minpar["mEt2Input"]
        lambda3=minpar["lambda3Input"]
        lambda4=minpar["lambda4Input"]
        lambda5=minpar["lambda5Input"]
        v=other["v"]

        mN=np.array([maj_masses["Mn(1,1)"], 
                     maj_masses["Mn(2,2)"], 
                     maj_masses["Mn(3,3)"]])
        metR = np.sqrt(mEt2**2 + (lambda3+lambda4+lambda5)*v**2)
        metI = np.sqrt(mEt2**2 + (lambda3+lambda4-lambda5)*v**2)

        #Neutrino data. Taken from NuFit http://www.nu-fit.org/?q=node/294, assuming normal ordering and atmospheric data to increase precision. Best fit ranges are taken

        m_nu_1=other["Nu_1M"]
        #angles in radians
        theta12=np.random.uniform(33.68-0.7, 33.68+0.73)*(np.pi/180)
        theta23=np.random.uniform(43.3-0.8, 43.3+1)*(np.pi/180)
        theta13=np.random.uniform(8.56-0.11, 8.45+0.11)*(np.pi/180)
        delta_cp=np.random.uniform(212-41, 212+26)*(np.pi/180)

        delta_mnu_12_2= np.random.uniform(7.49e-23-1.9e-24, 7.49e-23+1.9e-24)
        delta_mnu_31_2= np.random.uniform(2.513e-21-1.95e-23, 2.513e-21+2.1e-23)

        m_nu_2=np.sqrt(m_nu_1**2 + delta_mnu_12_2)
        m_nu_3=np.sqrt(m_nu_1**2 + delta_mnu_31_2)

        #building PMNS

        c12=np.cos(theta12)
        s12=np.sin(theta12)
        c23=np.cos(theta23)
        s23=np.sin(theta23)
        c13=np.cos(theta13)
        s13=np.sin(theta13)


        U1= np.array(
            [
                [c12, s12,0],
                [-s12,c12,0],
                [0,0,1]
            ]
        )

        U2=np.array(
            [
                [c13, 0, s13*np.exp(-1j*delta_cp)],
                [0, 1, 0],
                [-s13*np.exp(1j*delta_cp), 0, c13]
            ]
        )

        U3=np.array(
            [
                [1,0,0],
                [0, c23, s23],
                [0, -s23, c23]
            ]
        )

        UPMNS=U1@U2@U3

        #R matrix

        theta_1=np.random.uniform(0, 2*np.pi)
        theta_2=np.random.uniform(0, 2*np.pi)
        theta_3=np.random.uniform(0, 2*np.pi)

        c1=np.cos(theta_1)
        s1=np.sin(theta_1)
        c2=np.cos(theta_2)
        s2=np.sin(theta_2)
        c3=np.cos(theta_3)
        s3=np.sin(theta_3)
    

        R=np.array(
            [
                [c2*c3, -s3*c1-s1*s2*c3, s1*s3-c1*s2*c3],
                [c2*s3, c1*c3-s1*s2*s3, -s1*c3-s2*c1*s3],
                [s2, c2*s1, c2*c1]
            ]
        )

        #Neutrino mass matrix.

        M_nu_diag_sqrt=np.array([np.sqrt(m_nu_1), np.sqrt(m_nu_2), np.sqrt(m_nu_3)])
        M_nu_diag_sqrt=np.diag(M_nu_diag_sqrt)

        #loop computations

        lam_matrix = lam_function(mN, metR, metI)

        lam_matrix_inv = 1/lam_matrix
        lam_matrix_inv_sqrt = np.sqrt(lam_matrix_inv)
        lam_matrix_inv_sqrt = np.diag(lam_matrix_inv_sqrt)

    

        #Yukawa calculations

        Yn=lam_matrix_inv_sqrt @ R @ M_nu_diag_sqrt @ UPMNS.T

        #checks if Yn is properly calculated. If not, provides a zero matrix

        if Yn.shape != (3, 3):
            Yn=np.zeros((3, 3))
        
        real_part=np.real(Yn)
        imag_part=np.imag(Yn)

        #dictionary with results
        diccionario={
            "YNIN":{
                "Yn(1,1)": real_part[0, 0],
                "Yn(1,2)": real_part[0, 1],
                "Yn(1,3)": real_part[0, 2],
                "Yn(2,1)": real_part[1, 0],
                "Yn(2,2)": real_part[1, 1],
                "Yn(2,3)": real_part[1, 2],
                "Yn(3,1)": real_part[2, 0],
                "Yn(3,2)": real_part[2, 1],
                "Yn(3,3)": real_part[2, 2],
            },
            "IMYNIN":{
                "Yn(1,1)": imag_part[0, 0],
                "Yn(1,2)": imag_part[0, 1],
                "Yn(1,3)": imag_part[0, 2],
                "Yn(2,1)": imag_part[1, 0],
                "Yn(2,2)": imag_part[1, 1],
                "Yn(2,3)": imag_part[1, 2],
                "Yn(3,1)": imag_part[2, 0],
                "Yn(3,2)": imag_part[2, 1],
                "Yn(3,3)": imag_part[2, 2],
            }
        }

        return diccionario
    except KeyError as e:
        print(f"KeyError: {e}. Please check the input parameters. Yn taken as zero.")
        Yn=np.zeros((3, 3))
        diccionario={
            "YNIN":{
                "Yn(1,1)": 0.0,
                "Yn(1,2)": 0.0,
                "Yn(1,3)": 0.0,
                "Yn(2,1)": 0.0,
                "Yn(2,2)": 0.0,
                "Yn(2,3)": 0.0,
                "Yn(3,1)": 0.0,
                "Yn(3,2)": 0.0,
                "Yn(3,3)": 0.0,
            },
            "IMYNIN":{
                "Yn(1,1)": 0.0,
                "Yn(1,2)": 0.0,
                "Yn(1,3)": 0.0,
                "Yn(2,1)": 0.0,
                "Yn(2,2)": 0.0,
                "Yn(2,3)": 0.0,
                "Yn(3,1)": 0.0,
                "Yn(3,2)": 0.0,
                "Yn(3,3)": 0.0,
            }
        }
        return diccionario


