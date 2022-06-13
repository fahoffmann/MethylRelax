#!/home/hoffmann/anaconda2/bin/python2.7
# -*- coding: utf-8 -*-

import sys
import numpy as np
import npufunc
import transforms3d.quaternions as qops

class gyromag:
    def __init__(self, isotope):
        self.name=isotope
        self.csa=0.0
        self.timeUnit='s'
        self.time_fact=_return_time_fact(self.timeUnit)
        self.set_gamma(self.name)

    def set_gamma(self, name):
        """
        Sets the gamma value based on the gyromag's isotope and time unit definition.
        Gamma is in units of rad s^-1 T^-1 .
        """
        if name=='1H':
            self.gamma=267.513e6*self.time_fact
            #In standard freq: 42.576e6
        elif name=='2H':
            self.gamma=42.065e6*self.time_fact
        elif name=='13C':
            self.gamma=67.262e6*self.time_fact
            # Good average unknown
            self.csa=-130e-6
        elif name=='15N':
            self.gamma=-27.116e6*self.time_fact
            # Large variation possible. See Fushman, Tjandra, and Cowburn, 1998.
            # Also use more commonly accepted value.
            self.csa=-170e-6
        elif name=='17O':
            self.gamma=-36.264e6*self.time_fact
        elif name=='19F':
            self.gamma=251.662e6*self.time_fact
        elif name=='31P':
            self.gamma=108.291e6*self.time_fact

    def set_time_unit(self, tu):
        old = self.time_fact
        self.time_fact = _return_time_fact(tu)
        mult = self.time_fact/old
        self.gamma *= mult

class diffusionModel:
    def __init__(self, model, *args):
        self.timeUnit=args[0]
        self.time_fact=_return_time_fact(self.timeUnit)
        if   model=='direct_transform':
            # Dummy entry for diffusion model with np.nan to ensure errors.
            self.name='direct_transform'
            self.D=np.nan
        elif model=='rigid_sphere_T':
            self.name='rigid_sphere'
            self.D=1.0/6.0*float(args[1])
            self.D_coeff=self.D
            self.D_delta=0.0
            #self.A_coeff=1.0
        elif model=='rigid_sphere_D':
            self.name='rigid_sphere'
            self.D=float(args[1])
            self.D_coeff=self.D
            self.D_delta=0.0
            #self.A_coeff=1.0
        elif model=='rigid_symmtop_Dref':
            #Dperp = 3.*Diso/(2+aniso)
            #Dpar  = aniso*Dperp
            self.name='rigid_symmtop'
            self.D=np.zeros(2)
            #self.D[2]=args[1]
            self.D[1]=3.0*args[1]/(2.0+args[2])
            self.D[0]=args[2]*self.D[1]
            self.D_coeff=D_coefficients_symmtop(self.D)
            self.D_delta=0.0
            # Convention: Diso, Dani --> Dpar, Dperp
        elif model=='rigid_symmtop_D':
            self.name='rigid_symmtop'
            self.D=np.zeros(2)
            self.D[0]=args[1]
            self.D[1]=args[2]
            self.D_coeff=D_coefficients_symmtop(self.D)
            self.D_delta=0.0
            # Convention: Dpar, Dperp
        elif model=='rigid_symmtop_T':
            self.name='rigid_symmtop'
            self.D=calculate_Dglob( (args[1], args[2], args[3]) )
            self.D_coeff=D_coefficients_symmtop(self.D)
            self.D_delta=0.0
            # Convention:
        elif model=='rigid_ellipsoid_D':
            self.name='rigid_ellipsoid'
            self.D=np.zeros(3)
            self.D[0]=args[1]
            self.D[1]=args[2]
            self.D[2]=args[3]
            # Convention: D0 <= D1 <= D2
            self.D_coeff, self.D_delta = D_coefficients_ellipsoid(D, True)

    def obtain_A_coeff(self, v):
        if model=='rigid_sphere':
            return 1.0
        if model=='rigid_symmtop':
            return A_coefficients_symmtop(v, bProlate=(self.D[0]>self.D[1]))
        if model=='rigid_ellipsoid':
            return A_coefficients_ellipsoid(v, self.D_delta, True)

    def set_time_unit(self, tu):
        old = self.time_fact
        self.time_fact = _return_time_fact(tu)
        mult = self.time_fact/old
        self.D *= mult
        self.D_coeff *= mult
        self.D_delta   *= mult

    def change_Diso(self, Diso):
        if self.name=='rigid_sphere':
            self.D=Diso
            self.D_coeff=self.D
            self.D_delta=0.0
        elif self.name=='rigid_symmtop':
            tmp=self.D[0]/self.D[1]
            self.D[1]=3.0*Diso/(2.0+tmp)
            self.D[0]=tmp*self.D[1]
            self.D_coeff=D_coefficients_symmtop(self.D)
            self.D_delta=0.0
        elif self.name=='rigid_ellipsoid':
            print >> sys.stderr, "= = ERROR: change_Diso for fully anisotropic models, not implemented."
            sys.exit(1)

class relaxationModel:
    """
    Help for class relaxationModel:
    This is the overall handling class used to compute spin relaxations from trajectories.
    It collects a number of smaller classes that are responsible for functionally distinct sub-components,
    i.e.:
    - the NMR measurements, which handles frequencies, NH types, and spins.
    - the molecule, which handles sequence information, domain definitions, rotational diffusion models, vector data

    This overall class contains the following functions:
    - the computing and fitting procedures that need the above sub-classes.

    Attributes:
        bond - Name of the intended vector bond, eg 'NH'
        B_0  - Background magnetic field, in Teslas.
        timeUnit - Units of time used for the freqency, such as 'ns', or 'ps'.
        rotdifModel - The Rotational Diffusion model used to define spectral densitiies J(omega)
    """

    # = = = Static class variables = = =
    # Default indexing to create the five NMR-component frequencies
    # J(0), J(wX), J(wH-wX), J(wH), J(wH+wX)
    # In ascending order. This is used to obtain relaxation properties from a five-value J(function)
    iOmX = 1
    iOmH = 3

    def __init__(self, bondType, B_0):
        # Parameters associated with units
        self.omega=np.array(0.0)
        self.timeUnit='ns'
        self.time_fact=_return_time_fact(self.timeUnit)
        self.dist_unit='nm'
        self.dist_fact=_return_dist_fact(self.dist_unit)

        # Atomic parameters.
        self.bondType = bondType
        self.B_0  = B_0
        if   bondType=='NH':
            self.gH  = gyromag('1H')
            self.gX  = gyromag('15N')
            # = = Question use conventional 1.02, or 1.04 to include librations, according to Case, J Biomol. NMR, 1999? = = =
            self.rXH = 1.02e-1
            #self.rXH = 1.04e-1
        elif bondType=='CH':
            self.gH  = gyromag('1H')
            self.gX  = gyromag('13C')
            # Need to update bond length for this atoms
            self.rXH = 1.02e-1
        elif bondType=='methyl':
            self.gH  = gyromag('1H')
            self.gX  = gyromag('13C')
            #The gemetry used for methyl group based on Zhang, JACS, 2006.
            self.rXH = 1.115e-1
            self.rHH = 1.821e-1 
            
        else:
            print >> sys.stderr, "= = ERROR in relaxationModel: wrong bondType definition! = =" % bondType
            sys.exit(1)

        # relaxation model. Note in time units of host object.
        self.set_rotdif_model('rigid_sphere_T', 1.0)

    def set_B0(self, B_0):
        self.B_0 = B_0

    def set_time_unit(self, tu):
        old=self.time_fact
        self.time_fact = _return_time_fact(tu)
        # Update all time units can measurements.
        self.timeUnit=tu
        mult = self.time_fact / old
        self.omega *= mult
        self.rotdifModel.set_time_unit(tu)
        #self.gH.set_time_unit(tu) - leave immune to time unit changes!
        #self.gX.set_time_unit(tu)

    def set_freq_relaxation(self):
        """
        This function sets the 5 frequencies of nuclei at the instance's magnetic field,
        in the following order:
            - 0.0 , omega_X, omega_H-omega_X, omega_H, omega_H+omega_X.
        This order will be used for calculating J(w) and relaxaion values.
        """
        self.num_omega=5
        self.omega=np.zeros(self.num_omega)
        iOmH = 3 ; # indexing for the frequencies.
        iOmX = 1 ; #
        # First determine the frequencies omega and J from given inputs.
        self.omega[iOmH] = -1.0*self.gH.gamma*self.B_0*self.time_fact # for 1H
        self.omega[iOmX] = -1.0*self.gX.gamma*self.B_0*self.time_fact # for 13C
        self.omega[iOmH-iOmX] = (self.omega[iOmH]-self.omega[iOmX]) 
        self.omega[iOmH+iOmX] = (self.omega[iOmH]+self.omega[iOmH]) # for 2* 1H
        #self.omega[0] = 0.0*self.gH.gamma*self.B_0*self.time_fact
        #self.omega[1] = 1.0*self.gH.gamma*self.B_0*self.time_fact
        #self.omega[2] = 2.0*self.gH.gamma*self.B_0*self.time_fact

    def print_freq_order(self):
        print "omega=(0, iOmX, iOmH-iOmX, iOmH, iOmH+iOmX )"

    def set_freq_defined(self, wmin, wmax, wstep):
        self.omega=np.arange(wmin, wmax, wstep)
        self.num_omega=len(self.omega)

    def set_rotdif_model(self, model, *args):
        """
        Define the relaxation model for this object, taking as arguments the global parameters necessary
        to define each model. Available models are:
            - rigid_sphere_T, tau_iso
            - rigid_sphere_D, D_iso
            - rigid_symmtop_D, D_par, D_perp
            - rigid_ellipsoid_D, Dx, Dy, Dz
        """
        self.rotdifModel=diffusionModel(model, self.timeUnit, *args)

    def get_Jomega(self, vNH):
        """
        Calculate the spectral density function for a given set of unit vectors and the current model.
        """
        num_vecs = len(vNH)
        J = np.zeros( (num_vecs, self.num_omega) )
        for i in range(num_vecs):
            J[i] = function_to_be_written(vNH[i])
        return 'Not composed'

    def get_relax_from_J(self, J1, J2, J3, J4, J5,J6, CSAvalue=None):
        """
        The maths behind this is:
        f_DD  = 0.25* (mu_0*hbar/4.0/pi)**2 * gamma_13C**2 * gamma_1H**2 * r_CH**-6.0
        f_DD2  = 0.25* (mu_0*hbar/4.0/pi)**2 * gamma_1H**4 * r_HH**-6.0
        f_DD3  = 1/45* (mu_0*hbar/4.0/pi)**2 * gamma_13C**2 * gamma_1H**2 * r_CH**-6.0
        f_DD4  = 1/45* (mu_0*hbar/4.0/pi)**2  gamma_1H**4 * r_HH**-6.0
	mu_0 = 4*pi*1e-7      ; # m   kg s^-2 A-2
        hbar = 1.0545718e-34  ; # m^2 kg s^-1
        pi   = 3.14159265359
        gamma_1H  = 267.513e6  ; # rad s^-1 T^-1
        gamma_13C = 67.262e6  ; # rad s^-1 T^-1 
        omega_13C = - gamma_15N * B_0 .
        r_CH = 1.115e-10 ;# m
        r_HH = 1.821e-10 ;# m
         (mu_0*hbar/4.0/pi)**2 m^-1 s^2 is the 10^-82 number below. f_DD and f_CSA are maintained in SI units.
	     Liu Journal of Biomolecular NMR 27: 351–364, 2003 
	     Zhang JACS 2006 
	     Sabo the protein socity 2012
	Please note:
		1) The prefactor (2/5) is included in the spectral density function for CCR, but not included in the spectral density of CCR_{rigid}.
		2) The protons contribution is negilable for CCR_{rigid} calculation, due to its small contribution.
        """
        iOmX = 1; iOmH = 3

        f_DD = 0.25 * 1.1121216813552401e-82*self.gH.gamma**2.0*self.gX.gamma**2.0 *(self.rXH*self.dist_fact)**-6.0
        f_DD2 = 0.25 * 1.1121216813552401e-82*self.gH.gamma**4.0 *(self.rHH*self.dist_fact)**-6.0
        f_DD3 = 1.0/45.0 * 1.1121216813552401e-82*self.gH.gamma**2.0*self.gX.gamma**2.0 *(self.rXH*self.dist_fact)**-6.0
        f_DD4 = 1.0/45.0 * 1.1121216813552401e-82*self.gH.gamma**4.0 *(self.rHH*self.dist_fact)**-6.0
        if CSAvalue is None:
            f_CSA = 2.0/15.0 * self.gX.csa**2.0 * ( self.gX.gamma * self.B_0 )**2
        else:
            f_CSA = 2.0/15.0 * CSAvalue**2.0 * ( self.gX.gamma * self.B_0 )**2

        # Note: since J is in the units of inverse time, data needs to be converted back to s^-1
        ccr = self.time_fact*( f_DD*( 4*J2[0] + 3*J2[iOmX]) + f_DD2*(3*J3[iOmH] + 3*J1[iOmH+iOmX]))
        ccr_rigid = self.time_fact*(( f_DD3*J4[iOmX] ))#+ f_DD4*((1.5*J5[iOmH])+(1.5*J6[iOmH])))
        order = ccr/ccr_rigid
        return ccr, ccr_rigid, order



    def get_relax_from_J_simd(self, J1, J2, J3, J4, axis=-1, CSAvalue=None):
    
    	"""
    	NOTE: This script so far calculates the rigid limit of CCRR by using the isotropic diffusion constant. It is not implemented (yet) the residue specific rigid limit of CCRR.
    	"""
        iOmX = 1; iOmH = 3

        f_DD = 0.25 * 1.1121216813552401e-82*self.gH.gamma**2.0*self.gX.gamma**2.0 *(self.rXH*self.dist_fact)**-6.0
        f_DD2 = 0.25 * 1.1121216813552401e-82*self.gH.gamma**4.0 *(self.rHH*self.dist_fact)**-6.0
        f_DD3 = 1.0/45.0 * 1.1121216813552401e-82*self.gH.gamma**2.0*self.gX.gamma**2.0 *(self.rXH*self.dist_fact)**-6.0
        f_DD4 = 1.0/45.0 * 1.1121216813552401e-82*self.gH.gamma**4.0 *(self.rHH*self.dist_fact)**-6.0        
        pref = 2 * np.pi * 167000
        if CSAvalue is None:
            f_CSA = 2.0/15.0 * self.gX.csa**2.0 * ( self.gX.gamma * self.B_0 )**2
        else:
            f_CSA = 2.0/15.0 * CSAvalue**2.0 * ( self.gX.gamma * self.B_0 )**2

        if axis==-1:
            ccr = self.time_fact*( f_DD*( 4*J2[...,0] + 3*J2[...,iOmX]) + f_DD2*(3*J3[...,iOmH]  + 3*J1[...,iOmH+iOmX] ))
            ccr_rigid = self.time_fact*( f_DD3*J4[...,iOmX])#+ f_DD4*(3.0/2.0*J5[...,iOmH]+9.0/2.0*J5[...,iOmH+iOmX]))
            order = ccr/ccr_rigid
        elif axis==0:
            ccr = self.time_fact*( f_DD*( 4*J2[0,...] + 3*J2[iOmX,...]) + f_DD2*(3*J3[iOmH,...] + 3*J1[iOmH+iOmX,...] ))
            ccr_rigid = self.time_fact*( f_DD3*J4[iOmX,...])# + f_DD4*(3.0/2.0*J5[iOmH,...]+9.0/2.0*J5[iOmH+iOmX,...]))
            order = ccr/ccr_rigid

        return ccr, ccr_rigid, order


    def _get_f_DD(self):
        return 0.10 * 1.1121216813552401e-82*self.gH.gamma**2.0*self.gX.gamma**2.0 *(self.rXH*self.dist_fact)**-6.0

    def _get_f_CSA(self):
        return 2.0/15.0 * self.gX.csa**2.0 * ( self.gX.gamma * self.B_0 )**2

    def get_R1(self, J):
        f_DD = _get_f_DD() ; f_CSA = _get_f_CSA()
        return self.time_fact*( f_DD*( J[iOmH-iOmX] + 3*J[iOmX] + 6*J[iOmH+iOmX] ) + f_CSA*J[iOmX] )

    def get_R2(self, J):
        f_DD = _get_f_DD() ; f_CSA = _get_f_CSA()
        return self.time_fact*( 0.5*f_DD*( 4*J[0] + J[iOmH-iOmX] + 3*J[iOmX] + 6*J[iOmH+iOmX] + 6*J[iOmH] ) + 1.0/6.0*f_CSA*(4*J[0] + 3*J[iOmX] ) )

    def get_NOE(self, J):
        f_DD = _get_f_DD() ; f_CSA = _get_f_CSA()
        return 1.0 + self.time_fact * self.gH.gamma/(self.gX.gamma*R1) * f_DD*(6*J[iOmH+iOmX] - J[iOmH-iOmX])

    def get_etaZ(self, J, beta=0.0):
        """
        Notation following that of Kroenke et al., JACS 1998. Eq. 2
        Here, beta is the angle (in radians) between the symmetry axis of the CSA tensor and the N-H bond.
        """
        # mu_0 hbar / 4_pi = hbar* 10^-7
        fact = -1.0545718e-41*self.gH.gamma*self.gX.gamma**2.0*(self.rXH*self.dist_fact)**-3.0 * self.B_0*self.gX.csa* 0.4
        return fact*(1.5*cos(beta)-0.5)*J[iOmX]

    def get_etaXY(self, J, beta=0.0):
        """
        Notation following that of Kroenke et al., JACS 1998.
        Here, beta is the angle (in radians) between the symmetry axis of the CSA tensor and the N-H bond.
        """
        fact = -1.0545718e-41*self.gH.gamma*self.gX.gamma**2.0*(self.rXH*self.dist_fact)**-3.0 * self.B_0*self.gX.csa* 0.4
        return fact/6.0*(1.5*cos(beta)-0.5)*( 4.0*J[0] + 3.0*J[iOmX] )

    def get_rho_from_J(self, J):
        """
        Taking Eq. 4 of Ghose, Fushman and Cowburn (2001), and define rho as a ratio of modified R1' and R2'
        that have high frequency components removed.
        """
        return J[self.iOmX]/J[0]

    def get_rho_from_J_simd(self, J, axis=-1):
        if axis == -1:
            return J[...,self.iOmX]/J[...,0]
        elif axis == 0:
            return J[self.iOmX,...]/J[0,...]

    def calculate_rho_from_relaxation(self, rvec, drvec=[] ):
        """
        Taking Eq. 4 of Ghose, Fushman and Cowburn (2001), calculate rho from R1, R2, and NOE directly,
        rather than from the spectral density J(omega). This is used to convert experimental measurements to rho.
        rvec is the triple of (R1, R2, NOE)
        Error is known to be bad.
        """
        if  drvec==[]:
            R1=rvec[0] ; R2=rvec[1] ; NOE=rvec[2]
            HF  = -0.2*(self.gX.gamma/self.gH.gamma)*(1-NOE)*R1
            R1p = R1 - 7.0*(0.921/0.87)**2.0*HF
            R2p = R2 - 6.5*(0.955/0.87)**2.0*HF
            return 4.0/3.0*R1p/(2.0*R2p-R1p)
        else:
            R1=rvec[0]  ;  R2=rvec[1]  ;  NOE=rvec[2]
            dR1=drvec[0] ; dR2=drvec[1] ; dNOE=drvec[2]
            HF  = -0.2*(self.gX.gamma/self.gH.gamma)*(1-NOE)*R1
            R1p = R1 - 7.0*(0.921/0.87)**2.0*HF
            R2p = R2 - 6.5*(0.955/0.87)**2.0*HF
            rho  = 4.0/3.0*R1p/(2.0*R2p-R1p)
            drho = 0
            print "= = ERROR: drho calculation is not implemented!"
            sys.exit(1)
            return (rho, drho)




# --------------------------------------------------------------------------------------------------------------#

def rotate_vector_simd(v, q, axis=-1, bNormalised=False):
    """
    Alternative formulation of quaternion multiplication on a set of vectors that I hope should be quicker.
    Uses numpy broadcasting. So allowed combination are 1 vector/ ND-vectors with 1 quaternion/ND-quaternions,
    as long a ND are the same shape in the non-vector components.

    For all rotations, q must be normalised to a unit quaternion.

    The axis vector components must be either the first or last axis to satisfy broadcasting.
    i.e. v[...,3] & q[..,4] or alternatively v[3,...] & q[4,...]
    """

    if not bNormalised:
        q = vecnorm_NDarray(q)
    v=np.array(v)
    q_w, q_v = decompose_quat(q, axis=axis, bReshape=True)
    # = = = Determine a
    if axis==-1:
        tmp = np.cross(q_v, v, axisa=axis, axisb=axis) + q_w[...,None]*v
    elif axis==0:
        tmp = np.cross(q_v, v, axisa=axis, axisb=axis) + q_w[None,...]*v
    else:
        print sys.stderr, "= = = ERROR: rotate_vector_simd does not support arbitrary axis definitions."
        sys.exit(1)
    # = = = Determine b
    tmp = np.cross(q_v, tmp, axisa=axis, axisb=axis)
    return tmp+tmp+v

def read_vector_distribution_from_file( fileName ):
    """
    Returns the vectors, and mayber weights whose dimensions are (nResidue, nSamples, 3).
    Currently supports only phi-theta formats of vector definitions.
    For straight xmgrace data files, this corresponds to the number of plots, then the data-points in each plot.
    """
    weights = None
    if fileName.endswith('.npz'):
        # = = = Treat as a numpy binary file.
        obj = np.load(fileName, allow_pickle=True)
        # = = = Determine data type
        resIDs = obj['names']
        if obj['bHistogram']:
            if obj['dataType'] != 'LambertCylindrical':
                print >> sys.stderr, "= = = Histogram projection not supported! %s" % obj['dataType']
                sys.exit(1)
            vecs, weights = convert_LambertCylindricalHist_to_vecs(obj['data'], obj['edges'])
        else:
            if obj['dataType'] != 'PhiTheta':
                print >> sys.stderr, "= = = Numpy binary datatype not supported! %s" % obj['dataType']
                sys.exit(1)
            # = = = Pass phi and theta directly to rtp_to_xyz
            vecs = rtp_to_xyz( obj['data'], vaxis=-1, bUnit=True )
    else:
        resIDs, dist_phis, dist_thetas, dum = gs.load_sxydylist(args.distfn, 'legend')
        vecs = rtp_to_xyz( np.stack( (dist_phis,dist_thetas), axis=-1), vaxis=-1, bUnit=True )
    if not weights is None:
        print "    ...converted input phi_theta data to vecXH / weights, whose shapes are:", vecs.shape, weights.shape
    else:
        print "    ...converted input phi_theta data to vecXH, whose shape is:", vecs.shape
    return resIDs, vecs, weights

def load_xys(fn):
    x=[]
    y=[]
    for l in open(fn):
        if l[0]=="#" or l[0]=="@" or l[0]=="&" or l=="":
            continue
        vars = [ float(i) for i in l.split() ]
        x.append(vars[0])
        y.append(vars[1:])
    return np.array(x), np.array(y)

def D_coefficients_symmtop(D):
    """
    Computes the 3 axisymmetric D-coefficients associated with the D_rot ellipsoid.
    """
    Dpar=D[0]
    Dperp=D[1]
    D_J=np.zeros(3)
    D_J[0]= 5*Dperp +   Dpar
    D_J[1]= 2*Dperp + 4*Dpar
    D_J[2]= 6*Dperp
    return D_J

def _return_time_fact(tu):
    if tu=='ps':
        return 1.0e-12
    elif tu=='ns':
        return 1.0e-9
    elif tu=='us':
        return 1.0e-6
    elif tu=='ms':
        return 1.0e-3
    elif tu=='s':
        return 1.0e-0
    else:
        print >> sys.stderr, "= = ERROR in object definition: invalid time unit definition!"
        return

def _return_dist_fact(du):
    if du=='pm':
        return 1.0e-12
    elif du== 'A':
        return 1.0e-10
    elif du=='nm':
        return 1.0e-9
    elif du=='um':
        return 1.0e-6
    elif du=='mm':
        return 1.0e-3
    elif du=='m':
        return 1.0e-0
    else:
        print >> sys.stderr, "= = ERROR in relaxationModel: invalid distance unit definition!"
        return

def J_direct_transform(om, consts, taus):
    """
    This calculates the direct fourier transform of C(t) without a global tumbling factor.
    In this case the order parameter makes no contributions whatsoever?
    """
    ndecay=len(consts) ; noms=len(om)
    Jmat = np.zeros( (ndecay, noms ) )
    for i in range(ndecay):
        Jmat[i] = consts[i]*taus[i] /(1 + (taus[i]*om)**2.)
    return Jmat.sum(axis=0)


def _obtain_Jomega(RObj, nSites, S2, consts, taus, vecXH, weights=None):
    """
    The inputs vectors have dimensions (nSites, nSamples, 3) or just (nSites, 3)
    the datablock being returned has dimensions:
    - (nFrequencies, nSites)    of there is no uncertainty calculations. 5 being the five frequencies J(0), J(wN) J(wH+wN), J(wH), J(wH-wN)
    - (nFrequencies, nSites, 2) if there is uncertainty calculations.
    """
    nFrequencies= len(RObj.omega)
    #nFrequencies= np.arange(0,0.002, 0.00001)
    if RObj.rotdifModel.name == 'rigid_sphere':
        datablock=np.zeros((5,nSites), dtype=np.float32)
        #datablock=np.zeros((5,nSites), dtype=np.float32)
        #datablock=np.zeros((len(nFrequencies),nSites), dtype=np.float32)
        for i in range(nSites):
            J = J_combine_isotropic_exp_decayN(RObj.omega, 1.0/(6.0*RObj.rotdifModel.D), S2[i], consts[i], taus[i])
            #J = J_combine_isotropic_exp_decayN(nFrequencies, 1.0/(6.0*RObj.rotdifModel.D), S2[i], consts[i], taus[i])
            datablock[:,i]=J
        return datablock
    elif RObj.rotdifModel.name == 'rigid_symmtop':
        # Automatically use the vector-form of function.
        if len(vecXH.shape) > 2:
            # An ensemble of vectors at each site. Measure values for all of them then average with/without weights.
            datablock=np.zeros( (nFrequencies, nSites, 2), dtype=np.float32)
            npts=vecXH.shape[1]
            for i in range(nSites):
                # = = = Calculate at each residue and sum over Jmat( nSamples, 2)
                Jmat = J_combine_symmtop_exp_decayN(RObj.omega, vecXH[i], RObj.rotdifModel.D[0], RObj.rotdifModel.D[1], S2[i], consts[i], taus[i])
                if weights is None:
                    datablock[:,i,0] = np.mean(Jmat, axis=0)
                    datablock[:,i,1] = np.std(Jmat, axis=0)
                else:
                    datablock[:,i,0] = np.average(Jmat, axis=0, weights=weights[i])
                    datablock[:,i,1] = np.sqrt( np.average( (Jmat - datablock[:,i,0])**2.0, axis=0, weights=weights[i]) )
            return datablock
        else:
            #Single XH vector at each site. Ignore weights, as they should not exist.
            datablock=np.zeros((5,nSites), dtype=np.float32)
            for i in range(nSites):
                Jmat = J_combine_symmtop_exp_decayN(RObj.omega, vecXH[i], RObj.rotdifModel.D[0], RObj.rotdifModel.D[1], S2[i], consts[i], taus[i])
                datablock[:,i]=Jmat
            return datablock

    # = = Should only happen with fully anisotropic models.
    print >> sys.stderr, "= = ERROR: Unknown rotdifModel in the relaxation object used in calculations!"
    return []

def _obtain_R1R2NOErho(RObj, nSites, S21, S22, S23, consts1, consts2, consts3, taus1, taus2, taus3, vecXH, vecCX, vecHH, weights1=None, weights2=None, weights3=None, CSAvaluesArray=None):
    """
    The inputs vectors have dimensions (nSites, nSamples, 3) or just (nSites, 3)
    the datablock being returned has dimensions:
    - ( 4, nSites)    of there is no uncertainty calculations. 4 corresponding each to R1, R2, NOE, and rho.
    - ( 4, nSites, 2) if there is uncertainty calculations.
    """
    if CSAvaluesArray is None:
        CSAvaluesArray = np.repeat(CSAvaluesArray, nSites)
    if RObj.rotdifModel.name == 'direct_transform':
        datablock=np.zeros((4,nSites), dtype=np.float32)
        for i in range(nSites):
            J1 = J_direct_transform(RObj.omega, consts1[i], taus1[i])
            J2 = J_direct_transform(RObj.omega, consts2[i], taus2[i])
            J3 = J_direct_transform(RObj.omega, consts3[i], taus3[i])
            J4 = J_direct_transform(RObj.omega, consts3[i], taus3[i])
            R1, R2, NOE = RObj.get_relax_from_J( J1, J2, J3, J4, CSAvalue=CSAvaluesArray[i] )
            rho = RObj.get_rho_from_J( J1 )
            datablock[:,i]=[R1,R2,NOE,rho]
        return datablock
    elif RObj.rotdifModel.name == 'rigid_sphere':
        datablock=np.zeros((4,nSites), dtype=np.float32)
        for i in range(nSites):
            J1 = J_combine_isotropic_exp_decayN(RObj.omega, 1.0/(6.0*RObj.rotdifModel.D), S21[i], consts1[i], taus1[i])
            J2 = J_combine_isotropic_exp_decayN(RObj.omega, 1.0/(6.0*RObj.rotdifModel.D), S22[i], consts2[i], taus2[i])
            J3 = J_combine_isotropic_exp_decayN(RObj.omega, 1.0/(6.0*RObj.rotdifModel.D), S23[i], consts3[i], taus3[i])
            J4 = J_combine_isotropic_exp_decayN4(RObj.omega, 1.0/(6.0*RObj.rotdifModel.D))
            J5 = J_combine_isotropic_exp_decayN2(RObj.omega, 1.0/(6.0*RObj.rotdifModel.D))  
            J6 = J_combine_isotropic_exp_decayN3(RObj.omega, 1.0/(6.0*RObj.rotdifModel.D))  
            R1, R2, NOE = RObj.get_relax_from_J( J1, J2, J3, J4, J5,J6, CSAvalue=CSAvaluesArray[i] )
            rho = RObj.get_rho_from_J( J1 )
            datablock[:,i]=[R1,R2,NOE,rho]
        return datablock
    elif RObj.rotdifModel.name == 'rigid_symmtop':
        # Automatically use the vector-form of function.
        if len(vecXH.shape) > 2:
            # An ensemble of vectors for each site.
            datablock=np.zeros((4,nSites,2), dtype=np.float32)
            npts=vecXH.shape[1]
            #tmpR1  = np.zeros(npts) ; tmpR2 = np.zeros(npts) ; tmpNOE = np.zeros(npts)
            #tmprho = np.zeros(npts)
            for i in range(nSites):
                Jmat1 = J_combine_symmtop_exp_decayN(RObj.omega, vecXH[i], RObj.rotdifModel.D[0], RObj.rotdifModel.D[1], S21[i], consts1[i], taus1[i])
                Jmat2 = J_combine_symmtop_exp_decayN(RObj.omega, vecHH[i], RObj.rotdifModel.D[0], RObj.rotdifModel.D[1], S22[i], consts2[i], taus2[i])
                Jmat3 = J_combine_symmtop_exp_decayN(RObj.omega, vecHH[i], RObj.rotdifModel.D[0], RObj.rotdifModel.D[1], S23[i], consts3[i], taus3[i])
                Jmat4 = J_combine_isotropic_exp_decayN4(RObj.omega, 1.0/(6.0*(RObj.rotdifModel.D[0]+2*RObj.rotdifModel.D[1])/3))
                #Jmat4 = J_combine_symmtop_exp_decayN2(RObj.omega, vecHH[i], RObj.rotdifModel.D[0], RObj.rotdifModel.D[1])
                # = = = Calculate relaxation values from the entire sample of vectors before any averagins is to be done
                tmpR1, tmpR2, tmpNOE = RObj.get_relax_from_J_simd( Jmat1, Jmat2, Jmat3, Jmat4, CSAvalue=CSAvaluesArray[i] )
                tmprho = RObj.get_rho_from_J_simd( Jmat1 )
                #for j in range(npts):
                #    tmpR1[j], tmpR2[j], tmpNOE[j] = RObj.get_relax_from_J( Jmat[j] )
                #    tmprho[j] = RObj.get_rho_from_J( Jmat[j] )
                if weights1 is None:
                    R1 = np.mean(tmpR1)  ; R2 = np.mean(tmpR2)   ; NOE = np.mean(tmpNOE)
                    R1sig = np.std(tmpR1); R2sig = np.std(tmpR2) ; NOEsig = np.std(tmpNOE)
                    rho = np.mean(tmprho); rhosig = np.std(tmprho)
                    datablock[:,i]=[[R1,R1sig],[R2,R2sig],[NOE,NOEsig],[rho,rhosig]]
                else:
                    datablock[0,i]=weighted_average_stdev(tmpR1, weights1[i])
                    #datablock[1,i]=weighted_average_stdev(tmpR2, weights1[i])
                    datablock[1,i]=tmpR2    
                    datablock[2,i]=weighted_average_stdev(tmpNOE, weights1[i])
                    datablock[3,i]=weighted_average_stdev(tmprho, weights1[i])
            return datablock
        else:
            #Single XH vector for each site.
            datablock=np.zeros((4,nSites), dtype=np.float32)
            for i in range(nSites):
                Jmat1 = J_combine_symmtop_exp_decayN(RObj.omega, vecXH[i], RObj.rotdifModel.D[0], RObj.rotdifModel.D[1], S21[i], consts1[i], taus1[i])
                Jmat2 = J_combine_symmtop_exp_decayN(RObj.omega, vecCX[i], RObj.rotdifModel.D[0], RObj.rotdifModel.D[1], S22[i], consts2[i], taus2[i])
                Jmat3 = J_combine_symmtop_exp_decayN(RObj.omega, vecHH[i], RObj.rotdifModel.D[0], RObj.rotdifModel.D[1], S23[i], consts3[i], taus3[i])
                Jmat4 = J_combine_symmtop_exp_decayN(RObj.omega, vecHH[i], RObj.rotdifModel.D[0], RObj.rotdifModel.D[1], S23[i], consts3[i], taus3[i])
                if len(Jmat1.shape) == 1:
                    # A single vector was given.
                    R1, R2, NOE = RObj.get_relax_from_J( Jmat1, Jmat2, Jmat3, Jmat4, CSAvalue=CSAvaluesArray[i] )
                    rho = RObj.get_rho_from_J( Jmat1 )
                datablock[:,i]=[R1,R2,NOE,rho]
            return datablock
    # = = Should only happen with fully anisotropic models.
    print >> sys.stderr, "= = ERROR: Unknown rotdifModel in the relaxation object used in calculations!"
    return []

def weighted_average_stdev(values, weights, axis=-1):
    """
    Computes the weighted average and stdev when values are weighted.
    values  in N   dimensions
    weights in N-1 dimensions
    """
    #print type(values), values.shape
    #print type(weights), weights.shape
    avg   = np.average(values, axis=axis, weights=weights)
    stdev = np.sqrt( np.average( (values-avg)**2.0, axis=axis, weights=weights) )
    return avg, stdev

def convert_LambertCylindricalHist_to_vecs(hist, edges):
    print "= = = Reading histogram in Lambert-Cylindral projection, and returning distribution of non-zero vectors."
    # = = = Expect histograms as a list of 2D entries: (nResidues, phi, cosTheta)
    nResidues   = hist.shape[0]
    phis   = 0.5*(edges[0][:-1]+edges[0][1:])
    thetas = np.arccos( 0.5*(edges[1][:-1]+edges[1][1:]) )
    pt = np.moveaxis( np.array( np.meshgrid( phis, thetas, indexing='ij') ), 0, -1)
    binVecs = rtp_to_xyz( pt, vaxis=-1, bUnit=True )
    del pt, phis, thetas
    print "    ...shapes of first histogram and average-vector array:", hist[0].shape, binVecs.shape
    nPoints = hist[0].shape[0]*hist[0].shape[1]
    # = = = just in case this is a list of histograms..
    # = = = Keep all of the zero-weight entries vecause it keeps the broadcasting speed.
    #vecs    = np.zeros( (nResidues, nPoints, 3 ), dtpye=binVecs.dtype )
    #weights = np.zeros_like( vecs )
    return np.repeat( binVecs.reshape(nPoints,3)[np.newaxis,...], nResidues, axis=0), \
           np.reshape( hist, ( nResidues, nPoints) )
    # return vecs, weights

def _perturb_tuple(t,mod,axis):
    l = list(t)
    l[axis]+=mod
    return tuple(l)

def rtp_to_xyz(rtp, vaxis=-1 , bUnit=False ):
    """
    Converts a vector or a set of vectors from R/Phi/Theta to X/Y/Z.
    Noting that 0 ~ Theta ~ pi from positive Z.
    vaxis denotes the dimension in which the X/Y/Z value resides.
    This is the first (0) or last (-1) dimension of the array.
    If bUnit, expect only Phi/Theta instead of R/Phi/THeta
    """
    sh = rtp.shape
    dims = len(sh)
    if bUnit:
        if dims == 1:
            uv = np.zeros(3, dtype=rtp.dtype)
            uv[0]=np.cos(rtp[0])*np.sin(rtp[1])
            uv[1]=np.sin(rtp[0])*np.sin(rtp[1])
            uv[2]=np.cos(rtp[1])
        elif vaxis == -1:
            uv = np.zeros( _perturb_tuple(sh,mod=1,axis=-1) , dtype=rtp.dtype)
            uv[...,0]=np.cos(rtp[...,0])*np.sin(rtp[...,1])
            uv[...,1]=np.sin(rtp[...,0])*np.sin(rtp[...,1])
            uv[...,2]=np.cos(rtp[...,1])
        elif vaxis == 0:
            uv = np.zeros( _perturb_tuple(sh,mod=1,axis=0) , dtype=rtp.dtype)
            uv[0,...]=np.cos(rtp[0,...])*np.sin(rtp[1,...])
            uv[1,...]=np.sin(rtp[0,...])*np.sin(rtp[1,...])
            uv[2,...]=np.cos(rtp[1,...])
        else:
            print >> sys.stderr, "= = ERROR encountered in rtp-to-vec in general_maths.py, vaxis only accepts arguments of -1 or 0 for now."
    else:
        uv = np.zeros_like( rtp )
        if dims == 1:
            uv[0]=rtp[0]*np.cos(rtp[1])*np.sin(rtp[2])
            uv[1]=rtp[0]*np.sin(rtp[1])*np.sin(rtp[2])
            uv[2]=rtp[0]*np.cos(rtp[2])
        elif vaxis == -1:
            uv[...,0]=rtp[0]*np.cos(rtp[...,1])*np.sin(rtp[...,2])
            uv[...,1]=rtp[0]*np.sin(rtp[...,1])*np.sin(rtp[...,2])
            uv[...,2]=rtp[0]*np.cos(rtp[...,2])
        elif vaxis == 0:
            uv[0,...]=rtp[0]*np.cos(rtp[1,...])*np.sin(rtp[2,...])
            uv[1,...]=rtp[0]*np.sin(rtp[1,...])*np.sin(rtp[2,...])
            uv[2,...]=rtp[0]*np.cos(rtp[2,...])
        else:
            print >> sys.stderr, "= = ERROR encountered in rtp-to-vec in general_maths.py, vaxis only accepts arguments of -1 or 0 for now."

    return uv

# Read the formatted file headers in _fittedCt.dat. These are of the form:
# # Residue: 1
# # Chi-value: 1.15659e-05
# # Param XXX: ### +- ###
def read_fittedCt_file(filename):

    resid=[]
    param_name=[]
    param_val=[]
    tmp_name=[]
    tmp_val=[]
    for raw in open(filename):
        if raw == "" or raw[0]!="#":
            continue

        line=raw.split()
        if 'Vector' in line[1]:
            resid.append(int(line[-1]))
            if len(tmp_name)>0:
                param_name.append(tmp_name)
                tmp_name=[]
            if len(tmp_val)>0:
                param_val.append(tmp_val)
                tmp_val=[]
        elif 'Param' in line[1]:
            tmp_name.append(line[2][:-1])
            tmp_val.append(float(line[-3]))

    if len(tmp_name)>0:
        param_name.append(tmp_name)
        tmp_name=[]
    if len(tmp_val)>0:
        param_val.append(tmp_val)
        tmp_val=[]

    if len(resid) != len(param_name) != len(param_val):
        print >> sys.stderr, "= = ERROR in read_fittedCt_file: the header entries don't have the same number of residues as entries!"
        sys.exit(1)
    return resid, param_name, param_val

def read_fittedCt_file2(filename):

    resid=[]
    param_name=[]
    param_val=[]
    err_val = []
    tmp_name=[]
    tmp_val=[]
    tmp_err=[]
    for raw in open(filename):
        if raw == "" or raw[0]!="#":
            continue

        line=raw.split()
        if 'Vector' in line[1]:
            resid.append(int(line[-1]))
            if len(tmp_name)>0:
                param_name.append(tmp_name)
                tmp_name=[]
            if len(tmp_val)>0:
                param_val.append(tmp_val)
                err_val.append(tmp_err)
                tmp_val=[]
                tmp_err=[]
        elif 'Param' in line[1]:
            tmp_name.append(line[2][:-1])
            tmp_val.append(float(line[-3]))
            tmp_err.append(float(line[-1]))

    if len(tmp_name)>0:
        param_name.append(tmp_name)
        tmp_name=[]
    if len(tmp_val)>0:
        param_val.append(tmp_val)
        err_val.append(tmp_err)
        tmp_val=[]
        tmp_err=[]
    add2 = []
    for i in range(len(param_name)):
        add1 = []
        add2.append(add1)
        for j in range(len(param_val[0])):
            add = abs(param_val[i][j])#+err_val[i][j])
            add1.append(add)
    print add2
    if len(resid) != len(param_name) != len(param_val) != len(err_val):
        print >> sys.stderr, "= = ERROR in read_fittedCt_file: the header entries don't have the same number of residues as entries!"
        sys.exit(1)
    return resid, param_name, param_val, add2


def sanity_check_two_list(listA, listB, string, bVerbose=False):
    if not np.all( np.equal(listA, listB) ):
        print "= = ERROR: Sanity checked failed for %s!" % string
        if bVerbose:
            print listA
            print listB
        else:
            print "    ...first residues:", listA[0], listB[0]
            print "    ...set intersection (unordered):", set(listA).intersection(set(listB))
        sys.exit(1)
    return

def parse_parameters(names, values):
    if len(names) != len(values):
        print >> sys.stderr, "= = ERROR in parse_parameters! The lengths of names and values arrays are not the same!"
        sys.exit(1)
    S2=np.nan ; consts=[] ; taus=[] ; Sf=np.nan
    for i in range(len(names)):
        if names[i]=='S2_0':
            S2=values[i]
        elif 'C_' in names[i]:
            consts.append(values[i])
        elif 'tau_' in names[i]:
            taus.append(values[i])
        elif 'S2_fast'==names[i]:
            Sf=values[i]
        else:
            print >> sys.stderr, "= = ERROR: parameter name not recognised! %s %s" % (names[i], values[i])
    # Relic code for when order parameters are not explicitly written:
    if np.isnan(S2):
        S2 = 1.0-np.sum( consts )
        Sf = 0.0
    if np.isnan(Sf):
        Sf = 1-S2-np.sum(consts)
    return S2, consts, taus, Sf

def print_fitting_params_headers( names, values, units, bFit ):
    sumstr=""
    for i in range(len(names)):
        if bFit[i]:
            s1="Optimised"
        else:
            s1="Fixed"
        tmpstr="# %s %s: %g %s\n" % (s1, names[i], values[i], units[i])
        sumstr=sumstr+tmpstr
    return sumstr

def J_combine_isotropic_exp_decayN(om, tau_iso, S2, consts, taus):
    """
    This calculats the J value for combining an isotropic global tumbling with
    a fitted internal autocorrelation C(t), where
    C(t) = S2 + Sum{ consts[i] * exp ( -t/tau[i] }
    thus this allows fits to multiple time constants in C(t).
    """
    norm=2.0/5.0
    k = (1.0/tau_iso)+(1.0/np.array(taus))
    ndecay=len(consts) ; noms  =len(om)
    Jmat = np.zeros( (ndecay+1, noms ) )
    Jmat[0]= norm*S2*tau_iso/(1.+(om*tau_iso)**2.)
    for i in range(ndecay):
        Jmat[i+1] = norm*consts[i]*k[i] /(k[i]**2.+om**2.)
    return Jmat.sum(axis=0)


def J_combine_isotropic_exp_decayN2(om, tau_iso):
    norm=1#(2.0/5.0)
    noms  =len(om)
    Jmat = np.zeros( ( noms ) )
    for i in range(noms):
        Jmat[i] = (tau_iso/(1.+(om[i]*tau_iso)**2.))
    return Jmat


def J_combine_isotropic_exp_decayN3(om, tau_iso):
    norm=1.0#(2.0/5.0)
    noms  =len(om)
    Jmat = np.zeros( (noms ) )
    for i in range(noms):
        Jmat[i] = (tau_iso/(1.+(2.0*om[i]*tau_iso)**2.))
    return Jmat

def J_combine_isotropic_exp_decayN4(om, tau_iso):
    norm=1#(2.0/5.0)
    k = (1.0/tau_iso)
    noms  =len(om)
    Jmat = np.zeros( ( noms ) )
    for i in range(noms):
        Jmat[i]= (2*tau_iso)+(3*tau_iso/(2*(1.+(om[i]*tau_iso)**2.)))
    return Jmat




def J_combine_symmtop_exp_decayN(om, v, Dpar, Dperp, S2, consts, taus):
    """
    Calculats the J value for combining a symmetric-top anisotropic tumbling with
    a fitted internal autocorrelation C(t), where
    C(t) = S2 + Sum{ consts[i] * exp ( -t/tau[i] }
    thus this allows fits to multiple time constants in C(t).
    Note that v needs to be in the frame of the rotational diffusion tensor D, i.e. PAF.

    This function supports giving multiple vectors at once, of the form v.shape=(L,M,N,...,3)
    """
    #v can be given as an array, with the X/Y/Z cartesian axisin the last position.
    #"""
    D_J=D_coefficients_symmtop((Dpar, Dperp))
    # A_J is the same shape as v, so 3 in this case.
    A_J=A_coefficients_symmtop(v, bProlate=(Dpar>Dperp) )
    ndecay=len(consts) ; noms  =len(om)

    if len(v.shape) > 1:
        Jmat0 = _do_Jsum(om, S2*A_J, D_J)
        sh_J = Jmat0.shape ; sh_out=list(sh_J) ; sh_out.insert(0, ndecay+1)
        Jmat = np.zeros(sh_out)
        Jmat[0] = Jmat0
        for i in range(ndecay):
            Jmat[i+1] = _do_Jsum(om, consts[i]*A_J, D_J+1./taus[i])
        return Jmat.sum(axis=0)
    else:
        Jmat = np.zeros( (ndecay+1, noms ) )
        Jmat[0]= _do_Jsum(om, S2*A_J, D_J)
        for i in range(ndecay):
            Jmat[i+1] = _do_Jsum(om, consts[i]*A_J, D_J+1./taus[i])
        return Jmat.sum(axis=0)
    #return _do_Jsum( S2*A_J, D_J) + np.sum([ _do_Jsum(consts[i]*A_J, D_J+1./taus[i]) for i in range(len(consts)) ])

def J_combine_symmtop_exp_decayN2(om, v, Dpar, Dperp):
    """
    Calculats the J value for combining a symmetric-top anisotropic tumbling with
    a fitted internal autocorrelation C(t), where
    C(t) = S2 + Sum{ consts[i] * exp ( -t/tau[i] }
    thus this allows fits to multiple time constants in C(t).
    Note that v needs to be in the frame of the rotational diffusion tensor D, i.e. PAF.

    This function supports giving multiple vectors at once, of the form v.shape=(L,M,N,...,3)
    """
    #v can be given as an array, with the X/Y/Z cartesian axisin the last position.
    #"""
    D_J=D_coefficients_symmtop((Dpar, Dperp))
    # A_J is the same shape as v, so 3 in this case.
    A_J=A_coefficients_symmtop(v, bProlate=(Dpar>Dperp) )

    x = np.array([ np.sum([(2.0*A_J[i]/D_J[i])+(3.0*A_J[i]*D_J[i]/(2.0*(D_J[i]**2 + om[j]**2))) for i in range(len(D_J))]) for j in range(len(om)) ])
    print x
    return x

def A_coefficients_symmtop(v, bProlate=True):
    """
    Computes the 3 axisymmetric A-coefficients associated with orientation of the vector w.r.t. to the D_rot ellipsoid.
    v can be many dimensions, as lons as the X/Y/Z cartesian dimensions is the last. e.g. v.shape = (M,N,3)
    Note the current implementation is probably a bit slower on small sizes comapred to the trivial.
    This is designed to handle many vectors at once.
    Also note: The unique axis changes when Daniso > 1 and when Daniso < 1 so as to preserve Dx<Dy<Dz formalism.
    This is governed by bProlate, which changes the unique axis to x when tumbling is oblate.
    """
    v=_sanitise_v(v)
    if bProlate:
        # Use z-dim.
        z2=np.square(v.take(-1,axis=-1))
    else:
        # Use x-dim.
        z2=np.square(v.take(0,axis=-1))
    onemz2=1-z2
    A0 = np.multiply( 3.0, np.multiply(z2,onemz2))
    A1 = np.multiply(0.75, np.square(onemz2))
    A2 = np.multiply(0.25, np.square(np.multiply(3,z2)-1))
    return np.stack((A0,A1,A2),axis=-1)
    #z2=v[2]*v[2]
    #A=np.zeros(3)
    #A[0]= 3.00*z2*(1-z2)
    #A[1]= 0.75*(1-z2)**2
    #A[2]= 0.25*(3*z2-1)**2
    #return A

def A_coefficients_ellipsoid(v, DD, bDDisDelta=False):
    """
    Computes the 5 sull-anisotropic A-coefficients associated with orientation of the vector w.r.t. to the D_rot ellipsoid.
    DD is given either as the D-Rot elements or its 'delta' transformation for direct use.
    """
    #v can be given as an array with X/Y/Z cartesian dimensions being the last.
    #"""
    if bDDisDelta:
        delta=DD
    else:
        delta=Ddelta_ellipsoid(dd)
    #v=_sanitise_v(v)
    #v2=np.square(v)
    #v4=np.square(v2)
    #fact2=np.multiply(0.75,np.sum(v4))-0.25
    v2 = [ v[i]*v[i] for i in range(3) ]
    v4 = [ v2[i]*v2[i] for i in range(3) ]
    fact2 = 0.25*( 3.0*(v4[0]+v4[1]+v4[2])-1.0)
    fact3 = 1.0/12.0*(delta[0]*(3*v4[0]+6*v2[1]*v2[2]-1) + delta[1]*(3*v4[1]+6*v2[0]*v2[2]-1) + delta[2]*(3*v4[2]+6*v2[0]*v2[1]-1))
    A=np.zeros(5)
    A[0]= 3*v2[1]*v2[2]
    A[1]= 3*v2[0]*v2[2]
    A[2]= 3*v2[0]*v2[1]
    A[3]= fact2-fact3
    A[4]= fact2+fact3
    return A

def _sanitise_v(v):
    if type(v) != np.ndarray:
        v=np.array(v)
    sh=v.shape
    if sh[-1] != 3:
        print >> sys.stderr, "= = ERROR in computation of A and D coefficients (spectral_densities.py): input v does not have 3 as its final dimension!"
        sys.exit(2)
    return v

def _do_Jsum(om, A_J, D_J):
    """
    Lowest level operation. J = Sum_i components for each om. Return dimensions (N_om) for one vector A_j, and (N_Aj,N_om) otherwise.
    Equivalent to the old implementation:
    return np.array([ np.sum([A_J[i]*D_J[i]/(D_J[i]**2 + om[j]**2) for i in range(len(D_J))]) for j in range(len(om)) ])

    Output J has MxN, where M is the remaining dimensions of A, and N is the number of frequencies.
    J = A_ij*D_j/(D_j^2+om_k^2) = A_ij T_jk, in Einstein summation form.
    A can have many dimensions as long as th elast dimension is the matching one.
    """
    norm=2.0/5.0
    Dmat=norm*npufunc.Jomega.outer(D_J,om)
    #sys.exit()
    return np.einsum('...j,jk',A_J,Dmat)

def _do_Jsum2(om, A_J, D_J):
    Dmat=npufunc.Jomega.outer(D_J,om)
    #sys.exit()
    return np.einsum('...j,jk',A_J,Dmat)


def print_xy(fn, x, y, dy=[], header=""):
    fp=open(fn,'w')
    if header != "":
        print >>fp, header
    if dy==[]:
        for i in range(len(x)):
            print >> fp, x[i], y[i]
    else:
        for i in range(len(x)):
            print >> fp, x[i], y[i], dy[i]
    fp.close()

def print_xydy(fn, x, y, dy, header=""):
    print_xy(fn, x, y, dy, header)

out_pref='out'
bHaveDy=False
in_Ct_fn=out_pref+'_fittedCt_HH.dat'
in_Ct_fn2=out_pref+'_fittedCt_HCH.dat'
in_CT_fn3=out_pref+'_fittedCt_HHH.dat'
fittedCt_file = in_Ct_fn
fittedCt_file_C = in_Ct_fn2
fittedCt_file_H = in_CT_fn3
opt=None
expfn=None
nuclei='methyl'
B0=-1 # T
freq=field_strength*10**6 # Field strength in MHz
D="diffusion_values" 
tau=np.nan
aniso=1.0
qrot_str=''
vecfn=None
distfn=out_pref+'_vecHistogramHH.npz'
distfn_C=out_pref+'_vecHistogram_C.npz'
distfn_H=out_pref+'_vecHistogram_HHH.npz'
bRigid=False
csa=None
shiftres=0
Hz=freq
bJomega=False #True
zeta=1.0 #0.890023
if zeta != 1.0:
    print " = = Applying scaling to add zero-point QM vibrations (zeta) of %g" % zeta
if not opt is None:
    if expfn is None:
        print >> sys.stderr, "= = = ERROR: Cannot conduct optimisation without a target experimental scattering file! (Missing --expfn )"
        sys.exit(1)
    bOptPars = True
    optMode = opt
else:
    bOptPars = False

nuclei_pair = nuclei
timeUnit = 'ps'
if Hz != -1:
    B0 = 2.0*np.pi*Hz / 267.513e6
elif B0 != -1:
    B0 = B0
else:
    print >> sys.stderr, "= = = ERROR: Must give either the background magnetic field or the frequency! E.g., -B0 14.0956"
    sys.exit(1)

relax_obj = relaxationModel(nuclei_pair, B0)
relax_obj.set_time_unit(timeUnit)
relax_obj.set_freq_relaxation()
print "= = = Setting up magnetic field:", B0, "T"
print "= = = Angular frequencies in ps^-1 based on given parameters:"
relax_obj.print_freq_order()
print relax_obj.omega
print "= = = Gamma values: (X) %g , (H) %g" % (relax_obj.gX.gamma, relax_obj.gH.gamma)

if D == '-1':
    if np.isnan(tau):
        diff_type = 'direct'
        Diso = 0.0
    else:
        tau_iso=tau
        Diso  = 1.0/(6*tau)
        if aniso != 1.0:
            aniso = aniso
            diff_type = 'symmtop'
        else:
            diff_type = 'spherical'
else:
    tmp   = [ float(x) for x in D.split() ]
    Diso  = tmp[0]
    if len(tmp)==1:
        diff_type = 'spherical'
    elif len(tmp)==2:
        aniso = tmp[1]
        diff_type = 'symmtop'
    else:
        aniso = tmp[1]
        rhomb = tmp[2]
        diff_type = 'anisotropic'
    tau_iso = 1.0/(6*Diso)

vecXH=None
vecXHweights=None
vecCX=None
vecCXweights=None
vecHH=None
vecHHweights=None
if diff_type=='direct':
    print "= = = No global rotational diffusion selected. Calculating the direct transform."
    relax_obj.set_rotdif_model('direct_transform')
elif diff_type=='spherical':
    print "= = = Using a spherical rotational diffusion model."
    relax_obj.set_rotdif_model('rigid_sphere_D', Diso)
elif diff_type=='symmtop':
    Dperp = 3.*Diso/(2+aniso)
    Dpar  = aniso*Dperp
    print "= = = Calculated anisotropy to be: ", aniso
    print "= = = With Dpar, Dperp: %g, %g %s^-1" % ( Dpar, Dperp, timeUnit)
    # This part is ignored for now..
    relax_obj.set_rotdif_model('rigid_symmtop_D', Dpar, Dperp)
    # Read quaternion
    if qrot_str != "":
        bQuatRot = True
        q_rot = np.array([ float(v) for v in rot_str.split() ])
        if not qops.qisunit(q_rot):
            q_rot = q_rot/np.linalg.norm(q_rot)
    else:
        bQuatRot = False
    # Read the source of vectors.
    bHaveVec = False
    bHaveVDist = False
    if not vecfn is None:
        print "= = = Using average vectors. Reading X-H vectors from %s ..." % vecfn
        resNH, vecXH = load_xys(vecfn, dtype=float32)
        bHaveVec = True
    elif not distfn is None:
        print "= = = Using vector distribution in spherical coordinates. Reading X-H vector distribution from %s ..." % distfn
        resNH, vecXH, vecXHweights = read_vector_distribution_from_file( distfn )
        resNH = [ int(x)+shiftres for x in resNH ]
        print "= = = Using vector distribution in spherical coordinates. Reading X-H vector distribution from %s ..." % distfn_C
        resCX, vecCX, vecCXweights = read_vector_distribution_from_file( distfn_C )
        resCX = [ int(x)+shiftres for x in resCX ]
        print "= = = Using vector distribution in spherical coordinates. Reading H-H vector distribution from %s ..." % distfn_H
        resHH, vecHH, vecHHweights = read_vector_distribution_from_file( distfn_H )
        resHH = [ int(x)+shiftres for x in resHH ]
        bHaveVDist = True
        bHaveDy = True ;# We have an ensemble of vectors now.
    elif not bRigid:
        print >> sys.stderr, "= = = ERROR: non-spherical diffusion models require a vector source!" \
                                    "Please supply the average vectors or a trajectory and reference!"
        sys.exit(1)
    if bHaveVec or bHaveVDist:
        print "= = = Note: the shape of the X-H vector distribution is:", vecXH.shape
        print "= = = Note: the shape of the C-X vector distribution is:", vecCX.shape
        print "= = = Note: the shape of the H-H vector distribution is:", vecHH.shape
        if bQuatRot:
            print "    ....rotating input vectors into PAF frame using q_rot."
            vecXH = rotate_vector_simd(vecXH, q_rot)
            vecCX = rotate_vector_simd(vecCX, q_rot)
            vecHH = rotate_vector_simd(vecHH, q_rot)
            print "    ....X-H vector input processing completed."

if bRigid:
    if diff_type == 'direct':
        print >> sys.stderr, "= = = ERROR: Rigid-sphere argument cannot be applied without an input for the global rotational diffusion!"
        sys.exit(1)
    if diff_type == 'spherical' or diff_type == 'anisotropic':
        num_vecs=1
        S2_list=[zeta]
        consts_list=[[0.]]
        taus_list=[[99999.]]
        vecXH=[]
        vecCX=[]
        vecHH=[]
    else:
        num_vecs=3
        S2_list=[zeta,zeta,zeta]
        consts_list=[[0.],[0.],[0.]]
        taus_list=[[99999.],[99999.],[99999.]]
        vecXH=np.identity(3)
        vecCX=np.identity(3)
        vecHH=np.identity(3)
        #datablock = _obtain_R1R2NOErho(relax_obj, num_vecs, S2_list, consts_list, taus_list, vecXH)
        datablock = _obtain_R1R2NOErho(relax_obj, num_vecs, S2_list, consts_list, taus_list, vecCX)
        if diff_type == 'spherical':
            print "...Isotropic baseline values:"
        else:
            print "...Anistropic axial baseline values (x/y/z):"
        print "Rz:",  str(datablock[0]).strip('[]')
        print "Ry:",  str(datablock[1]).strip('[]')
        print "R3z2-2:", str(datablock[2]).strip('[]')
    sys.exit()

sim_resid, param_names, param_vals = read_fittedCt_file(fittedCt_file)
sim_resid_C, param_names_C, param_vals_C = read_fittedCt_file(fittedCt_file_C)
sim_resid_H, param_names_H, param_vals_H = read_fittedCt_file(fittedCt_file_H)
num_vecs = len(sim_resid)
if diff_type == 'symmtop' or diff_type == 'anisotropic':
    sanity_check_two_list(sim_resid, resNH, "resid from fitted_Ct -versus- vectors as defined in anisotropy")
    sanity_check_two_list(sim_resid_C, resCX, "resid from fitted_Ct_C -versus- vectors as defined in anisotropy")
    sanity_check_two_list(sim_resid_H, resHH, "resid from fitted_Ct_H -versus- vectors as defined in anisotropy")

CSAvaluesArray = None
if csa is None:
    print "= = = Using default CSA value: %g" % relax_obj.gX.csa
    CSAvaluesArray = np.repeat( relax_obj.gX.csa, num_vecs )

S2_list=[]
taus_list=[]
consts_list=[]
S2_list_C=[]
taus_list_C=[]
consts_list_C=[]
S2_list_H=[]
taus_list_H=[]
consts_list_H=[]
for i in range(num_vecs):
    # Parse input parameters
    S2, consts, taus, Sf = parse_parameters( param_names[i], param_vals[i] )
    S2_C, consts_C, taus_C, Sf_C = parse_parameters( param_names_C[i], param_vals_C[i] )
    #S2_C, consts_C, taus_C, Sf_C = parse_parameters( param_names_C[i], err_val_C[i] )
    S2_H, consts_H, taus_H, Sf_H = parse_parameters( param_names_H[i], param_vals_H[i] )
    # = = = This section applies zero-point corrections to S2, 0.89 = = =
    S2 *= zeta
    S2_C *= zeta
    S2_H *= zeta
    consts = [ k*zeta for k in consts ]
    consts_C = [ k*zeta for k in consts_C ]
    consts_H = [ k*zeta for k in consts_H ]
    S2_list.append(S2)
    taus_list.append(taus)
    consts_list.append(consts)
    S2_list_C.append(S2_C)
    taus_list_C.append(taus_C)
    consts_list_C.append(consts_C)
    S2_list_H.append(S2_H)
    taus_list_H.append(taus_H)
    consts_list_H.append(consts_H)
# = = = Based on simulation fits, obtain R1, R2, NOE for this X-H vector
param_names=("Diso", "zeta", "CSA", "chi")
param_scaling=( 1.0, zeta, 1.0e6, 1.0 )
param_units=(relax_obj.timeUnit, "a.u.", "ppm", "a.u." )
optHeader=''
if not bOptPars:
    if bJomega:
        #datablock = _obtain_Jomega(relax_obj, num_vecs, S2_list, consts_list, taus_list, vecXH, weights=vecXHweights)
        datablock = _obtain_Jomega(relax_obj, num_vecs, S2_list_C, consts_list_C, taus_list_C, vecXH, weights=vecXHweights)
    else:
        #datablock = _obtain_R1R2NOErho(relax_obj, num_vecs, S2_list, consts_list, taus_list, vecXH, weights=vecXHweights, CSAvaluesArray = CSAvaluesArray )
        datablock = _obtain_R1R2NOErho(relax_obj, num_vecs, S2_list, S2_list_C, S2_list_H, consts_list, consts_list_C, consts_list_H, taus_list, taus_list_C, taus_list_H, vecXH, vecCX, vecHH, weights1=vecXHweights,weights2=vecCXweights, weights3=vecHHweights, CSAvaluesArray = CSAvaluesArray )
    optHeader=print_fitting_params_headers(names=param_names, values=np.multiply(param_scaling, (Diso, 1.0, relax_obj.gX.csa, 0.0)), units=param_units, bFit=(False, False, False, False) )
print " = = Completed Relaxation calculations."

# = = = Print
if bJomega:
    fp = open(out_pref+'_Jw.dat', 'w')
    if optHeader != '':
        print >> fp, '%s' % optHeader
    if bHaveDy:
        print >> fp, '@type xydy'
    s=0
    num_omega=relax_obj.num_omega
    xdat = np.fabs(relax_obj.omega)
    #xdat = np.arange(0, 0.002, 0.00001)
    for i in range(num_vecs):
        print >> fp, '@s%d legend "Resid: %d"' % (i, sim_resid[i])
        for j in np.argsort(xdat):
            if bHaveDy:
                print >> fp, '%g %g %g' % (xdat[j], datablock[j,i,0], datablock[j,i,1])
            else:
                print >> fp, '%g %g' % (xdat[j], datablock[j,i])
        print >> fp, '&'
        s+=1
    fp.close()
else:
    if not bHaveDy:
        print_xy(out_pref+'_CCR.dat',  sim_resid, datablock[0,:], header=optHeader)
        print_xy(out_pref+'_CCR_rigid.dat',  sim_resid, datablock[1,:], header=optHeader)
        print_xy(out_pref+'_S2.dat', sim_resid, datablock[2,:], header=optHeader)
        #print_xy(out_pref+'_rho.dat', sim_resid, datablock[3,:])
    else:
        print_xydy(out_pref+'_CCR.dat',  sim_resid, datablock[0,:,0], datablock[0,:,1], header=optHeader)
        print_xydy(out_pref+'_CCR_rigid.dat',  sim_resid, datablock[1,:,0], datablock[1,:,1], header=optHeader)
        print_xydy(out_pref+'_S2.dat', sim_resid, datablock[2,:,0], datablock[2,:,1], header=optHeader)
        #print_xydy(out_pref+'_rho.dat', sim_resid, datablock[3,:,0], datablock[3,:,1])
