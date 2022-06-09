#!/home/hoffmann/anaconda2/bin/python2.7

from math import fmod
import numpy as np
from scipy.optimize import curve_fit
from scipy import fftpack
import sys


def sort_parameters(num_pars, params):
    if np.fmod( num_pars, 2 ) == 1:
        S2     = params[0]
        consts = [ params[k] for k in range(1,num_pars,2) ]
        taus   = [ params[k] for k in range(2,num_pars,2) ]
        Sf     = 1-params[0]-np.sum(consts)
    elif num_pars==2:
        S2     = -0.125-params[0]
        consts = params[0]
        taus   = params[1]
        Sf     = 0.0
    elif num_pars==4:
         S2     = -0.125-params[0]-params[2]
         Sf     = -0.125-params[0]
         consts = [ params[k] for k in range(0,num_pars,2) ]
         taus   = [ params[k] for k in range(1,num_pars,2) ]
    return S2, consts, taus, Sf


def calc_chi(y1, y2, dy=[]):
    if dy != []:
        return np.sum( (y1-y2)**2.0/dy )/len(y1)
    else:
        return np.sum( (y1-y2)**2.0 )/len(y1)

def aic(nparm, y1, y2, dy):
    rss= np.sum((y1-y2)**2.0)/len(y1)
    return 2*nparm-np.log(rss)

def func_exp_decay1(t, tau_a):
    return np.exp(-t/tau_a)
def func_exp_decay2(t, A, tau_a):
    return (-0.125-A) + A*np.exp(-t/tau_a)
def func_exp_decay3(t, S2, A, tau_a):
    return S2 + A*np.exp(-t/tau_a)
def func_exp_decay4(t, A, tau_a, B, tau_b):
    return (-0.125-A-B) + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b)
def func_exp_decay5(t, S2, A, tau_a, B, tau_b ):
    return S2 + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b)
def func_exp_decay6(t, A, tau_a, B, tau_b, G, tau_g ):
    return (-0.125-A-B-G) + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g)
def func_exp_decay7(t, S2, A, tau_a, B, tau_b, G, tau_g ):
    return S2 + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g)
def func_exp_decay8(t, A, tau_a, B, tau_b, G, tau_g, D, tau_d):
    return (-0.125-A-B-G-D) + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + D*np.exp(-t/tau_d)
def func_exp_decay9(t, S2, A, tau_a, B, tau_b, G, tau_g, D, tau_d):
    return S2 + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + D*np.exp(-t/tau_d)
def func_exp_decay10(t, A, tau_a, B, tau_b, G, tau_g, D, tau_d, E, tau_e):
    return (-0.125-A-B-G-D-E) + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + D*np.exp(-t/tau_d) + E*np.exp(-t/tau_e)
def func_exp_decay11(t, S2, A, tau_a, B, tau_b, G, tau_g, D, tau_d, E, tau_e):
    return S2 + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + D*np.exp(-t/tau_d) + E*np.exp(-t/tau_e)

def _bound_check(func, params):
    """
    Hack for now.
    """
    if len(params) == 1:
        return False
    elif len(params) %2 == 0 :
        s = sum(params[0::2])
        return (s>1)
    else:
        s = params[0]+sum(params[1::2])
        return (s>1)

def _return_parameter_names(num_pars):
    if num_pars==1:
        return ['tau_a']
    elif num_pars==2:
         return ['C_a', 'tau_a']
    elif num_pars==3:
         return ['S2_0', 'C_a', 'tau_a']
    elif num_pars==4:
         return ['C_a', 'tau_a', 'C_b', 'tau_b']
    elif num_pars==5:
         return ['S2_0', 'C_a', 'tau_a', 'C_b', 'tau_b']
    elif num_pars==6:
         return ['C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g']
    elif num_pars==7:
         return ['S2_0', 'C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g']
    elif num_pars==8:
         return ['C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g', 'C_d', 'tau_d']
    elif num_pars==9:
         return ['S2_0', 'C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g', 'C_d', 'tau_d']
    elif num_pars==10:
         return ['C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g', 'C_d', 'tau_d', 'C_e', 'tau_e']
    elif num_pars==11:
         return ['S2_0', 'C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g', 'C_d', 'tau_d', 'C_e', 'tau_e']
    return []

def do_Expstyle_fit(num_pars, x, y, dy=[]):
    if num_pars==1:
        func=func_exp_decay1
        guess=(x[-1]/2.0)
        bound=(0.,np.inf)
    elif num_pars==2:
        func=func_exp_decay2
        guess=(-0.3, 15)
        bound=([-1.,0],[1.,5000])
    elif num_pars==3:
        func=func_exp_decay3
        guess=(0.33, -0.3, 15)
        bound=([0.,-1.,0.],[1.,0.,5000])
    elif num_pars==4:
        func=func_exp_decay4
        guess=(-0.3, 15, 0., 15)
        bound=([-1.0, 2.5, 0., 2.5],[1.,5000,1.,15000])
    elif num_pars==5:
        func=func_exp_decay5
        guess=(0.5, -0.3, 15, 0., 1500)
        bound=([0.018, -0.3,0.,-1.,0.],[1.,0., 5000 ,1.,15000])
    elif num_pars==6:
        func=func_exp_decay6
        guess=(-0.3, 15, 0., 1500, 0., 1500)
        bound=([-1., 0., -1., 0., -1., 0.],[0.,5000,1.,15000,1.,15000])
    elif num_pars==7:
        func=func_exp_decay7
        guess=(0.33, -0.3, 15, 0., 1500 , 0, 1500)
        bound=([0.,-1.,0.,-1.,0,-1.,0.],[1.,0,5000,1.,15000,1.,15000])
    elif num_pars==8:
        func=func_exp_decay8
        guess=(-0.3, 15, 0., 1500, 0., 1500, 0., 1500)
        bound=([-1., 0., -1., 0., -1., 0., -1., 0.],[0.,5000,1.,15000,1.,15000,1.,15000])
    elif num_pars==9:
        func=func_exp_decay9
        guess=(0.33, -0.3, 15, 0., 1500, 0., 1500, 0., 1500 )
        bound=([0.,-1.,0.,-1.,0.,-1.,0.,-1.,0.],[1.,0,5000,1.,15000,1.,15000,1.,15000])
    if dy != []:
        popt, popv = curve_fit(func, x, y, p0=guess, sigma=dy, bounds=bound)
    else:
        popt, popv = curve_fit(func, x, y, p0=guess, bounds=bound)
    ymodel=[ float(func(x[i], *popt)) for i in range(len(x)) ]
    #print ymodel
    bExceed=False#_bound_check(func, popt)
    if bExceed:
        print >> sys.stderr, "= = = WARNING, curve fitting in do_LSstyle_fit returns a sum>1.//"
        return 9999.99, popt, np.sqrt(np.diag(popv)), ymodel
    else:
        return calc_chi(y, ymodel, dy), popt, np.sqrt(np.diag(popv)), ymodel, aic(num_pars, y, ymodel,dy)

def run_Expstyle_fits(x, y, dy, npars):
    names = _return_parameter_names(npars)
    try:
        chi, params, errors, ymodel, aic = do_Expstyle_fit(npars, x, y, dy)
    except:
        print " ...fit returns an error! Continuing."
    for i in range(npars):
        print "Parameter %i %s: %g +- %g " % (i, names[i], params[i], errors[i])
    return chi, names, params, errors, ymodel


#def findbest_Expstyle_fits(x, y, dy=[], bPrint=True, par_list=[3,5,7,9], threshold=-0.0018):
def findbest_Expstyle_fits(x, y, dy=[], bPrint=True, par_list=[2,4,6,8], threshold=-0.00138):
    chi_min=np.inf
    # Search forwards
    for npars in par_list:
        names = _return_parameter_names(npars)
        try:
            chi, params, errors, ymodel, aic = do_Expstyle_fit(npars, x, y, dy)
            print 'number of parameters= ', npars, 'AIC= ', aic
        except:
            print " ...fit returns an error! Continuing."
            break
        bBadFit=False
        if npars == 2:
           print 'Chi(2)=', chi
        if npars == 4:
           print 'Chi(4)=', chi
        for i in range(npars):
            if errors[i]/ abs(params[i]) > 0.15:
                print  " --- fit shows overfitting with %d parameters." % npars
                print  "  --- Occurred with parameter %s: %g +- %g " % (names[i], params[i], errors[i])
                bBadFit=True
                break
        if (not bBadFit) and chi-chi_min < threshold:
            chi_min=chi ; par_min=params ; err_min=errors ; npar_min=npars ; ymod_min=ymodel
        else:
            break
    if bPrint:
        names = _return_parameter_names(npar_min)
        print "= = Found %d parameters to be the minimum necessary to describe curve: chi(%d) = %g vs. chi(%d) = %g)" % (npar_min, npar_min, chi_min,  npars, chi)
        S2_all=1.0
        for i in range(npar_min):
            print "Parameter %d %s: %g +- %g " % (i, names[i], par_min[i], err_min[i])
            if 'S2' in names[i]:
                S2_all=S2_all*par_min[i]
        #print "Overall S2: %g" % S2_all
        # Special case for 2:
        if npar_min == 2:
            S2_all= 1.0 - par_min[0]
    return chi_min, names, par_min, err_min, ymod_min


def load_sxydylist(fn, key="legend"):
    leglist=[]
    xlist=[]
    ylist=[]
    dylist=[]
    x=[] ; y=[] ; dy=[]
    for l in open(fn):
        lines = l.split()
        if l=="" or l=="\n":
            continue
        if l[0]=="#" or l[0]=="@":
            if key in l:
                leglist.append(lines[-1].strip('"'))
            continue
        if l[0]=="&":
            xlist.append(x) ; ylist.append(y)
            if len(dy)>0:
                dylist.append(dy)
            x=[] ; y=[] ; dy=[]
            continue
        x.append(float(lines[0]))
        y.append(float(lines[1]))
        if len(lines) > 2:
            dy.append(float(lines[2]))

    if x != []:
        xlist.append(x) ; ylist.append(y) ; dylist.append(dy)

    if dylist != []:
        return leglist, np.array(xlist), np.array(ylist), np.array(dylist)
    else:
        return leglist, np.array(xlist), np.array(ylist), []

out_pref='out'
out_fn=out_pref+'_fittedCt_HHH.dat'
in_file_list = [out_pref+'_Ctint_HHH.dat']
bNoFast=True #False
nc=-1


num_files = len(in_file_list)
if (num_files == 1):
    legs, dt, Ct, Cterr = load_sxydylist(in_file_list[0], 'legend')
    legs = [ float(x) for x in legs ]
num_comp=nc
bUseSFast=(not bNoFast)
sim_resid = legs
num_vecs = len(dt)
fp=open(out_fn, 'w')
for i in range(num_vecs):
    print "...Running C(t)-fit for vector %i:" % sim_resid[i]
    if len(Cterr)>0:
        dy_loc=Cterr[i]
    if num_comp == -1:
        # Automatically find the best number of parameters
        if bUseSFast:
            chi, names, pars, errs, ymodel = findbest_Expstyle_fits(dt[i], Ct[i], dy_loc, par_list=[2,3,5,7,9], threshold=-0.0018)
        else:
            chi, names, pars, errs, ymodel = findbest_Expstyle_fits(dt[i], Ct[i], dy_loc, par_list=[2,4], threshold=-0.0018)
        num_pars=len(names)
    else:
        # Use a specified number of parameters
        if bUseSFast:
            num_pars=2*nc+1
        else:
            num_pars=2*nc
        chi, names, pars, errs, ymodel = run_Expstyle_fits(dt[i], Ct[i], dy_loc, num_pars)
    S2, consts, taus, Sf = sort_parameters(num_pars, pars)
    #print consts
    #consts = consts1
    # Print header into the Ct model file
    print >> fp, '# Vector: %i ' % sim_resid[i]
    print >> fp, '# Chi-value: %g ' % chi
    if fmod( num_pars, 2 ) == 1:
        print >> fp, '# Param %s: %g +- %g' % ('S2_fast', Sf, 0.0)
    elif num_pars==2:
         print >> fp, '# Param %s: %g +- %g' % ('S2_0', S2, 0.0)
    elif num_pars==4:
        print >> fp, '# Param %s: %g +- %g' % ('S2_0', -0.125-pars[0]-pars[2], 0.0)
        #print >> fp, '# Param %s: %g +- %g' % ('Sf', -0.316-pars[0], 0.0)
    for j in range(num_pars):
        print >> fp, "# Param %s: %g +- %g" % (names[j], pars[j], errs[j])
    #Print the fitted Ct model into file
    print >> fp, "@s%d legend \"Res %d\"" % (i*2, sim_resid[i])
    for j in range(len(ymodel)):
        print >> fp, dt[i][j], ymodel[j]
    print >> fp, '&'
    for j in range(len(ymodel)):
        print >> fp, dt[i][j], Ct[i][j]
    print >> fp, '&'
print " = = Completed C(t)-fits."
