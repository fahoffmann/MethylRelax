#!/usr/bin/python2.7 

from math import acos, log, exp, ceil
from scipy.optimize import fmin_powell
import numpy as np
import transforms3d.quaternions as qops
import sys

def aniso(D):
    return 2*D[2]/(D[1]+D[0])

def rhomb(D):
    return 3*(D[1]-D[0])/(2*D[2]-D[1]-D[0])

def calculate_aniso_nosort(D):
    iso   = np.mean(D)
    aniL  = aniso(D)
    rhomL = rhomb(D)
    aniS  = aniso(D[::-1])
    rhomS = rhomb(D[::-1])
    return (iso, aniL, rhomL, aniS, rhomS)

def calculate_anisotropies( D, chunkD=[]):
    if len(chunkD)==0:
        D_sorted = np.sort(D)
        return calculate_aniso_nosort(D_sorted)
    else:
        # Do some set magic
        block=np.concatenate( (np.reshape(D,(1,3)),chunkD) ).T
        block=block[block[:,0].argsort()]
        # calculate values from sub-chunks
        chunkdat=np.array( [ calculate_aniso_nosort( block[:,i]) for i in range(len(chunkD)+1)] )
        out=[ ( chunkdat[0,i],np.std(chunkdat[1:,i]) ) for i in range(5) ]
        # Output format [ (iso, isoErr), (aniL, aniLErr), (rhomL,ehomLErr), ... ]
    return out

def format_header(style_str, tau, taus=[]):
    l=[]
    if style_str=='iso':
        l.append('# model fit, tau = %e [ps]' % (tau) )
        Dval = 0.5e12/tau
        l.append("# Converted D_iso = %e [s^-1]" % (Dval))
        l.append("# t cos(th) P2[cos(th)] cos(th/2) th")
    elif style_str=='iso_err':
        # Print overall stats
        bound=get_flex_bounds(tau, taus)
        l.append('# model fit, tau = %e +- %e %e [ps]'  % (bound[0], bound[1], bound[2]))
        Dval = 0.5e12/tau ; Dvals= [0.5e12/taus[i] for i in range(len(taus)) ]
        bound=get_flex_bounds(Dval, Dvals)
        l.append('# Converted D_iso = %e +- %e %e [s^-1]' % (bound[0], bound[1], bound[2]))
        # Print chunk data:
        for i in range(len(taus)):
            l.append('# Chunk_%d D_iso = %e [s^-1]' % (i, Dvals[i]))
        l.append("# t cos(th) P2[cos(th)] cos(th/2) th")
    elif style_str=='aniso':
        Dval = 0.5e12/tau
        for i in range(3):
            l.append("# model fit, e_%i tau = %e [ps]" % (i, tau[i]))
            l.append("# Converted D_%i = %e [s^-1]" % (i, Dval[i]))
        anis=calculate_anisotropies( Dval )
        l.append("# Converted Diso = %e [s^-1]" % (anis[0]))
        l.append("# Converted Dani_L = %f" % (anis[1]))
        l.append("# Converted Drho_L = %f" % (anis[2]))
        l.append("# Converted Dani_S = %f" % (anis[3]))
        l.append("# Converted Drho_S = %f" % (anis[4]))
        l.append("# t <1-2x^2> <1-2y^2> <1-2z^2>")
    elif style_str=='aniso_err':
        # Print overall stats.
        Dval=0.5e12/tau
        Dvals=0.5e12/taus
        for i in range(3):
            bound=get_flex_bounds(tau[i], taus[:,i])
            l.append('# model fit, e_%i tau = %e +- %e %e [ps]'  % (i, bound[0], bound[1], bound[2]))
            bound=get_flex_bounds(Dval[i], Dvals[:,i])
            l.append('# Converted D_%i = %e +- %e %e [s^-1]' % (i, bound[0], bound[1], bound[2]))
        anis=calculate_anisotropies(Dval, Dvals)
        l.append("# Converted Diso = %e +- %e [s^-1]" % anis[0])
        l.append("# Converted Dani_L = %f +- %f" % anis[1])
        l.append("# Converted Drho_L = %f +- %f" % anis[2])
        l.append("# Converted Dani_S = %f +- %f" % anis[3])
        l.append("# Converted Drho_S = %f +- %f" % anis[4])
        # Print chunk data
        for j in range(len(taus)):
            for i in range(3):
                l.append('# Chunk_%d D_%d = %e [s^-1]' % (j, i, Dvals[j,i]))
        l.append("# t <1-2x^2> <1-2y^2> <1-2z^2>")
    return l

def print_model_fits_gen(fname, ydims, str_header, xlist, ylist):
    """
    Generic XVG  printout. Uses the dimension of y1list to determine
    how to print them. THe number of entries in each plot must be in the last dimension.
    - 1D ylist is a single plot
    - 2D ylist will have multiple plots on the samge graph
    - 3D ylist will have multiple graphs (first axis), on which multiple plots will exist(second axis).
    """
    g=0; s=0
    ndat=len(xlist)
    fp = open(fname, 'w')
    for line in str_header:
        print >> fp, "%s" % line
    if ydims==1:
        for i in range(ndat):
            print >> fp, "%g %g" % (xlist[i], ylist[i])
    elif ydims==2:
        dim1=len(ylist)
        for j in range(dim1):
            print >> fp, "@target g%d.s%d" % (g,s)
            for i in range(ndat):
                print >> fp, "%g %g" % (xlist[i], ylist[j][i])
            print >> fp, "&"
            s+=1
    elif ydims==3:
        dim1=len(ylist)
        print "dim1: ", dim1
        for k in range(dim1):
            print >> fp, "@g%d on" % g
            g+=1
        g=0
        for k in range(dim1):
            dim2=len(ylist[k])
            print "dim2: ", dim2
            for j in range(dim2):
                print >> fp, "@target g%d.s%d" % (g,s)
                for i in range(ndat):
                    print >> fp, "%g %g" % (xlist[i], ylist[k][j][i])
                print >> fp, "&"
                s+=1
            g+=1; s=0
    else:
        print "= = = Critical ERROR: invalid dimension specifier in print_model_fits_gen!"
        sys.exit(1)
    fp.close()

def print_xylist(fname, xlist, ylist):
    fp = open(fname, 'w')
    for i in range(len(xlist)):
        print >> fp, "%10f " % float(xlist[i]),
        for j in range(len(ylist[i])):
            print >> fp, "%10f " % float(ylist[i][j]),
            #print >> fp, fmtstr[:-1] % ylist[i]
        print >> fp , ""
    print >> fp, "&"
    fp.close()

def print_axes_as_xyz(fname, mat):
    fp = open(fname, 'w')
    for i in range(len(mat)):
        print >> fp, "3"
        print >> fp, "AXES"
        print >> fp, "X %g %g %g" % (out_moilist[i][0,0], out_moilist[i][0,1], out_moilist[i][0,2])
        print >> fp, "Y %g %g %g" % (out_moilist[i][1,0], out_moilist[i][1,1], out_moilist[i][1,2])
        print >> fp, "Z %g %g %g" % (out_moilist[i][2,0], out_moilist[i][2,1], out_moilist[i][2,2])
    fp.close()

def format_header_quat(q):
    return '# Quaternion orientation frame: %f %f %f %f' % (q[0], q[1], q[2], q[3])

def anisotropic_decay_noc(x, a):
    return 0.5*np.exp(-x/a)+0.5

def isotropic_decay(x, a):
    return 1.5*np.exp(-x/a)-0.5

def build_model(func, args, xvals):
    out=[]
    for i in range(len(xvals)):
        out.append( func(xvals[i], args) )
    return out

def powell_expdecay(pos, *args):
    """
    Fits y = C0 exp (x/A) + C1
    """
    x=args[0]
    y=args[1]
    C0=args[2]
    C1=args[3]
    A=pos
    chi2=0.0
    nval=len(x)
    for i in range(nval):
        ymodel=C0*exp(-x[i]/A)+C1
        chi2+=(ymodel-y[i])**2
    return chi2/nval

def obtain_exponential_guess(x, y, C1):
    return (x[0]-x[1])/log((y[1]-C1)/(y[0]-C1))

def conduct_exponential_fit(xlist, ylist, C0, C1):
    print '= = Begin exponential fit.'
    xguess=[xlist[0],xlist[1]]
    yguess=[ylist[0],ylist[1]]
    guess=obtain_exponential_guess(xguess, yguess, C1)
    print '= = = guessed initial tau: ', guess
    fitOut = fmin_powell(powell_expdecay, guess, args=(xlist, ylist, C0, C1))
    print '= = = = Tau obtained: ', fitOut
    return fitOut

def quat_v1v2(v1, v2):
    """
    Return the minimum-angle quaternion that rotates v1 to v2.
    Non-SIMD version for clarity of maths.
    """
    th=acos(np.dot(v1,v2))
    ax=np.cross(v1,v2)
    if all( np.isnan(ax) ):
        # = = = This happens when the two vectors are identical
        return qops.qeye()
    else:
        # = = = Do normalisation within the next function
        return qops.axangle2quat(ax, th)

# Returns the minimum version that has the smallest cosine angle
# to the positive or negative axis.
def quat_frame_transform_min(axes):
    ref=np.array( ((1,0,0),(0,1,0),(0,0,1)) )

    q1a=quat_v1v2(axes[2],(0,0, 1))
    q1b=quat_v1v2(axes[2],(0,0,-1))
    q1 = q1a if q1a[0]>q1b[0] else q1b
    arot=[ qops.rotate_vector(axes[i],q1) for i in range(3)]

    q2a=quat_v1v2(arot[0],( 1,0,0))
    q2b=quat_v1v2(arot[0],(-1,0,0))
    q2 = q2a if q2a[0]>q2b[0] else q2b

    return qops.qmult(q2,q1)

def LegendreP1_quat(v_q):
    return 1 - 2*np.sum(np.square(v_q))

def average_LegendreP1quat(ndat, vq):
    out=0.0
    for i in range(ndat):
        out+=LegendreP1_quat(vq[i])
    return out/ndat

#def average_anisotropic_tensor(ndat, vq, qframe=(1,0,0,0)):
#    out=np.zeros((3,3))
#    # Check if there is no rotation.
#    if qops.nearly_equivalent(qframe,(1,0,0,0)):
#        for i in range(ndat):
#            out+=np.outer(vq[i],vq[i])
#    else:
#        for i in range(ndat):
#            vrot=qops.rotate_vector(vq[i],qframe)
#            out+=np.outer(vrot,vrot)
#    out/=ndat
#    return out

def average_anisotropic_tensor(vq, qframe=(1,0,0,0)):
      """
      Given the list of N components of q_v in te shape of (N,3)
      Calculates the average tensor < q_i q_j >, i,j={x,y,z}
      """
      ndat=vq.shape[0]
      #print "= = Debug: vq shape", vq.shape
      # Check if there is no rotation.
      if not qops.nearly_equivalent(qframe,(1,0,0,0)):
          vq=rotate_vector_simd(vq, qframe)
      return np.einsum('ij,ik->jk', vq, vq) / ndat

def quat_invert(q):
    return q*[1.0,-1.0,-1.0,-1.0]

def quat_mult_simd(q1, q2):
    """
    SIMD-version of transforms3d.quaternions.qmult using vector operations.
    when at least one of these is N-dimensional, it expects that q1 is the N-dimensional quantity.
    The axis is assumed to be the last one.
    To do this faster, we'll need a C-code ufunc.
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z])
    """
    #w1, v1 = decompose_quat(q1)
    #w2, v2 = decompose_quat(q2)
    #del v1 ; del v2
    out=np.zeros_like(q1)
    out[...,0]  = q1[...,0]*q2[...,0] - np.einsum('...i,...i',q1[...,1:4],q2[...,1:4])
    out[...,1:4]= q1[...,0,None]*q2[...,1:4] + q2[...,0,None]*q1[...,1:4] + np.cross(q1[...,1:4],q2[...,1:4])
    return out

def quat_reduce_simd(q, qref=(1,0,0,0),axis=-1):
    """
    Return the closer image of q or -q to a reference quaternion qref.
    The default is (1,0,0,0) to restrict rotations to be less than 180 degrees.
    """
    if axis==-1:
        sgn=np.sign( np.einsum('...i,i',q,qref) )
        sgn[sgn==0]=1.0
        return q*sgn[:,None]
    elif axis==0:
        sgn=np.sign( np.einsum('i...,i',q,qref) )
        sgn[sgn==0]=1.0
        return q*sgn[None,:]
    else:
        print sys.stderr, "= = = ERROR: quat_reduce_simd does not support arbitrary axis definitions!"
        sys.exit(1)


def obtain_self_dq(q, delta):
    """
    Vectorise { q^{-1}(t) q(t+delta) } over the range of q.
    Assumes that there are only two input dimensions q(N, 4) and returns dq(N-delta,4) with imaging to reduce rotations
    """
    return quat_reduce_simd( quat_mult_simd( quat_invert(q[:-delta]), q[delta:]) )

def rotmatrix_to_quaternion(time, matrix, bInvert=False):
    """
    Converts a matrix to quaternion, with a reverse if necessary
    """

    nPts=len(time)
    if nPts != len(matrix):
        print >> sys.stderr, "= = = ERROR in rotmatrix_to_quaternion: lengths are not the same!"
        return

    out=np.zeros( (5,nPts) )
    for i in range(nPts):
        out[0,i]=time[i]
        if bInvert:
            out[1:5,i]=qops.qinverse( qops.mat2quat( matrix[i] ) )
        else:
            out[1:5,i]=qops.mat2quat( matrix[i] )
    return out

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

def vecnorm_NDarray(v, axis=-1):
    """
    Vector normalisation performed along an arbitrary dimension, which by default is the last one.
    Comes with workaround by casting that to zero instead of keeping np.nan or np.inf.
    """
    # = = = need to protect against 0/0 errors when the vector is (0,0,0)
    if len(v.shape)>1:
        # = = = Obey broadcasting rules by applying 1 to the axis that is being reduced.
        sh=list(v.shape)
        sh[axis]=1
        return np.nan_to_num( v / np.linalg.norm(v,axis=axis).reshape(sh) )
    else:
        return np.nan_to_num( v/np.linalg.norm(v) )

def decompose_quat(q, axis=-1, bReshape=False):
    """
    Dynamic decomposition of quaternions into their w-component and v-component while preserving input axes.
    If bReshape is True, then the shape of q_w is augmented to have the same dimensionality as before
    with a value of 1 for broadcasting purposes.
    The reshaping apparently does not survive the function return....!
    """
    q=np.array(q)
    if axis==-1:
        q_w = q[...,0]
        q_v = q[...,1:4]
    elif axis==0:
        q_w = q[0,...]
        q_v = q[1:4,...]
    else:
        print sys.stderr, "= = = ERROR: decompose_quat does not support arbitrary axis definitions."
        sys.exit(1)

    if bReshape and len(q_w.shape)>1:
        newShape=list(q_w.shape)
        if axis==-1:
            newShape.append(1)
        elif axis==0:
            newShape.insert(0,1)
        q_w = q_w.reshape(newShape)

    return q_w, q_v

def average_LegendreP1quat_chunk(ndat, vq, nchunk):
    nblock=int(ceil(1.0*ndat/nchunk))
    out=np.zeros(nchunk)
    for i in range(nchunk):
        jmin=nblock*i
        jmax=min(ndat, nblock*(i+1))
        out[i]=average_LegendreP1quat(jmax-jmin, vq[jmin:jmax])
    return out

def average_anisotropic_tensor_chunk(vq, nchunk, qframe=(1,0,0,0)):
    ndat=vq.shape[0]
    nblock=int(ceil(1.0*ndat/nchunk))
    out=np.zeros((nchunk,3,3))
    for i in range(nchunk):
        jmin=nblock*i
        jmax=min(ndat, nblock*(i+1))
        out[i]=average_anisotropic_tensor(vq[jmin:jmax], qframe)
    return out

def get_flex_bounds(x, samples, nsig=1):
    """
    Here, we wish to report the distribution of the subchunks 'sample'
    along with the value of the full sample 'x'
    So this function will return x, x_lower_bound, x_upper_bound,
    where the range of the lower and upper bound expresses
    the standard deviation of the sample distribution, the mean
    of which is often not aligned with x.
    """
    mean=np.mean(samples); sig=np.std(samples)
    return [x, nsig*sig+x-mean, nsig*sig+mean-x]

bDoIso=True
bDoAniso=True
bDoFullTensor=False
bDoSubchunk=False
num_chunk=5
if num_chunk>1:
    bDoSubchunk=True
min_dt=0.0
max_dt=100000.0
skip_dt=10.0
in_fname_list='q_file'
out_pref='out'
i=0
nr_rep=len(in_fname_list.split())
tmp1, tmp2 = load_xys(in_fname_list.split()[0])
data_tmp = rotmatrix_to_quaternion(tmp1, tmp2, bInvert=True )
data=np.zeros((nr_rep,data_tmp.shape[0],data_tmp.shape[1]))
for in_fname in in_fname_list.split():
    tmp1, tmp2 = load_xys(in_fname)
    data[i] = rotmatrix_to_quaternion(tmp1, tmp2, bInvert=True )
    i=i+1
    tmp1=None
    tmp2=None

data = np.transpose(data, (0,2,1))
data_shape     = data.shape
num_replicas   = data_shape[0]
num_timepoints = data_shape[1]

print "= = Input data found to be as %i replicate trajectories, each with %i entries. = =" % (num_replicas, num_timepoints)
if num_chunk > 1 and num_replicas % num_chunk != 0:
    print >> sys.stderr, "= = ERROR: Uncertainty estimation is not appropriately handled: the number of sub_chunks must divide the total number of replicate trajectories! num_chunk: %d num_replicas: %d" % (num_chunk, num_replicas)
    sys.exit(1)

data_delta_t=data[0,1,0]-data[0,0,0]
skip_int=max(1,int(skip_dt/data_delta_t))
min_int=max(skip_int,int(min_dt/data_delta_t))
max_int=int(max_dt/data_delta_t)
q_frame=(1,0,0,0)
bFirst=True
nIntervals=(max_int-min_int)/skip_int+1
out_dtlist=np.zeros(nIntervals)
out_isolist=np.zeros(nIntervals)
out_aniso1list=np.zeros((3,nIntervals))
out_aniso2list=np.zeros((3,nIntervals))
out_qlist=np.zeros((nIntervals,4))
out_moilist=np.zeros((nIntervals,3,3))
if bDoFullTensor:
    out_RT=np.zeros((nIntervals,3,3))
if bDoSubchunk:
    chunk_isolist=np.zeros((num_chunk,nIntervals))
    #chunk_aniso1list=np.zeros((num_chunk,3,nIntervals))
    chunk_aniso2list=np.zeros((num_chunk,3,nIntervals))
    #chunk_qlist=np.zeros((num_chunk,nIntervals,4))
    #chunk_moilist=np.zeros((num_chunk,nIntervals,3,3))
index=0
for index, delta in enumerate (range(min_int,max_int+1,skip_int)):
    time_delta=delta*data_delta_t
    num_one=num_timepoints-delta
    num_nd=num_one*num_replicas
    #v_dq=obtain_self_dq(data[1:5].T, delta )[...,1:4]
    v_dq=np.zeros((num_nd,3), dtype=data.dtype)
    for ch in range(num_replicas):
        chunk = np.delete(np.take(data, indices=ch, axis=0), 0, axis=1)
        v_dq[num_one*ch:num_one*(ch+1)]=obtain_self_dq( chunk, delta )[...,1:4]
    iso_sum1=average_LegendreP1quat(num_nd, v_dq)
    moi=average_anisotropic_tensor(v_dq)
    if not qops.nearly_equivalent(q_frame,(1,0,0,0)):
        moiR1=average_anisotropic_tensor(v_dq, q_frame)
    else:
        moiR1=moi
    print " = = %i of %i intervals summed." % ((delta-min_int)/skip_int+1, nIntervals)
    out_dtlist[index]=time_delta
    if bDoIso:
        out_isolist[index]=iso_sum1
    if bDoAniso:
        eigval, eigvec = np.linalg.eigh(moi)
        moi_axes=eigvec.T
        q_rot=quat_frame_transform_min(moi_axes)
        if bFirst:
            bFirst=False
            q_frame=q_rot
            moiR1=average_anisotropic_tensor(v_dq, q_frame)
        out_aniso1list[:,index]=[ 1-2*eigval[0], 1-2*eigval[1], 1-2*eigval[2]]
        out_aniso2list[:,index]=[ 1-2*moiR1[0,0], 1-2*moiR1[1,1], 1-2*moiR1[2,2]]
        out_qlist[index] = q_rot
        out_moilist[index] = moi_axes
    if bDoSubchunk:
        chunk_isolist[:,index]=average_LegendreP1quat_chunk(num_nd, v_dq, num_chunk)
        #Anisotropic Diffusion. Need tensor of vq
        if not qops.nearly_equivalent(q_frame,(1,0,0,0)):
            temp2=average_anisotropic_tensor_chunk(v_dq, num_chunk, q_frame)
        else:
            temp2=average_anisotropic_tensor_chunk(v_dq, num_chunk)
        chunk_aniso2list[:,:,index]=[ [1-2*temp2[i][0,0], 1-2*temp2[i][1,1], 1-2*temp2[i][2,2]] for i in range(num_chunk) ]
    if bDoFullTensor:
        out_RT[index]=moiR1
    #index+=1

max_fit_Ctext=10001 # ps
nr_fitted_points=len(out_dtlist[out_dtlist<max_fit_Ctext])


if bDoIso:
    tau=conduct_exponential_fit(out_dtlist[:nr_fitted_points], out_isolist[:nr_fitted_points], 1.5, -0.5)
    model=build_model(isotropic_decay,tau,out_dtlist)
    if bDoSubchunk:
        chtaus=[ conduct_exponential_fit(out_dtlist[:nr_fitted_points], chunk_isolist[i,:nr_fitted_points], 1.5, -0.5) for i in range(num_chunk) ]
        chmodels=[ build_model(isotropic_decay,chtaus[i],out_dtlist) for i in range(num_chunk)]
        printlist=[[out_isolist, model]]
        for i in range(num_chunk):
            printlist.append([chunk_isolist[i],chmodels[i]])
        file_header=format_header('iso_err', tau, chtaus)
        print_model_fits_gen(out_pref+"-iso.dat", 3, file_header, out_dtlist, printlist)
    else:
        file_header=format_header('iso', tau)
        print_model_fits_gen(out_pref+"-iso.dat", 2, file_header, out_dtlist, [out_isolist, model] )

if bDoAniso:
    taus=[ conduct_exponential_fit(out_dtlist[:nr_fitted_points],out_aniso2list[i,:nr_fitted_points], 0.5, 0.5) for i in range(3) ]
    taus=np.array(taus)
    models=[ build_model(anisotropic_decay_noc, taus[i], out_dtlist) for i in range(3) ]
    if bDoSubchunk:
        chtaus=[[ conduct_exponential_fit(out_dtlist[:nr_fitted_points], chunk_aniso2list[i][j][:nr_fitted_points], 0.5, 0.5) for j in range(3) ] for i in range(num_chunk) ]
        chtaus=np.array(chtaus)
        chmodels=[[ build_model(anisotropic_decay_noc,chtaus[i][j],out_dtlist) for j in range(3) ] for i in range(num_chunk)]
        file_header=format_header('aniso_err', taus, chtaus)
        file_header.append( format_header_quat(q_frame) )
        printlist=[ np.concatenate((out_aniso2list, models)) ]
        for i in range(num_chunk):
            printlist.append( np.concatenate((chunk_aniso2list[i],chmodels[i])) )
        print_model_fits_gen(out_pref+"-aniso2.dat", 3, file_header, out_dtlist, printlist)
    else:
        file_header=format_header('aniso', taus)
        file_header.append( format_header_quat(q_frame) )
        print_model_fits_gen(out_pref+"-aniso2.dat", 2, file_header, out_dtlist, np.concatenate((out_aniso2list, models)) )
    print_xylist(out_pref+"-aniso_q.dat", out_dtlist, out_qlist)
    print_axes_as_xyz(out_pref+"-moi.xyz", out_moilist)

if bDoFullTensor:
        print_xylist(out_pref+"-tensor.dat", out_dtlist, out_RT.reshape(tot_int, 9))
