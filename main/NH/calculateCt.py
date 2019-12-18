#!/usr/bin/python2.7

import numpy as np
import mdtraj as md

def obtain_XHres(traj, seltxt):
    indexH = traj.topology.select(seltxt)
    if len(indexH) == 0:
        print "= = = ERROR: selection text failed to find atoms! ", seltxt
        print "     ....debug: N(%s) = %i " % (Hseltxt, numH)
        sys.exit(1)
    #    indexH = traj.topology.select("name H and resSeq 3")
    resXH = [ traj.topology.atom(indexH[i]).residue.resSeq for i in range(len(indexH)) ]
    return resXH

def print_sxylist(fn, legend, x, ylist, header=[]):
    fp = open( fn, 'w')
    for line in header:
        print >> fp, "%s" % line

    ylist=np.array(ylist)
    shape=ylist.shape
    nplot=len(ylist)
    s=0
    for i in range(nplot):
        print >> fp, "@s%d legend \"%s\"" % (s, legend[i])
        for j in range(len(x)):
            print >> fp, x[j], str(ylist[i][j]).strip('[]')
        print >> fp, "&"
        s+=1
    fp.close()

def calculate_Ct_Palmer(vecs):
    """
    Definition: < P2( v(t).v(t+dt) )  >
    (Rewritten) This proc assumes vecs to be of square dimensions ( nReplicates, nFrames, nResidues, 3).
    Operates a single einsum per delta-t timepoints to produce the P2(v(t).v(t+dt)) with dimensions ( nReplicates, nResidues )
    then produces the statistics from there according to Palmer's theory that trajectory be divide into N-replcates with a fixed memory time.
    Output Ct and dCt should take dimensions ( nResidues, nDeltas )
    """
    sh = vecs.shape
    print "= = = Debug of calculate_Ct_Palmer confirming the dimensions of vecs:", sh
    if sh[1]<50:
        print >> sys.stderr,"= = = WARNING: there are less than 50 frames per block of memory-time!"

    if len(sh)!=4:
        # Not in the right form...
        print >> sys.stderr, "= = = ERROR: The input vectors to calculate_Ct_Palmer is not of the expected 4-dimensional form! " % sh
        sys.exit(1)
    nReplicates=sh[0] ; nDeltas=sh[1]/2 ; nResidues=sh[2]
    Ct  = np.zeros( (nDeltas,nResidues), dtype=vecs.dtype )
    dCt = np.zeros( (nDeltas,nResidues), dtype=vecs.dtype )
    bFirst=True
    for delta in range(1,1+nDeltas):
        nVals=sh[1]-delta
        # = = Create < vi.v'i > with dimensions (nRep, nFr, nRes, 3) -> (nRep, nFr, nRes) -> ( nRep, nRes ), then average across replicates with SEM.
        tmp = -0.5 + 1.5 * np.square( np.einsum( 'ijkl,ijkl->ijk', vecs[:,:-delta,...] ,vecs[:,delta:,...] ) )
        tmp  = np.einsum( 'ijk->ik', tmp ) / nVals
        Ct[delta-1]  = np.mean( tmp,axis=0 )
        dCt[delta-1] = np.std( tmp,axis=0 ) / ( np.sqrt(nReplicates) - 1.0 )
        #if bFirst:
        #    bFirst=False
        #    print tmp.shape, P2.shape
        #    print tmp[0,0,0], P2[0,0]
        #Ct[delta-1]  = np.mean( tmp,axis=(0,1) )
        #dCt[delta-1] = np.std( tmp,axis=(0,1) ) / ( np.sqrt(nReplicates*nVals) - 1.0 )

    #print "= = Bond %i Ct computed. Ct(%g) = %g , Ct(%g) = %g " % (i, dt[0], Ct_loc[0], dt[-1], Ct_loc[-1])
    # Return with dimensions ( nDeltas, nResidues ) by default.
    return Ct, dCt

def calculate_dt(dt, tau):
    nPts = int(0.5*tau/dt)
    out = ( np.arange( nPts ) + 1.0) * dt
    return out

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

def obtain_XHvecs(traj, Hseltxt, Xseltxt):
    print "= = = Obtaining XH-vectors from trajectory..."
    #nFrames= traj.n_frames
    indexX = traj.topology.select(Xseltxt)
    indexH = traj.topology.select(Hseltxt)
    numX = len(indexX) ; numH = len(indexH)
    if numX == 0 or numH == 0 :
        print "= = = ERROR: selection text failed to find atoms!"
        print "     ....debug: N(%s) = %i , N(%s) = %i" % (Xseltxt, numX, Hseltxt, numH)
        sys.exit(1)
    if len(indexH) != len(indexX):
        print "= = = ERROR: selection text found different number of atoms!"
        print "     ....debug: N(%s) = %i , N(%s) = %i" % (Xseltxt, numX, Hseltxt, numH)
        sys.exit(1)
    #Do dangerous trick to select nitrogens connexted to HN..
    #indexX = [ indexH[i]-1 for i in range(len(indexH))]

    # Extract submatrix of vector trajectory
    vecXH = np.take(traj.xyz, indexH, axis=1) - np.take(traj.xyz, indexX, axis=1)
    vecXH = vecnorm_NDarray(vecXH, axis=2)
    return  vecXH


def reformat_vecs_by_tau(vecs, dt, tau):
    """
    This proc assumes that vecs list is N 3D-arrays in the form <Nfile>,(frames, bonds, XYZ).
    We take advantage of Palmer's iteration where the trajectory is divided into N chunks each of tau in length,
    to reformulate everything into fast 4D np.arrays of form (nchunk, frames, bonds, XYZ) so as to
    take full advantage of broadcasting.
    This will throw away additional frame data in each trajectory that does not fit into a single block of memory time tau.
    """
    # Don't assume all files have the same number of frames.
    nFiles = len(vecs)
    nFramesPerChunk=int(tau/dt)
    print "    ...debug: Using %i frames per chunk based on tau/dt (%g/%g)." % (nFramesPerChunk, tau, dt)
    used_frames     = np.zeros(nFiles, dtype=int)
    remainders = np.zeros(nFiles, dtype=int)
    for i in range(nFiles):
        nFrames = vecs[i].shape[0]
        used_frames[i] = int(nFrames/nFramesPerChunk)*nFramesPerChunk
        remainders[i] = nFrames % nFramesPerChunk
        print "    ...Source %i divided into %i chunks. Usage rate: %g %%" % (i, used_frames[i]/nFramesPerChunk, 100.0*used_frames[i]/nFrames )

    nFramesTot = int( used_frames.sum() )
    out = np.zeros( ( nFramesTot, vecs[0].shape[1], vecs[0].shape[2] ) , dtype=vecs[0].dtype)
    start = 0
    for i in range(nFiles):
        end=int(start+used_frames[i])
        endv=int(used_frames[i])
        out[start:end,...] = vecs[i][0:endv,...]
        start=end
    sh = out.shape
    print "    ...Done. vecs reformatted into %i chunks." % ( nFramesTot/nFramesPerChunk )
    return out.reshape ( (nFramesTot/nFramesPerChunk, nFramesPerChunk, sh[-2], sh[-1]) )

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

def xyz_to_rtp(uv, vaxis=-1, bUnit=False ):
    """
    Converts a vector or a set of vectors from the X/Y/Z to R/Phi/Theta.
    Noting that 0 ~ Theta ~ pi from positive Z.
    vaxis denotes the dimension in which the X/Y/Z value resides.
    This is the first (0) or last (-1) dimension of the array.
    """
    sh = uv.shape
    dims= len(sh)
    if bUnit:
        if dims == 1:
            rtp = np.zeros(2, dtype=uv.dtype)
            rtp[0] = np.arctan2(uv[1], uv[0])
            rtp[1] = np.arccos(uv[2]/rtp[0])
        elif vaxis==-1:
            rtp = np.zeros(_perturb_tuple(sh,mod=-1,axis=-1), dtype=uv.dtype)
            rtp[...,0] = np.arctan2(uv[...,1], uv[...,0])
            rtp[...,1] = np.arccos(uv[...,2]/rtp[...,0])
        elif vaxis==0:
            rtp = np.zeros(_perturb_tuple(sh,mod=-1,axis=0), dtype=uv.dtype)
            rtp[0,...] = np.arctan2(uv[1,...], uv[0,...])
            rtp[1,...] = np.arccos(uv[2,...]/rtp[0,...])
        else:
            print >> sys.stderr, "= = ERROR encountered in vec-to-rtp in general_maths.py, vaxis only accepts arguments of -1 or 0 for now."
    else:
        rtp = np.zeros_like(uv)
        if dims == 1:
            rtp[0] = np.linalg.norm(uv)
            rtp[1] = np.arctan2(uv[1], uv[0])
            rtp[2] = np.arccos(uv[2]/rtp[0])
        elif vaxis==-1:
            rtp[...,0] = np.linalg.norm(uv,axis=-1)
            rtp[...,1] = np.arctan2(uv[...,1], uv[...,0])
            rtp[...,2] = np.arccos(uv[...,2]/rtp[...,0])
        elif vaxis==0:
            rtp[0,...] = np.linalg.norm(uv,axis=0)
            rtp[1,...] = np.arctan2(uv[1,...], uv[0,...])
            rtp[2,...] = np.arccos(uv[2,...]/rtp[0,...])
        else:
            print >> sys.stderr, "= = ERROR encountered in vec-to-rtp in general_maths.py, vaxis only accepts arguments of -1 or 0 for now."
    return rtp

def print_s3d(fn, legend, arr, cols, header=[]):
    fp = open( fn, 'w')
    for line in header:
        print >> fp, "%s" % line

    shape=arr.shape
    ncols=len(cols)
    nplot=shape[0]
    s=0
    for i in range(nplot):
        print >> fp, "@s%d legend \"%s\"" % (s, legend[i])
        for j in range(shape[1]):
            for k in range(ncols):
                print >> fp, arr[i,j,cols[k]],
                print >> fp, " ",
            print >> fp, ""
        print >> fp, "&"
        s+=1
    fp.close()

def print_gplot_hist(fn, hist, edges, header='', bSphere=False):
    """
    Gnuplot is not a histogram plotter, and so we will plot each bin entry at the enter of the bin.
    For spherical outputs, the entire spherical coverage is assumed and additional data is printed to complete the sphere. These include:
    - Two column caps at the z-poles using the final values of ymin and ymax.
    - One additional row  at 2pi + 0.5*(x[0]+x[1]).
    Note that ndemunerate method iterates across all specified dimensions
    First gather and print some data as comments to aid interpretation and future scripting.
    """

    fp = open(fn, 'w')
    nbins=hist.shape
    dim=len(nbins)

    if header != '':
        print >> fp, '%s' % header
    print >> fp, '# DIMENSIONS: %i' % dim
    binwidth=np.zeros(dim, dtype=hist.dtype)
    print >> fp, '# BINWIDTH: ',
    for i in range(dim):
        binwidth[i]=(edges[i][-1]-edges[i][0])/nbins[i]
        print >> fp, '%g ' % binwidth[i],
    print >> fp, ''
    print >> fp, '# NBINS: ',
    for i in range(dim):
        print >> fp, '%g ' % nbins[i],
    print >> fp, ''
    if bSphere:
        if dim != 2:
            print >> sys.stderr, "= = = ERROR: histogram data is not in 2D, but spherical histogram plotting is requested!"
            sys.exit(1)
        # Handle spherical data by assuming that X is wrapped, and y is extended.
        # Can assume data is only 2D.
        xmin=0.5*(edges[0][0]+edges[0][1])
        ymin=edges[1][0]
        ymax=edges[1][-1]
        for eX in range(nbins[0]):
            xavg=0.5*(edges[0][eX]+edges[0][eX+1])
            # Print polar-caps to complete sphere
            print >> fp, '%g %g %g' % (xavg, ymin, hist[eX][0])
            for eY in range(nbins[1]):
                yavg=0.5*(edges[1][eY]+edges[1][eY+1])
                print >> fp, '%g %g %g' % (xavg, yavg, hist[eX][eY])
            print >> fp, '%g %g %g' % (xavg, ymax, hist[eX][-1])
            print >> fp, ''
        # Print first line again to complete sphere, with 2-pi deviation just in case.
        print >> fp, '%g %g %g' % (xmin+2*np.pi, ymin, hist[0][0])
        for eY in range(nbins[1]):
            yavg=0.5*(edges[1][eY]+edges[1][eY+1])
            print >> fp, '%g %g %g' % (xmin+2*np.pi, yavg, hist[0][eY])
        print >> fp, '%g %g %g' % (xmin+2*np.pi, ymax, hist[0][-1])
        print >> fp, ''
    else:
        for index, val in np.ndenumerate(hist):
            for i in range(dim):
                x=(edges[i][index[i]]+edges[i][index[i]+1])/2.0
                print >> fp, '%g ' % x ,
            print >> fp, '%g' % val
            if index[-1] == nbins[-1]-1:
                print >> fp, ''
    fp.close()

def get_indices_mdtraj( seltxt, top, filename):
    """
    NB: A workaround for MDTraj is needed becusae the standard reader
    does not return topologies.
    """
    if seltxt == 'custom occupancy':
        pdb  = md.formats.pdb.pdbstructure.PdbStructure(open(filename))
        mask = [ atom.get_occupancy() for atom in pdb.iter_atoms() ]
        inds = top.select('all')
        return [ inds[i] for i in range(len(mask)) if mask[i] > 0.0 ]
    else:
        return top.select(seltxt)

bRotVec=True
bDoVecDistrib=False
bBinary=True
bDoVecHist=True
if bDoVecHist and not bDoVecDistrib:
    bDoVecDistrib=True
histBin=72
histBinX=histBin
histBinY=histBinX/2
q_rot=np.array( [ float(v) for v in 'quat_values'.split() ] )

out_pref='out'
bDoCt=True
Hseltxt='name H'
Xseltxt='name N and not resname PRO'
in_flist=[i for i in 'traj.xtc'.split()]
in_reflist=[i for i in 'reference.pdb'.split()]
n_refs = len(in_reflist)
n_trjs = len(in_flist)
if n_refs == 1:
    bMultiRef=False
    top_filename=in_reflist[0]
    ref = md.load(top_filename)
    print "= = = Loaded single reference file: %s" % (top_filename)
    # Load the atom indices over which the atom fit will take place.
    fit_indices = get_indices_mdtraj(top=ref.topology, filename=top_filename, seltxt=fittxt)
    print "= = = Debug: fit_indices number: %i" % len(fit_indices)
else:
    print "= = = Detected multiple reference file inputs."
    bMultiRef=True
    if n_refs != n_trjs:
        print >> sys.stderr, "= = ERROR: When giving multiple reference files, you must have one for each trajecfile file given!"
        sys.exit(1)

tau_memory=10000.0
top_filename=in_reflist[0]
ref = md.load(top_filename)
fittxt='custom occupancy'
print "= = = Loaded single reference file: %s" % (top_filename)
#Load the atom indices over which the atom fit will take place.
fit_indices = get_indices_mdtraj(top=ref.topology, filename=top_filename, seltxt=fittxt)
print "= = = Debug: fit_indices number: %i" % len(fit_indices)
resXH = []
vecXH = []
vecXHfit = []
deltaT = np.nan
nBonds = np.nan
bFirst=True
for i in range(n_trjs):
    if bMultiRef:
        top_filename=in_reflist[i]
        ref = md.load(top_filename)
        print "= = = Loaded reference file %i: %s" % (i, top_filename)
        fit_indices = get_indices_mdtraj( top=ref.topology, filename=top_filename, seltxt=fittxt)
        print "= = = Debug: fit_indices number: %i" % len(fit_indices)
    trj = md.load(in_flist[i], top=top_filename)
    deltaT_loc = trj.timestep
    resXH_loc = obtain_XHres(trj, Hseltxt)
    vecXH_loc = obtain_XHvecs(trj, Hseltxt, Xseltxt)
    trj.center_coordinates()
    print "= = DEBUG: Fitted indices are ", fit_indices
    trj.superpose(ref, frame=0, atom_indices=fit_indices )
    print "= = = Molecule centered and fitted."
    vecXHfit_loc = obtain_XHvecs(trj, Hseltxt, Xseltxt)
    nBonds_loc = vecXH_loc.shape[1]
    del trj
    if bFirst:
        resXH = resXH_loc
        deltaT = deltaT_loc
        nBonds = nBonds_loc
    else:
        if deltaT != deltaT_loc or nBonds != nBonds_loc or not np.equal(resXH, resXH_loc):
            print >> sys.stderr, "= = = ERROR: Differences in trajectories have been detected! Aborting."
            print >> sys.stderr, "      ...delta-t: %g vs.%g " % (deltaT, deltaT_loc)
            print >> sys.stderr, "      ...n-bonds: %g vs.%g " % (nBonds, nBonds_loc)
            print >> sys.stderr, "      ...Residue-XORs: %s " % ( set(resXH)^set(resXH_loc) )
    vecXH.append(vecXH_loc)
    vecXHfit.append(vecXHfit_loc)
del vecXH_loc
del vecXHfit_loc
vecXH=np.array(vecXH)
vecXHfit=np.array(vecXHfit)
vecXH = reformat_vecs_by_tau(vecXH, deltaT, tau_memory)
vecXHfit = reformat_vecs_by_tau(vecXHfit, deltaT, tau_memory)
if bDoCt:
        print "= = = Conducting Ct_external using Palmer's approach."
        print "= = = timestep: ", deltaT, "ps"
        print "= = = tau_memory: ", tau_memory, "ps"
        #if n_trjs > 1:
        #    print "= = = N.B.: For multiple files, 2D averaging is conducted at each datapoint."
        dt = calculate_dt(deltaT, tau_memory)
        Ct, dCt = calculate_Ct_Palmer(vecXH)
        #Ct = calculate_Ct_Palmer(vecXH)
        print_sxylist(out_pref+'_Ctext.dat', resXH, dt, np.stack( (Ct.T,dCt.T), axis=-1) )
        #print_sxylist(out_pref+'_Ctext.dat', resXH, dt, np.stack( (Ct.T), axis=-1) )
        print "= = = Conducting Ct_internal using Palmer's approach."
        Ct, dCt = calculate_Ct_Palmer(vecXHfit)
        #Ct = calculate_Ct_Palmer(vecXHfit)
        print_sxylist(out_pref+'_Ctint.dat', resXH, dt, np.stack( (Ct.T,dCt.T), axis=-1) )
        #print_sxylist(out_pref+'_Ctint.dat', resXH, dt, np.stack( (Ct.T), axis=-1) )
        del Ct, dCt
sh = vecXHfit.shape
vecXHfit = vecXHfit.reshape( ( sh[0]*sh[1], sh[-2], sh[-1]) )
for i in range(sh[0]):
    vecXHfit[i*sh[1]:(i+1)*sh[1]] = rotate_vector_simd(vecXHfit[i*sh[1]:(i+1)*sh[1] ], q_rot)
if bDoVecDistrib:
    print "= = = Converting vectors into spherical coordinates."
    try:
        rtp = xyz_to_rtp(vecXHfit)
        # print type(rtp[0,0,0])
    except MemoryError:
        print >> sys.stderr, "= = ERROR: Ran out of memory running spherical conversion!"
        sys.exit(9)
        # = = = Don't bother rescuing.
        #for i in range(sh[0]*sh[1]):
        #    vecXHfit[i] = get_spherical_coords(vecXHfit[i])
        #vecXHfit = np.transpose(vecXHfit,axes=(1,0,2)) ;# Convert from time first, to resid first.
        #gs.print_s3d(out_pref+'_PhiTheta.dat', resXH, vecXHfit, (1,2))
        #sys.exit(9)
    rtp = np.transpose(rtp,axes=(1,0,2)) ;# Convert from time first, to resid first.
    print "= = = Debug: shape of the spherical vector distribution:", rtp.shape
    if not bDoVecHist:
        if bBinary:
            np.savez_compressed(out_pref+'_vecPhiTheta.npz', names=resXH, dataType='PhiTheta', axisLabels=['phi','theta'], bHistogram=False, data=rtp[...,1:3])
        else:
            print_s3d(out_pref+'_vecPhiTheta.dat', resXH, rtp, (1,2))
    else:
        # = = = Conduct histograms on the dimension of phi, cos(theta). The Lambert Cylindrical projection preserves sample area, and therefore preserves the bin occupancy rates.
        # = = = ...this leads to relative bin occupancies that make more sense when plotted directly.
        rtp = np.delete( rtp, 0, axis=2)
        print "= = = Histgrams will use Lambert Cylindrical projection by converting Theta spanning (0,pi) to cos(Theta) spanning (-1,1)"
        rtp[...,1]=np.cos(rtp[...,1])
        hist_list=np.zeros((nBonds,histBinX,histBinY), dtype=rtp.dtype)
        bFirst=True
        for i in range(nBonds):
            hist, edges = np.histogramdd(rtp[i],bins=(histBinX,histBinY),range=((-np.pi,np.pi),(-1,1)),normed=False)
            if bFirst:
                bFirst=False
                edgesAll = edges
            #else:
            #    if np.any(edgesAll!=edges):
            #        print >> sys.stderr, "= = = ERROR: histogram borders are not equal. This should never happen!"
            #        sys.exit(1)
            hist_list[i]=hist
        if bBinary:
            np.savez_compressed(out_pref+'_vecHistogram.npz', names=resXH, dataType='LambertCylindrical', bHistogram=True, edges=edgesAll, axisLabels=['phi','cos(theta)'], data=hist_list)
        else:
            for i in range(nBonds):
                ofile=out_pref+'_vecXH_'+str(resXH[i])+'.hist'
                print_gplot_hist(ofile, hist, edges, header='# Lamber Cylindrical Histogram over phi,cos(theta).', bSphere=True)
                #gs.print_R_hist(ofile, hist, edges, header='# Lamber Cylindrical Histogram over phi,cos(theta).')
                print "= = = Written to output: ", ofile
