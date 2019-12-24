# MethylRelax
Scripts to calculate the NMR relaxation rates for backbone N-H and methyl C-H and C-C bonds from protein simulations.

Parts of the code were adapted from https://github.com/zharmad, but here with an extension to methyl relaxation rates. They have been originally used in this publication:

DOI:10.26434/chemrxiv.8982338 

All scripts should be executed in the order of their occurence in the file full_process.sh file of the /NH and /methyl subfolders. The calculation of the rotational matrix is required for the calculation of backbone and methyl relaxation data, but only included in the full_process.sh file of the /NH subfolder. This part of the file has to be run before any calculation of relaxation data. Therefore, start to use the script in the /NH subfolder. After completion of all steps until transform.py one can use the script in the /methyl subfolder.

The following packages are needed for the analysis:

gromacs

mdtraj

python>2.7

numpy

scipy

npufunc (created via python ./setup.py build_ext --inplace)

transforms3d

The force field folder contains python scripts which can be used to convert a gromacs generated topology into a new topology with reparametrized methyl rotation barriers as described in these publications:

DOI:10.1021/acs.jpcb.8b02769

DOI:10.26434/chemrxiv.8982338 
