# MethylRelax
Scripts to calculate the NMR relaxation rates for backbone N-H and methyl C-H and C-C bonds from protein simulations.

The scripts in their current version are a copy of scripts from the SpinRelax program from Po-Chia Chen (https://github.com/zharmad), but with an extension to methyl relaxation rates. They have been originally used in this publication:
DOI:10.26434/chemrxiv.8982338 

All scripts should be executed in the order of their occurence in the full_process.sh file of the /NH and /methyl subfolders. The calculation of the rotational matrix is required for the calculation of backbone and methyl relaxation data, but only included in the full_process.sh file of the /NH subfolder. This part of the file has to be run before any calculation of relaxation data. Therefore, start to use the script in the /NH subfolder. After completion of all steps until transform.py one can use the script in the /methyl subfolder.

The following packages are needed for the analysis:
gromacs
mdtraj
python>2.7
numpy
scipy
npufunc (created via python ./setup.py build_ext --inplace)
transforms3d
