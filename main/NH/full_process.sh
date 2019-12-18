#!/bin/bash

nr_sim=10

folder_list=""
xtc_list=""
refpdb_list=""
q_file_multi=""

# Copy files in subfolders
for ((i=1; i<=${nr_sim}; i++)); do
        mkdir sim${i}
        folder_list=${folder_list}" sim${i}"
        xtc_list=${xtc_list}" sim${i}\/solute.xtc"
        refpdb_list=${refpdb_list}" sim${i}\/reference.pdb"
        q_file_multi=$(echo ${q_file_multi}" sim${i}\/colvar-qorient.xvg")
        #cp ${TPR_file}.tpr sim${i}/topol.tpr
        #cp {XTC_file}.xtc sim${i}/traj.xtc
done

# center the protein, calculate rotational matrix
for ((i=1; i<=${nr_sim}; i++)); do
        j=$i
        ./create-reference-pdb.bash sim${i}/reference.pdb sim${i}/topol.tpr
        ./center-solute-gromacs.bash sim${i}/solute.xtc sim${i}/topol.tpr sim${i}/traj.xtc
        gmx rotmat -f sim${i}/solute.xtc -s sim${i}/topol.tpr -n solute.ndx -o sim${i}/colvar-qorient
done

sed "s/q_file/${q_file_multi}/g" transform.py > transform_tmp.py
python transform_tmp.py
quat=$(head -n 1 out-aniso_q.dat | awk '{print $2, $3, $4, $5}')
sed "s/traj.xtc/${xtc_list}/g" calculateCt.py > calculateCt_tmp.py
sed -i "s/reference.pdb/${refpdb_list}/g" calculateCt_tmp.py
sed -i "s/quat_values/${quat}/g" calculateCt_tmp.py
python calculateCt_tmp.py
python calculate-fitted-Ct.py
diffusion=$(./get_anisotropy.sh)
sed "s/diffusion_values/${diffusion}/g" calculate-relaxations-from-Ct.py > calculate-relaxations-from-Ct_tmp.py
python calculate-relaxations-from-Ct_tmp.py
sed -i "s/bJomega=False/bJomega=True/g" calculate-relaxations-from-Ct_tmp.py
python calculate-relaxations-from-Ct_tmp.py
