#/bin/bash

nr_sim=10

folder_list=""
xtc_list=""
refpdb_list=""
q_file_multi=""

for ((i=1; i<=${nr_sim}; i++)); do
        mkdir sim${i}
        folder_list=${folder_list}" sim${i}"
        xtc_list=${xtc_list}" ..\/NH\/sim${i}\/solute.xtc"
        refpdb_list=${refpdb_list}" sim${i}\/reference.pdb"
        q_file_multi=$(echo ${q_file_multi}" sim${i}\/colvar-qorient.xvg")
done

for ((i=1; i<=${nr_sim}; i++)); do
        j=$i
        cp ../NH/sim${i}/reference.pdb sim${i}/
        cp ../NH/sim${i}/topol.tpr sim${i}/
        #cp ../NH/sim${i}/solute.xtc sim${i}/
        cp ../NH/sim${i}/colvar-qorient.xvg sim${i}/
done

cp ../NH/solute.ndx .
cp ../NH/out-aniso_q.dat .
cp ../NH/out-aniso2.dat .
quat=$(head -n 1 out-aniso_q.dat | awk '{print $2, $3, $4, $5}')
sed "s/traj.xtc/${xtc_list}/g" calculateCt.py > calculateCt_tmp.py
sed -i "s/reference.pdb/${refpdb_list}/g" calculateCt_tmp.py
sed -i "s/quat_values/${quat}/g" calculateCt_tmp.py
python calculateCt_tmp.py
python calculate-fitted-Ct.py
sed "s/fittedCt.dat/fittedCt_C.dat/g" calculate-fitted-Ct.py > calculate-fitted-Ct_C.py
sed -i "s/Ctint.dat/Ctint_C.dat/g" calculate-fitted-Ct_C.py
python calculate-fitted-Ct_C.py
diffusion=$(./get_anisotropy.sh)
sed "s/diffusion_values/${diffusion}/g" calculate-relaxations-from-Ct.py > calculate-relaxations-from-Ct_tmp.py
python calculate-relaxations-from-Ct_tmp.py
sed -i "s/bJomega=False/bJomega=True/g" calculate-relaxations-from-Ct_tmp.py
python calculate-relaxations-from-Ct_tmp.py
