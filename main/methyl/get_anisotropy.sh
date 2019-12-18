#/bin/bash

T_md=300 #K
T_exp=300 #K
c_D2O=0.09
outpref='out'

function convert_Diso() {
    # See Garcia et al., J Magn Res, 2000 + Wong and Case, J Phys Chem B, 2008
    awk -v D1=$1 -v T1=$2 -v T2=$3 -v cD2O=$4 \
'function eta(T) {
    T=T-273
    return 1.7753 - 5.65e-2*T + 1.0751e-3*T^2 - 9.222e-6*T^3
}
function D2Omod(ratio){
    return 1.23*ratio+(1.0-ratio)
}
BEGIN {
    print D1 * (1.0*T2/T1) * ( 1.0*eta(T1)/eta(T2) ) * D2Omod(cD2O)
}'
}

#D_fact=$(convert_Diso 1 $T_md $T_exp $c_D2O)
D_fact=1.0
Diso_loc=$(head -n 20 ${outpref}-aniso2.dat | grep Diso | awk -v fact=$D_fact '{print $(NF-3)*1e-12*fact}')
DaniL_loc=$(head -n 20 ${outpref}-aniso2.dat | grep Dani_L | awk '{print $(NF-2)}')
DrhoL_loc=$(head -n 20 ${outpref}-aniso2.dat | grep Drho_L | awk '{print $(NF-2)}')
DaniS_loc=$(head -n 20 ${outpref}-aniso2.dat | grep Dani_S | awk '{print $(NF-2)}')
DrhoS_loc=$(head -n 20 ${outpref}-aniso2.dat | grep Drho_S | awk '{print $(NF-2)}')

symmaxis=$(echo $DrhoL_loc $DrhoS_loc | awk '{if ($1<1.0) {
    print "z"
} else if ($2<1.0) {
    print "x"
} else {
    print "ERROR"
}
}')

if [[ "$symmaxis" == "z" ]] ; then
    #echo "= = = Long axis ellipsoid detected, pointing along Dz."
    Dani_loc=$DaniL_loc
elif [[ "$symmaxis" == "x" ]] ; then
    #echo "= = = Short axis ellipsoid detected, pointing along Dx."
    Dani_loc=$DaniS_loc
else
    #echo "= = = ERROR: neither Drho values are less than one in the global rotation diffusion calculation. This is not possible, therefore aborting."
    exit 1
fi

echo "${Diso_loc} ${Dani_loc}"
