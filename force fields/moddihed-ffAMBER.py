#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
   Requirements:  Python 2.7 or higher or Python 3.x

   This code is released under GNU Lesser General Public License
   See https://www.gnu.org/copyleft/lesser.html for more details

   Contact: Prof. Lars Schäfer, lars.schaefer@ruhr-uni-bochum.de

   Please contact us if you face any issues with this code. Thank you.
'''

import sys
__version__ = '201804a'
__author__ = 'Molecular Simulation Group (AK Schäfer) @ Ruhr University, Bochum, Germany'


topheader = ''';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Modified dihedral parameters for ALA, ILE, LEU, MET and VAL
;
; Please read and cite the following article(s) if you use our modified
; force field for your work. Thank you.
;     Hoffmann et al, DOI: 10.1021/acs.jpcb.8b02769
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;'''

import argparse

parser=argparse.ArgumentParser(
    description='''Modified dihedral parameters for ALA, ILE, LEU, MET and VAL. ''')
parser.add_argument('input', nargs=1, default="input.top", help='input topology (itp,top)')
args=parser.parse_args()

class RtError(Exception):
    pass


def ParseAtom(line, n, modatoms):
    RES = ['ALA', 'ILE', 'LEU', 'MET', 'VAL']
    ATOM = ['C', 'CA', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2', 'CE', 'HA', 'HB', 'HB1', 'HB2', 'HB3', 'HG', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'HD1', 'HD2', 'HD3', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23', 'HE1', 'HE2', 'HE3', 'N', 'SD']
    if ';' in line:
        entry, comm = line.split(';', 1)
    else:
        entry = line
        

    tokens = entry.split()
    if len(tokens) < 8:
        msg = 'Number of Columns in the atom section must be >= 8 not {0}'.format(
            len(tokens))
        raise RtError(
            'ParseAtom:  on linenum: {0}  errmsg:{1}'.format(
                n + 1, msg))
    else:
        # 1    N3   1    MET   N   1   0.1592      14.01   ; qtot 0.1592
        resn = tokens[3]
        aname = tokens[4]
        atype = tokens[1]
        anumb = tokens[5]
        amass = float(tokens[-1])
        if 'MCH3' in atype and (amass >= 7 and  amass <= 8):
            msg = 'No methyl group parametrization with virtual sites.'
            raise RtError(
                  'ParseAtom on linenum: {0} errmsg:{1}'.format(
                      n + 1, msg))

        if resn in RES and aname in ATOM:
            modatoms.append((anumb, aname, resn))
    return modatoms


def ParseDihedral(line, n, modatoms):
    DICT = {
        'CBCGCD1HD11LEU': '     0.00   0.50674   3; LEU CB-CG-CD1-HD11 Modified',
        'CBCGCD1HD12LEU': '     0.00   0.50674   3; LEU CB-CG-CD1-HD12 Modified',
        'CBCGCD1HD13LEU': '     0.00   0.50674   3; LEU CB-CG-CD1-HD13 Modified',
        'CBCGCD2HD21LEU': '     0.00   0.50674   3; LEU CB-CG-CD2-HD21 Modified',
        'CBCGCD2HD22LEU': '     0.00   0.50674   3; LEU CB-CG-CD2-HD22 Modified',
        'CBCGCD2HD23LEU': '     0.00   0.50674   3; LEU CB-CG-CD2-HD23 Modified',
        'HGCGCD1HD11LEU': '     0.00   0.46490   3; LEU HG-CG-CD1-HD11 Modified',
        'HGCGCD1HD12LEU': '     0.00   0.46490   3; LEU HG-CG-CD1-HD12 Modified',
        'HGCGCD1HD13LEU': '     0.00   0.46490   3; LEU HG-CG-CD1-HD13 Modified',
        'HGCGCD2HD21LEU': '     0.00   0.46490   3; LEU HG-CG-CD2-HD21 Modified',
        'HGCGCD2HD22LEU': '     0.00   0.46490   3; LEU HG-CG-CD2-HD22 Modified',
        'HGCGCD2HD23LEU': '     0.00   0.46490   3; LEU HG-CG-CD2-HD23 Modified',
        'CD1CGCD2HD21LEU': '     0.00   0.50674   3; LEU CD1-CG-CD2-HD21 Modified',
        'CD1CGCD2HD22LEU': '     0.00   0.50674   3; LEU CD1-CG-CD2-HD22 Modified',
        'CD1CGCD2HD23LEU': '     0.00   0.50674   3; LEU CD1-CG-CD2-HD23 Modified',
        'CD2CGCD1HD11LEU': '     0.00   0.50674   3; LEU CD2-CG-CD1-HD11 Modified',
        'CD2CGCD1HD12LEU': '     0.00   0.50674   3; LEU CD2-CG-CD1-HD12 Modified',
        'CD2CGCD1HD13LEU': '     0.00   0.50674   3; LEU CD2-CG-CD1-HD13 Modified',
        'CBCG1CDHD1ILE': '     0.00   0.50674   3; ILE CB-CG1-CD-HD1 Modified',
        'CBCG1CDHD2ILE': '     0.00   0.50674   3; ILE CB-CG1-CD-HD2 Modified',
        'CBCG1CDHD3ILE': '     0.00   0.50674   3; ILE CB-CG1-CD-HD3 Modified',
        'CACBCG2HG21ILE': '     0.00   0.36724   3; ILE CA-CB-CG2-HG21 Modified',
        'CACBCG2HG22ILE': '     0.00   0.36724   3; ILE CA-CB-CG2-HG22 Modified',
        'CACBCG2HG23ILE': '     0.00   0.36724   3; ILE CA-CB-CG2-HG23 Modified',
        'HG11CG1CDHD1ILE': '     0.00   0.46490   3; ILE HG11-CG1-CD-HD1 Modified',
        'HG11CG1CDHD2ILE': '     0.00   0.46490   3; ILE HG11-CG1-CD-HD2 Modified',
        'HG11CG1CDHD3ILE': '     0.00   0.46490   3; ILE HG11-CG1-CD-HD3 Modified',
        'HBCBCG2HG21ILE': '     0.00   0.32540   3; ILE HB-CB-CG2-HG21 Modified',
        'HBCBCG2HG22ILE': '     0.00   0.32540   3; ILE HB-CB-CG2-HG22 Modified',
        'HBCBCG2HG23ILE': '     0.00   0.32540   3; ILE HB-CB-CG2-HG23 Modified',
        'HG12CG1CDHD1ILE': '     0.00   0.46490   3; ILE HG12-CG1-CD-HD1 Modified',
        'HG12CG1CDHD2ILE': '     0.00   0.46490   3; ILE HG12-CG1-CD-HD2 Modified',
        'HG12CG1CDHD3ILE': '     0.00   0.46490   3; ILE HG12-CG1-CD-HD3 Modified',
        'CG1CBCG2HG21ILE': '     0.00   0.36724   3; ILE CG1-CB-CG2-HG21 Modified',
        'CG1CBCG2HG22ILE': '     0.00   0.36724   3; ILE CG1-CB-CG2-HG22 Modified',
        'CG1CBCG2HG23ILE': '     0.00   0.36724   3; ILE CG1-CB-CG2-HG23 Modified',
        'CACBCG1HG11VAL': '     0.00   0.36724   3; VAL CA-CB-CG1-HG21 Modified',
        'CACBCG1HG12VAL': '     0.00   0.36724   3; VAL CA-CB-CG1-HG22 Modified',
        'CACBCG1HG13VAL': '     0.00   0.36724   3; VAL CA-CB-CG1-HG23 Modified',
        'CACBCG2HG21VAL': '     0.00   0.36724   3; VAL CA-CB-CG2-HG21 Modified',
        'CACBCG2HG22VAL': '     0.00   0.36724   3; VAL CA-CB-CG2-HG22 Modified',
        'CACBCG2HG23VAL': '     0.00   0.36724   3; VAL CA-CB-CG2-HG23 Modified',
        'HBCBCG1HG11VAL': '     0.00   0.32540   3; VAL HB-CB-CG1-HG11 Modified',
        'HBCBCG1HG12VAL': '     0.00   0.32540   3; VAL HB-CB-CG1-HG12 Modified',
        'HBCBCG1HG13VAL': '     0.00   0.32540   3; VAL HB-CB-CG1-HG13 Modified',
        'HBCBCG2HG21VAL': '     0.00   0.32540   3; VAL HB-CB-CG2-HG21 Modified',
        'HBCBCG2HG22VAL': '     0.00   0.32540   3; VAL HB-CB-CG2-HG22 Modified',
        'HBCBCG2HG23VAL': '     0.00   0.32540   3; VAL HB-CB-CG2-HG23 Modified',
        'CG2CBCG1HG11VAL': '     0.00   0.36724   3; VAL CG2-CB-CG1-HG11 Modified',
        'CG2CBCG1HG12VAL': '     0.00   0.36724   3; VAL CG2-CB-CG1-HG12 Modified',
        'CG2CBCG1HG13VAL': '     0.00   0.36724   3; VAL CG2-CB-CG1-HG13 Modified',
        'CG1CBCG2HG21VAL': '     0.00   0.36724   3; VAL CG1-CB-CG2-HG21 Modified',
        'CG1CBCG2HG22VAL': '     0.00   0.36724   3; VAL CG1-CB-CG2-HG22 Modified',
        'CG1CBCG2HG23VAL': '     0.00   0.36724   3; VAL CG1-CB-CG2-HG23 Modified',
        'CCACBHB1ALA': '     0.00   0.58120   3; ALA C-CA-CB-HB1 Modified',
        'CCACBHB2ALA': '     0.00   0.58120   3; ALA C-CA-CB-HB2 Modified',
        'CCACBHB3ALA': '     0.00   0.58120   3; ALA C-CA-CB-HB3 Modified',
        'HACACBHB1ALA': '     0.00   0.58120   3; ALA HA-CA-CB-HB1 Modified',
        'HACACBHB2ALA': '     0.00   0.58120   3; ALA HA-CA-CB-HB2 Modified',
        'HACACBHB3ALA': '     0.00   0.58120   3; ALA HA-CA-CB-HB3 Modified',
        'NCACBHB1ALA': '     0.00   0.58120   3; ALA N-CA-CB-HB1 Modified',
        'NCACBHB2ALA': '     0.00   0.58120   3; ALA N-CA-CB-HB2 Modified',
        'NCACBHB3ALA': '     0.00   0.58120   3; ALA N-CA-CB-HB3 Modified',
        'CGSDCEHE1MET': '     0.00   1.08087   3; MET CG-SD-CE-HE1 Modified',
        'CGSDCEHE2MET': '     0.00   1.08087   3; MET CG-SD-CE-HE2 Modified',
        'CGSDCEHE3MET': '     0.00   1.08087   3; MET CG-SD-CE-HE3 Modified',

        'HD11CD1CGCBLEU': '     0.00   0.50674   3; LEU HD11-CD1-CG-CB Modified',
        'HD12CD1CGCBLEU': '     0.00   0.50674   3; LEU HD12-CD1-CG-CB Modified',
        'HD13CD1CGCBLEU': '     0.00   0.50674   3; LEU HD13-CD1-CG-CB Modified',
        'HD21CD2CGCBLEU': '     0.00   0.50674   3; LEU HD21-CD2-CG-CB Modified',
        'HD22CD2CGCBLEU': '     0.00   0.50674   3; LEU HD22-CD2-CG-CB Modified',
        'HD23CD2CGCBLEU': '     0.00   0.50674   3; LEU HD23-CD2-CG-CB Modified',
        'HD11CD1CGHGLEU': '     0.00   0.46490   3; LEU HD11-CD1-CG-HG Modified',
        'HD12CD1CGHGLEU': '     0.00   0.46490   3; LEU HD12-CD1-CG-HG Modified',
        'HD13CD1CGHGLEU': '     0.00   0.46490   3; LEU HD13-CD1-CG-HG Modified',
        'HD21CD2CGHGLEU': '     0.00   0.46490   3; LEU HD21-CD2-CG-HG Modified',
        'HD22CD2CGHGLEU': '     0.00   0.46490   3; LEU HD22-CD2-CG-HG Modified',
        'HD23CD2CGHGLEU': '     0.00   0.46490   3; LEU HD23-CD2-CG-HG Modified',
        'HD21CD2CGCD1LEU': '     0.00   0.50674   3; LEU HD21-CD2-CG-CD1 Modified',
        'HD22CD2CGCD1LEU': '     0.00   0.50674   3; LEU HD22-CD2-CG-CD1 Modified',
        'HD23CD2CGCD1LEU': '     0.00   0.50674   3; LEU HD23-CD2-CG-CD1 Modified',
        'HD11CD1CGCD2LEU': '     0.00   0.50674   3; LEU HD11-CD1-CG-CD2 Modified',
        'HD12CD1CGCD2LEU': '     0.00   0.50674   3; LEU HD12-CD1-CG-CD2 Modified',
        'HD13CD1CGCD2LEU': '     0.00   0.50674   3; LEU HD13-CD1-CG-CD2 Modified',
        'HD1CDCG1CBILE': '     0.00   0.50674   3; ILE HD1-CD-CG1-CB Modified',
        'HD2CDCG1CBILE': '     0.00   0.50674   3; ILE HD2-CD-CG1-CB Modified',
        'HD3CDCG1CBILE': '     0.00   0.50674   3; ILE HD3-CD-CG1-CB Modified',
        'HG21CG2CBCAILE': '     0.00   0.36724   3; ILE HG21-CG2-CB-CA Modified',
        'HG22CG2CBCAILE': '     0.00   0.36724   3; ILE HG22-CG2-CB-CA Modified',
        'HG23CG2CBCAILE': '     0.00   0.36724   3; ILE HG23-CG2-CB-CA Modified',
        'HD1CDCG1HG11ILE': '     0.00   0.46490   3; ILE HD1-CD-CG1-HG11 Modified',
        'HD2CDCG1HG11ILE': '     0.00   0.46490   3; ILE HD2-CD-CG1-HG11 Modified',
        'HD3CDCG1HG11ILE': '     0.00   0.46490   3; ILE HD3-CD-CG1-HG11 Modified',
        'HG21CG2CBHBILE': '     0.00   0.32540   3; ILE HG21-CG2-CB-HB Modified',
        'HG22CG2CBHBILE': '     0.00   0.32540   3; ILE HG22-CG2-CB-HB Modified',
        'HG23CG2CBHBILE': '     0.00   0.32540   3; ILE HG23-CG2-CB-HB Modified',
        'HD1CDCG1HG12ILE': '     0.00   0.46490   3; ILE HD1-CD-CG1-HG12 Modified',
        'HD2CDCG1HG12ILE': '     0.00   0.46490   3; ILE HD2-CD-CG1-HG12 Modified',
        'HD3CDCG1HG12ILE': '     0.00   0.46490   3; ILE HD3-CD-CG1-HG12 Modified',
        'HG21CG2CBCG1ILE': '     0.00   0.36724   3; ILE HG21-CG2-CB-CG1 Modified',
        'HG22CG2CBCG1ILE': '     0.00   0.36724   3; ILE HG22-CG2-CB-CG1 Modified',
        'HG23CG2CBCG1ILE': '     0.00   0.36724   3; ILE HG23-CG2-CB-CG1 Modified',
        'HG11CG1CBCAVAL': '     0.00   0.36724   3; VAL HG21-CG1-CB-CA Modified',
        'HG12CG1CBCAVAL': '     0.00   0.36724   3; VAL HG22-CG1-CB-CA Modified',
        'HG13CG1CBCAVAL': '     0.00   0.36724   3; VAL HG23-CG1-CB-CA Modified',
        'HG21CG2CBCAVAL': '     0.00   0.36724   3; VAL HG21-CG2-CB-CA Modified',
        'HG22CG2CBCAVAL': '     0.00   0.36724   3; VAL HG22-CG2-CB-CA Modified',
        'HG23CG2CBCAVAL': '     0.00   0.36724   3; VAL HG23-CG2-CB-CA Modified',
        'HG11CG1CBHBVAL': '     0.00   0.32540   3; VAL HG11-CG1-CB-HB Modified',
        'HG12CG1CBHBVAL': '     0.00   0.32540   3; VAL HG12-CG1-CB-HB Modified',
        'HG13CG1CBHBVAL': '     0.00   0.32540   3; VAL HG13-CG1-CB-HB Modified',
        'HG21CG2CBHBVAL': '     0.00   0.32540   3; VAL HG21-CG2-CB-HB Modified',
        'HG22CG2CBHBVAL': '     0.00   0.32540   3; VAL HG22-CG2-CB-HBModified',
        'HG23CG2CBHBVAL': '     0.00   0.32540   3; VAL HG23-CG2-CB-HB Modified',
        'HG11CG1CBCG2VAL': '     0.00   0.36724   3; VAL HG11-CG1-CB-CG2 Modified',
        'HG12CG1CBCG2VAL': '     0.00   0.36724   3; VAL HG12-CG1-CB-CG2 Modified',
        'HG13CG1CBCG2VAL': '     0.00   0.36724   3; VAL HG13-CG1-CB-CG2 Modified',
        'HG21CG2CBCG1VAL': '     0.00   0.36724   3; VAL HG21-CG2-CB-CG1 Modified',
        'HG22CG2CBCG1VAL': '     0.00   0.36724   3; VAL HG22-CG2-CB-CG1 Modified',
        'HG23CG2CBCG1VAL': '     0.00   0.36724   3; VAL HG23-CG2-CB-CG1 Modified',
        'HB1CBCACALA': '     0.00   0.58120   3; ALA HB1-CB-CA-C Modified',
        'HB2CBCACALA': '     0.00   0.58120   3; ALA HB2-CB-CA-C Modified',
        'HB3CBCACALA': '     0.00   0.58120   3; ALA HB3-CB-CA-C Modified',
        'HB1CBCAHAALA': '     0.00   0.58120   3; ALA HB1-CB-CA-HA Modified',
        'HB2CBCAHAALA': '     0.00   0.58120   3; ALA HB2-CB-CA-HA Modified',
        'HB3CBCAHAALA': '     0.00   0.58120   3; ALA HB3-CB-CA-HA Modified',
        'HB1CBCANALA': '     0.00   0.58120   3; ALA HB1-CB-CA-N Modified',
        'HB2CBCANALA': '     0.00   0.58120   3; ALA HB2-CB-CA-N Modified',
        'HB3CBCANALA': '     0.00   0.58120   3; ALA HB3-CB-CA-N Modified',
        'HE1CESDCGMET': '     0.00   1.08087   3; MET HE1-CE-SD-CG Modified',
        'HE2CESDCGMET': '     0.00   1.08087   3; MET HE2-CE-SD-CG Modified',
        'HE3CESDCGMET': '     0.00   1.08087   3; MET HE3-CE-SD-CG Modified',

    }
    if ';' in line:
        entry, comm = line.split(';', 1)
    else:
        entry = line
    tokens = entry.split()
    if len(tokens) < 5:
        msg = 'Number of Columns in the dihedral section must be be at least 5. not {0}'.format(
            len(tokens))
        raise RtError(
            'ParseDihedral: on linenum:{0} errmsg:{1}'.format(
                n + 1, msg))
    else:
        ldih = [tokens[0], tokens[1], tokens[2], tokens[3]]
        ddih = GetDihedral(modatoms, ldih)
        dres = GetResidue(modatoms, ldih)
        ddihres = ddih + dres[3:6]
        if (ddihres not in DICT.keys()):
            return line
        else:
            s = '  ' + '   '.join(ldih)
            s += '     ' + tokens[4]
            s += DICT[ddihres] + '\n'
            return s


def GetDihedral(matoms, ldih):
    d = ''
    for j in ldih:
        i = [k for k in matoms if k[0] == j]
        if len(i) != 0:
            d += i[0][1]
    return d

def GetResidue(matoms, ldih):
    d = ''
    for j in ldih:
        i = [k for k in matoms if k[0] == j]
        if len(i) != 0:
            d += i[0][2]
    return d


def main():
    try:
        itpfile = sys.argv[1]
        modatoms = []
        section = None
        outcontents = topheader
        with open(itpfile, 'r') as finp:
            for i, line in enumerate(finp):
                sline = line.strip()
                if len(sline) == 0 or sline[0] in '#;':
                    outcontents += line
                elif sline[0] == '[':
                    try:
                        outcontents += line
                        section = line.split()[1]
                    except BaseException:
                        err = 'Unknown Error in parsing section'
                        msg = 'main: line num: {0} errmsg: {1}'.format(i, err)
                        raise RtError(msg)
                else:
                    if section == 'atoms':
                        outcontents += line
                        ParseAtom(line, i, modatoms)
                    elif section == 'dihedrals':
                        outcontents += ParseDihedral(line, i, modatoms)
                    else:
                        outcontents += line
        print(outcontents)
    except IndexError:
        print('M> usage: {0} protein.itp'.format(sys.argv[0]))
    except IOError:
        print('E> Error: File {0} not found'.format(itpfile))
    except BaseException:
        print('E> Error: Unexpected {0}'.format(sys.exc_info()))


if __name__ == '__main__':
    main()
