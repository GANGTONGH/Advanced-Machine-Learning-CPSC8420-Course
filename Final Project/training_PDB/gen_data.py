#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 20:50:37 2022

@author: gh
"""
# Transform 3D protein structures into Triplet CA distances

import Bio
import os
import numpy as np
import numpy as numpy
from Bio.PDB import *
from os.path import exists
from Bio.PDB.DSSP import DSSP

os.chdir(os.path.dirname(__file__))

# List of 20 standard AAs
protAA = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", \
"MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", \
"TRP", "TYR"]

d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

daaorder = {'A' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F' : 5, 'G' : 6, 'H' : 7, 'I' : 8, 'K' : 9, 'L' : 10, 'M' : 11, 'N' : 12, 'P' : 13, 'Q' : 14, 'R' : 15, 'S' : 16, 'T' : 17, 'V' : 18, 'W' : 19, 'Y' : 20}

statis = np.zeros([36, 36, 20])

cullPDB_fn = "/Volumes/PortableSSD/!Advanced_Machine_Learning/Final_project/training_3/cullpdb_pc25.0_res0.0-2.0_noBrks_len40-10000_R0.3_Xray+EM_d2022_04_25_chains6968"

with open(cullPDB_fn, 'r') as cullPDB:
    
    pdb_chain_list = [line.split()[0] for line in cullPDB.readlines()[1:]]
    pdb_list = [ s[0:4].lower() for s in pdb_chain_list ]
    chain_list = [ s[4:] for s in pdb_chain_list ]
    
    cullPDB.close()
    
raw_out_fn = 'CA_NN.dat'

def chain_length(residues):
    res_no = 0
    for r in residues:
        if r.id[0] == ' ':
            res_no += 1
    return res_no

# get atom ignoring insertion code
def get_res(chain, id):
    if chain.has_id((' ', id, ' ')):
        return chain[(' ', id, ' ')]
    elif chain.has_id((' ', id, 'A')):
        return chain[(' ', id, 'A')]
    elif chain.has_id((' ', id, 'B')):
        return chain[(' ', id, 'B')]

if len(pdb_list) == len(chain_list):
    
    try:
        '''
        Print raw data to a file
        '''
        with open(raw_out_fn, 'w+') as out_file:
            # for i in range(len(pdb_list)):
            for i in [1075]:
                print(pdb_list[i])
                
                # PDB file must exist
                if not exists('pdb' + pdb_list[i] + '.ent'): continue
            
                parser = PDBParser(PERMISSIVE=True)
                structure = parser.get_structure(pdb_list[i], 'pdb' + pdb_list[i] + '.ent')
                
                chain = structure[0][chain_list[i]]
                
                chain.atom_to_internal_coordinates()
                
                residues = chain.get_residues()
                l_chain = chain_length(residues)
                
                print(i)
                
                for i in range(1, l_chain - 2, 1):
                    
                    # Residue must exist and non-disorder
                    if get_res(chain, i) == None or \
                       get_res(chain, i+1)== None or \
                       get_res(chain, i+2) == None or \
                       get_res(chain, i).is_disordered() or \
                       get_res(chain, i+1).is_disordered() or \
                        get_res(chain, i+2).is_disordered() :
                        continue
                    
                    p = get_res(chain, i)
                    c = get_res(chain, i+1)
                    n = get_res(chain, i+2)
                     
                    # Must be the 20 standard amino acids
                    if p.get_resname() in protAA and c.get_resname() in protAA and n.get_resname() in protAA:
                        
                        if p.has_id('CA') and n.has_id('CA'):
                            CA_p = p['CA']
                            CA_n = n['CA']
                            # coord_p = CA_p.get_vector()
                            # CA_c = c['CA'].get_vector()
                            # coord_n = CA_n.get_vector()
                            dist = CA_p - CA_n
                            
                            print(d3to1[p.get_resname()], d3to1[c.get_resname()], d3to1[n.get_resname()], dist, file = out_file)
                    
            out_file.close()
            
    except Exception:
        pass
  
else:
    print("Error")
# print(statis)
