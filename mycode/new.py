import pandas as pd
import torch
from janus import JANUS, utils
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RDConfig, Descriptors
RDLogger.DisableLog("rdApp.*")

import torch
import selfies
import numpy as np
import pyscf


data = pd.read_csv('cleancsv.csv')[:500]
data = data.iloc[:,-5:]
initSmiles = data['smiles']

with open('smiles.txt','w') as f:
    for  i in initSmiles:
        f.write(i.strip())
        f.write('\n')


from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, scf, tdscf
import pyscf.lib

# Enable GPU support
# pyscf.lib.param.TMPLIBDIR = "/path/to/cuda"  # Set this to your CUDA library path

def get_s1_t1_energies(smiles):
    # Convert SMILES to PySCF geometry
    def smiles_to_pyscf_geom(smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)  # Add hydrogens
        AllChem.EmbedMolecule(mol, randomSeed=42)  # Generate 3D structure
        AllChem.UFFOptimizeMolecule(mol)  # Optimize geometry using UFF
        atoms = mol.GetAtoms()
        coords = mol.GetConformers()[0].GetPositions()

        atom_str = []
        for atom, coord in zip(atoms, coords):
            atom_str.append(f"{atom.GetSymbol()} {coord[0]} {coord[1]} {coord[2]}")
        return "\n".join(atom_str)

    # Generate geometry
    mol_geom = smiles_to_pyscf_geom(smiles)

    # PySCF Molecule object for singlet ground state (restricted HF)
    mol = gto.M(atom=mol_geom, basis='cc-pVDZ', spin=0, charge=0)

    # Perform restricted ground state calculation (for S1)
    mf = scf.RHF(mol).density_fit().run()

    # TDDFT for singlet excited states (S1)
    td_singlet = tdscf.TDHF(mf).run(nstates=1)
    s1_energy = td_singlet.e[0]  # Singlet excited state energy

    # PySCF Molecule object for triplet state (unrestricted HF)
    mol_triplet = gto.M(atom=mol_geom, basis='cc-pVDZ', spin=2, charge=0)  # Spin multiplicity of 3 (2S+1)

    # Perform unrestricted ground state calculation (for T1)
    mf_triplet = scf.UHF(mol_triplet).density_fit().run()

    # TDDFT for triplet excited states (T1)
    td_triplet = tdscf.TDHF(mf_triplet).run(nstates=1)
    t1_energy = td_triplet.e[0]  # Triplet excited state energy

    return s1_energy, t1_energy

print("Function to calculate s1 t1 is defined")
# Example usage
# smiles = 'N#CC1=C(N)C=C1'  # Input molecule SMILES
# s1_energy, t1_energy = get_s1_t1_energies(smiles)

# print(f"S1 energy: {s1_energy} Hartree")
# print(f"T1 energy: {t1_energy} Hartree")


dp = dict()



def calc(s1, t1):
    return 1 / (1e-10 + abs(float(s1) - float(t1)))


def fitness_function(smi: str) -> float:
    """User-defined function that takes in individual SMILES and outputs a fitness value."""
    try:
        if smi in dp.keys():
            return dp[smi]

        s1, t1 = get_s1_t1_energies(smi)
        
        dp[smi] = calc(s1, t1)
        
        with open('successful.txt', 'a') as f:
            f.write(f"{smi}\t{s1}\t{t1}\n")

        return dp[smi]
    
    except Exception as e:
        with open("errors.txt", "a") as f:
            f.write(f"{smi}\t{e}\n")
        print(e)
        return -1

def custom_filter(smi: str):
    """Function that takes in a SMILES string and returns a boolean indicating if it passes the filter."""
    # Filter based on the length of SMILES
    if len(smi) > 81 or len(smi) == 0:
        return False
    else:
        return True


print("fitness defined")

# Writing newly filtered data to 'successful.txt'
# with open('successful.txt', 'a') as f:
#     for i in data.iterrows():
#         row = i[1]
#         f.write(f"{row['smiles']}\t{float(row['s1_ref'])}\t{float(row['t1_ref'])}\n")

# Reading previously successful calculations and storing them in the dp dictionary
with open('successful.txt', 'r') as f:
    line = f.readline()
    while line:
        smile, s1, t1 = line.split()
        dp[smile] = calc(float(s1), float(t1))  # Ensure that s1 and t1 are treated as floats
        line = f.readline()


# freeze support issue
torch.multiprocessing.freeze_support()

# all parameters to be set, below are defaults
params_dict = {
    # Number of iterations that JANUS runs for
    "generations": 5,

    # The number of molecules for which fitness calculations are done,
    # exploration and exploitation each have their own population
    "generation_size": 100,

    # Number of molecules that are exchanged between the exploration and exploitation
    "num_exchanges": 10,

    # Callable filtering function (None defaults to no filtering)
    "custom_filter": custom_filter,

    # Fragments from starting population used to extend alphabet for mutations
    "use_fragments": True,

    # An option to use a classifier as selection bias
    "use_classifier": True,
}

# Set your SELFIES constraints (below used for manuscript)
# default_constraints = selfies.get_semantic_constraints()
# new_constraints = default_constraints
# new_constraints['S'] = 2
# new_constraints['P'] = 3
# selfies.set_semantic_constraints(new_constraints)  # update constraints


# Create JANUS object.
agent = JANUS(
    work_dir = 'RESULTS',                                   # where the results are saved
    fitness_function = fitness_function,                    # user-defined fitness for given smiles
    start_population = "./smiles.txt",   # file with starting smiles population
    **params_dict
)

# Alternatively, you can get hyperparameters from a yaml file
# Descriptions for all parameters are found in default_params.yml
params_dict = utils.from_yaml(
    work_dir = 'RESULTS',
    fitness_function = fitness_function,
    start_population = "./smiles.txt",
    yaml_file = 'default_params.yml',       # default yaml file with parameters
    **params_dict                           # overwrite yaml parameters with dictionary
)
print("Will start Janus now")
agent = JANUS(**params_dict)

# Run according to parameters
agent.run()     # RUN IT!

print("Janus running complete")
