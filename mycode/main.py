import pandas as pd
import torch
from janus import JANUS, utils
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RDConfig, Descriptors
RDLogger.DisableLog("rdApp.*")

import torch
import selfies

data = pd.read_csv('cleancsv.csv')[:200]
data = data.iloc[:,-5:]
initSmiles = data['smiles']

with open('smiles.txt','w') as f:
    for  i in initSmiles:
        f.write(i.strip())
        f.write('\n')

from pyscf import gto, scf, dft, tdscf
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_geometry(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.UFFOptimizeMolecule(mol)
    atoms = mol.GetAtoms()
    positions = mol.GetConformer().GetPositions()
    
    atom_symbols = [atom.GetSymbol() for atom in atoms]
    coords = [(atom_symbols[i], tuple(positions[i])) for i in range(len(atom_symbols))]
    
    return coords

def pyscf_oscillator_strength(coords):
    # Define the molecule
    mol = gto.M(
        atom=coords,               # geometry of the molecule
        basis='6-31G',             # basis set
        verbose=0,
        symmetry=True
    )
    
    # Perform the DFT calculation
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    # Time-dependent DFT (TD-DFT) calculation
    td = tdscf.TDA(mf)
    td.nstates = 1  # Number of excited states to calculate
    td.kernel()

    return td.oscillator_strength()[0]

# Convert SMILES to geometry
smiles = "CC"  # Ethane
coords = smiles_to_geometry(smiles)

# Perform oscillator strength calculation
pyscf_oscillator_strength(coords)

dp = dict()


def fitness_function(smi: str) -> float:
    """ User-defined function that takes in individual smiles
    and outputs a fitness value.
    """

    try:

        match = data[data['smiles'] == smi]
        if not match.empty:
            return float(match['oscillator_strength_ref'])

        if smi in dp.keys():
            return dp[smi]

        coords = smiles_to_geometry(smi)
        osci = pyscf_oscillator_strength(coords)
        print(f"Calculated oscillator strength of {smi} is {osci}")

        dp[smi] = osci


        return osci

    except Exception as e:
        with open("errors.txt","a") as f:
            f.write(f"{smi}\t{e}\n")
        print(e)
        return -1

def custom_filter(smi: str):
    """ Function that takes in a smile and returns a boolean.
    True indicates the smiles PASSES the filter.
    """
    # smiles length filter
    if len(smi) > 81 or len(smi) == 0:
        return False
    else:
        return True

torch.multiprocessing.freeze_support()

# all parameters to be set, below are defaults
params_dict = {
    # Number of iterations that JANUS runs for
    "generations": 3,

    # The number of molecules for which fitness calculations are done,
    # exploration and exploitation each have their own population
    "generation_size": 20,

    # Number of molecules that are exchanged between the exploration and exploitation
    "num_exchanges": 1,

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
agent = JANUS(**params_dict)

# Run according to parameters
agent.run()     # RUN IT!


