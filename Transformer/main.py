import pandas as pd
import torch
from janus import JANUS, utils
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RDConfig, Descriptors
RDLogger.DisableLog("rdApp.*")


data = pd.read_csv('data.csv')
print(data)
initSmiles = data['SMILES']





from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# SMILES string for prediction (example: use your own SMILES string)
smiles_string = "CCO"  # Replace with the desired SMILES string

# Tokenize the input
inputs = tokenizer(smiles_string, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

# Make prediction
with torch.no_grad():
    output = model(**inputs)
    prediction = output.logits.squeeze().cpu().numpy()

# Print the prediction
print(f"Predicted Oscillator Strength for '{smiles_string}': {prediction}")



def fitness_function(smi: str) -> float:
    """User-defined function that takes in individual SMILES and outputs a fitness value."""

    inputs = tokenizer(smi, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(**inputs)
        prediction = output.logits.squeeze().cpu().numpy()

    if prediction < 0.7:
        return -10

    return prediction





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
#         f.write(f"{row['SMILES']}\t{float(row['s1_ref'])}\t{float(row['t1_ref'])}\n")

# Reading previously successful calculations and storing them in the dp dictionary




# all parameters to be set, below are defaults
params_dict = {
    # Number of iterations that JANUS runs for
    "generations": 5,

    # The number of molecules for which fitness calculations are done,
    # exploration and exploitation each have their own population
    "generation_size": 500,

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
    fitness_function = fitness_function,                    # user-defined fitness for given SMILES
    start_population = "./SMILES.txt",   # file with starting SMILES population
    **params_dict
)

# Alternatively, you can get hyperparameters from a yaml file
# Descriptions for all parameters are found in default_params.yml
params_dict = utils.from_yaml(
    work_dir = 'RESULTS',
    fitness_function = fitness_function,
    start_population = "./SMILES.txt",
    yaml_file = 'default_params.yml',       # default yaml file with parameters
    **params_dict                           # overwrite yaml parameters with dictionary
)
print("Will start Janus now")
agent = JANUS(**params_dict)

# Run according to parameters
agent.run()     # RUN IT!

print("Janus running complete")





