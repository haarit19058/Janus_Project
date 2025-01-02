This repository has three folders:

1. **mycode**  
   This folder contains the implementation of Janus. The oscillator strength of each molecule was calculated using PySCF. While accurate, this method is very slow because PySCF takes a lot of time to compute S1, T1, and oscillator strength.

2. **Transformer**  
   This folder includes a transformer model trained on SMILES representations of molecules to estimate their properties. This method was much faster than the first one and produced more useful molecules.

3. **Transformer 2.0**  
   In this folder, molecules were filtered based on their S1 and T1 values, focusing on organic emitters. The transformer was trained on this smaller set of molecules, and Janus was applied to it. This approach worked better than the earlier ones and gave promising results.

The repository also includes a report about the project.

### Library Requirements
- `selfies` (v1.0.3)  
- `torch`  
- `transformers`  
- `pyscf`  
- `pandas`  
- `numpy`  
- `janus-ga`