# Efficient Magic State Cultivation on the Surface Code

This repository contains the code and data for the paper:

**[Efficient Magic State Cultivation on the Surface Code](https://arxiv.org/abs/2502.01743)**  
Yotam Vaknin, Shoham Jacoby, Arne Grimsmo, Alex Retzker

---

## Overview

This repository provides implementations and simulation results for magic state cultivation on the surface code. The codebase includes:

- Circuit generation scripts for surface code magic state cultivation protocols
- Clifford simulation tools using Stim and Sinter
- Full state vector simulation capabilities
- Analysis and plotting notebooks for reproducing figures from the paper

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{vaknin2025efficient,
  title={Efficient Magic State Cultivation on the Surface Code},
  author={Vaknin, Yotam and Jacoby, Shoham and Grimsmo, Arne and Retzker, Alex},
  journal={arXiv preprint arXiv:2502.01743},
  year={2025}
}
```

---

## Setup

### Requirements

- Python 3.7 or higher
- All Python dependencies are listed in `requirements.txt`

### Installation

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

---

## Repository Structure

The core results and code are located in:

- **`2025/HCultivationSurfaceCode/`** - Main codebase containing:
  - Circuit generation scripts
  - Simulation and sampling tools
  - Pre-computed results (CSV files)
  - Plotting notebooks

Main file:
- `simulation.py` - Generates the different circuits. The Simulation object represent a single surface code, with class methods such as:
   - `generate_double_ghz'
   - `generate_injection_circuit`
   - `generate_syndrome_circuit`
   generate the different steps of our protocol in using our Circuit object. The final expansion step is described in `stim_expansion_circuit`, only 
   defined for stim since it only occurs on the Clifford simulations. 


Key directories:
- `circuits/` - Generated Stim circuit files
- `fig/` - Output figures
- `sampler/` - Code for the two samplers we use, either full post-selection or based on gap-decoding. 
- `cultiv_code/` - Cultivation code from Gidney et al. with some changes

---

## Reproducing Results

### Main Results: Clifford Simulations

The main simulation results use Clifford tableau simulation via Stim and Sinter. To reproduce the results:

1. **Generate circuits**: Creates all Stim circuit files needed for simulation
   ```bash
   cd 2025/HCultivationSurfaceCode
   bash step1_generate_circuits.bash
   ```

2. **Sample circuits**: Runs Monte Carlo sampling to estimate error rates
   ```bash
   bash step2_sample.bash
   ```

3. **Combine results**: Aggregates all samples into a single CSV file
   ```bash
   bash step3_combine.bash
   ```

The final output is `stats_combined.csv`, which contains the main simulation results. A pre-computed version of this file is already included in the repository.

**Note**: The sampling step (step 2) can take significant computational time depending on the number of shots and circuit sizes.

### Full State Vector Simulation

For exact state vector simulations (used for smaller circuits):

1. Run the sampling script:
   ```bash
   python sample.py
   ```

2. Combine the results:
   ```bash
   python combine_state_vec_sim.py
   ```

This produces `combined_vec_sim.csv`. A pre-computed version is included in the repository. The computation-time requires to reproduce this step is over a million CPU hours.

### Generating Figures

All plots and figures from the paper can be reproduced using the Jupyter notebook:

```bash
jupyter notebook plots.ipynb
```

---

## Pre-computed Results

The following pre-computed result files are included in the repository:

- `stats_combined.csv` - Main Clifford simulation results
- `combined_vec_sim.csv` - Full state vector simulation results
- Additional CSV files in `2025/HCultivationSurfaceCode/` for various circuit configurations

These files allow for immediate analysis and plotting without running the computationally intensive sampling steps.

---

## Acknowledgments

This repository includes code adapted from the following sources:

- [Magic state cultivation: growing T states as cheap as CNOT gates](https://zenodo.org/records/13777072) (Gidney et al.)
- [Efficient Magic State Cultivation on RP2](https://github.com/Zihan-Chen-PhMA/Cultiv_T_RP2) (Chen et al.)

We thank the authors of these works for making their code publicly available.
