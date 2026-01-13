# Code for the publication  
[Efficient Magic State Cultivation on the Surface Code](https://arxiv.org/abs/2502.01743)  
Yotam Vaknin, Shoham Jacoby, Arne Grimsmo, Alex Retzker

---

# Instructions

## Setup
Install the Python dependencies listed in `requirements.txt`.

## Where the core results live
The core results are in:

- `2025/HCultivationSurfaceCode`

## Reproducing the main CSV output - Clifford Simulations
Run the following three scripts, in order, to:
1) generate the circuits,
2) sample them, and
3) combine the samples into a single CSV called `stats_combined.csv`.

- `step1_generate_circuits.bash`
- `step2_sample.bash`
- `step3_combine.bash`

A full version of `stats_combined.csv` is already included in the repository.

## Full Vector Simulation
Full vector simulations are done using `sample.py` script. Our samples are included in `combined_vec_sim.csv`

## Plots
Plots are generated in `plots.ipynb`.

---

## Acknowledgments / External Code

This repo includes copies and adaptations of code from other sources:

- [Magic state cultivation: growing T states as cheap as CNOT gates](https://zenodo.org/records/13777072)  
- [Efficient Magic State Cultivation on RP2](https://github.com/Zihan-Chen-PhMA/Cultiv_T_RP2)  