# Membrane protein aggregate diffusion (code release)

This repository contains:
1) generators for lattice-based aggregates (SAW, DLA, LA, DLCA) that output particle coordinates, and  
2) a hydrodynamic post-processing script to compute mobility metrics for each realization using a membrane Rotne–Prager–Yamakawa-type tensor in Kirkwood–Riseman theory.

## Repository structure
- `src/GenSAW.py`, `src/SAWFunctions.py`: self-avoiding walk generator
- `src/GenDLA.py`: diffusion-limited aggregation generator
- `src/GenLA.py`: lattice-animal generator (requires graph-tool)
- `src/GenDLCA.py`: diffusion-limited cluster aggregation (requires graph-tool)
- `src/KR_RPY_utils.py`: hydrodynamic tensor, Kirkwood–Riseman solver, hull sampling approximation
- `src/computeDiffusivity.py`: batch computation of drag metrics from coordinate files

## Installation (conda recommended)
Create the environment:
```bash
conda env create -f membrane_AggDiff.yml
conda activate membrane_AddDiff

## output file format
Rows correspond to different realizations with columns
	1.	Rg          (radius of gyration)
	2.	RHs         (hydrodynamic radius, “small” )
	3.	RHl         (hydrodynamic radius, “large”)
	4.	Np          (# particles)
	5.	Xixx        normalized drag coefficient xx
	6.	Xixy        normalized drag coefficient xy
	7.	Xiyy        normalized drag coefficient yy
	8.	XixxApprox  hull-sampled approximation xx
	9.	XixyApprox  hull-sampled approximation xy
	10.	XiyyApprox hull-sampled approximation yy
