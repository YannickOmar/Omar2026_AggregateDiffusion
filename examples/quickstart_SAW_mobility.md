# Quickstart: SAW → diffusion coefficients

## 1) Generate SAW realizations
Edit `src/GenSAW.py` near the top:
- set `N = 200` (small, fast)
- set `nSample = 5`

Run:
```bash
python src/GenSAW.py
- produces output in data/

## 2) Compute diffusion coefficients
python src/computeDiffusivity.py examples/example_input.txt
- creates output file in DiffOutput/  