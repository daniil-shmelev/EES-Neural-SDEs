@echo off
python -m GBM_2 --method reversible_heun
python -m GBM_2 --method ees25
python -m GBM_2 --method mcf_midpoint
python -m GBM_2 --method mcf_euler
python -m GBM_2 --method ees27