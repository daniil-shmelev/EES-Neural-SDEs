@echo off
python -m OU --method ees25
python -m OU --method reversible_heun
python -m OU --method mcf_midpoint
python -m OU --method mcf_euler
python -m OU --method ees27