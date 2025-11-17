@echo off
python -m sde_gan --method reversible_heun
python -m sde_gan --method ees25
python -m sde_gan --method mcf_midpoint
python -m sde_gan --method mcf_euler