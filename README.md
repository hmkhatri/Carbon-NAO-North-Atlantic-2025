# Response of Dissolved Inorganic Carbon in the North Atlantic Ocean to the North Atlantic Oscillation

The repository contains scripts for analysing the changes in temperatures, dissolved ingorganic carbon (DIC) and nutrient concentrations in the North Atlantic Ocean due to a single North Atlantic Oscillation (NAO) event, see [Khatri and Williams (2025)](https://doi.org/10.22541/essoar.173884326.61031637/v1). The ocean response is decomposed into fast and slow responses. Furthermore, tracer budget and transport diagnostics are analysed to identify the dominant processes driving changes in temperatures and tracer concentrations.     

Python scripts for computing diagnsotics are in [Python-Scripts](./Python-Scripts) and plotting notebooks are in [Plots](./Plots). Below is a short description of different scripts.

| Script Name Structure | Content |
| --- | --- | 
| Compute_Tracer_timeries | To compute time-series of tracers |
| Compute_Tracer_Budget | To compute tracer budget (domain-integrated) for heat and carbon |
| Compute_Overturning_Transport | To compute meridional overturning and transports of heat, carbon and nutrients |
| Compute_Climatology | To compute climatology of certain diagnostics |
| Composites_ | To compute NAO-based composites of budget and transport diagnostics | 
| Annual_maps | To compute NAO-based annual maps of tracer composites |

A tutorial on creating NAO-based composites is avaiable in [notebok](https://github.com/hmkhatri/Ocean-Memory-Atlantic-GRL-2024/blob/main/Tutorials/NAO-SST-Composite.ipynb). The same approach also applies to other ocean diagnostics, see [Khatri et al. (2022)](https://doi.org/10.1029/2022GL101480).
