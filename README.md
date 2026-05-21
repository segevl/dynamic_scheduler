# dynamic_scheduler
This repository contains the complete source code for the thesis "Cloudy with a Chance of Data: A Dynamic Scheduler for All‑Sky Surveys with Real‑Time Optimization" (Larom Segev, Harvard University, April 2026).

The system uses DREAM all‑sky camera data to predict cloud motion in real time and optimizes telescope pointing for the Legacy Survey of Space and Time (LSST). Three pointing strategies are compared:

| Strategy | Description |
|----------|-------------|
| DREAM Absolute | Slew to the instantaneous minimum cloud extinction |
| DREAM Motion   | Track the motion‑predicted clear patch using cross‑correlation |
| Greedy Scheduler | Baseline Rubin scheduler (OpSim) that assumes a clear sky |

All three are evaluated on real DREAM cloud data using slew‑gated photon collection and OpSim‑consistent 5σ depth. For more information on the structure of the repository and what each notebook or python file contains, see the respository guide.
