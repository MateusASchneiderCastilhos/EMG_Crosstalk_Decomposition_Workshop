# Information about the decomposed files:

The parameters used in the decomposition that generated these files are:

- Number of Iterations: $M = 300$
- Extension Factor: $R = 20$
- Internal Loop Iterations: $40$
- FastICA Convergence: $10^{-4}$
- SIL Threshold: $90%$
- Contrast Function: $g(x) = x^2$
- Initializations of $w_j$: $Maximum$

There are two main files for each decomposition, the `...Decomposition_Results.mat` and `...Pulse_Trains_Metrics.csv`, where the three dots `...` indicates the name of the original HD sEMG file.

-The `...Decomposition_Results.mat` files are save in structured format that allows our MatLab functions (files are not in this repository) read them properly.
-The `...Pulse_Trains_Metrics.csv` files are structured as tables with columns being the quality and statistical metrics, and the columns been the motor units.