# Linear Switched Systems

## Linear Parameter Varying Systems : 

**Linear Parameter-Varying (LPV) systems** are models used to represent systems whose dynamics depend on parameters that change over time, called scheduling signals. LPV models extend linear time-invariant systems by allowing these parameters to vary, enabling accurate modeling of nonlinear or time-varying processes while preserving a linear structure for analysis and identification. They are widely used in control and system identification due to their flexibility and practical relevance.

Under the right technical assumptions, any LPV can be decomposed into a
noiseless **deterministic linear parameter-varying (dLPV) system** which is driven only by the control input, and a **autonomous linear paramter-varying (asLPV)** system driven only by the noise. Moreover, the identification of these two subsystems can be carried out separately.

## Minimization of asLPV 
In this part of the project, we define what it means for an LPV (Linear Parameter-Varying) system to have a minimal state-space representation in innovation form. We present algebraic conditions to verify whether a asLPV state-space representation is minimal in forward innovation form and provide an algorithm to transform any stochastic asLPV model into a minimal innovation form.

To model these systems, we provide classes such as asLPV, dLPV, and LPV, which encapsulate the structure and simulation of these models. A main script is included to run simulations on user defined system matrices, allowing you to directly test and visualize your LPV models.

*N.B* : To execute the following scripts, position yourself in the root directory `Linear-Switched-Systems` where the makefile is located.

To run this script, first customize your matrices in `src/main/main.py`. Then, execute it using the following command:
```sh
make run
```
Additionally, an automatic main script that generates system matrices in innovation form, minimizes them without requiring manual parameter setup.
```sh
make auto_run
```
### Exemples & Graphs
Several examples are also provided and visualized using graphs. To generate and view these graphs, simply run: 
```sh
make graph
```

## Minimal covariance realisation :
In what follows, we present a Ho-Kalman-like algorithm for computing a minimal aLSS
realization innovation form from My,u. To this end, we first recall the Ho-Kalman realization algorithm for dLSS. LSS meaning Linear Switched Systems, a model class that generalizes LPVs and captures nonlinear behavior while preserving a linear input-output relationship. This structure makes LSS suitable for modeling certain nonlinear systems and applying classical linear control techniques. 

Although efficient control methods exist for some LSS subclasses, identifying such systems, particularly in the presence of stochastic noise, remains challenging. The code takes a first step by addressing the realization theory for stochastic LSS. 

HoKalmanIdentifier is a class for identifying Linear Parameter-Varying (LPV) systems using Ho-Kalman and True Ho-Kalman algorithms.
It automates the computation of sub-Hankel matrices and the extraction of state-space representations from data, particularly for dLPV systems in realization theory.

Consquently, we use these functions together to make the primary algorithm **'Minimal Covariance Realization Algorithm'** that builds a minimal state-space model from measured covariances between input and output signals.

### Tools & Requirements
- [![numpy](https://img.shields.io/badge/numpy-%3E=1.24-blue)](https://numpy.org/)
- [![scipy](https://img.shields.io/badge/scipy-%3E=1.10-blue)](https://scipy.org/)
- [![plotext](https://img.shields.io/badge/plotext-latest-lightgrey)](https://github.com/piccolomo/plotext)



