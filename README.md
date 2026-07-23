# Tunable many-body burst in isolated quantum systems

This repository contains the Julia code used for the numerical analysis in:

> S. Yamada, A. Hokkyo, and M. Ueda, *Tunable many-body burst in isolated quantum systems*  
> Paper: https://arxiv.org/abs/2602.09665

## Environment

- Julia: **1.11.6**
- Package versions:
  - ITensors: **0.9.9**
  - ITensorMPS: **0.3.20**
  - JLD2: **0.6.0**
  - ProgressMeter: **1.11.0**

## Usage

Parameters are set directly in the configuration section of each script. Use `Magz` for the main-text magnetization results and `Magy` for Fig. 6. In the code, the field coefficients are entered as `hx = 0.9045/2` and `hz = h_z/2`, consistently with the factor `1/2` in the Hamiltonian used in the paper.

Create the cache directories before running the finite-system calculations:

```bash
mkdir -p submit_UOU submit_hz_UOU
```

### 1. Generate the time-evolved observable `O(\tau)`

Edit the system size `L`, longitudinal field `hz`, observable, and time-evolution parameters in:

```bash
julia burst_UOU.jl
```

Run the script for every required combination of `L` and `hz`. Store the generated JLD2 cache files in the locations expected by the downstream scripts:

- `submit_UOU/`: fixed-`h_z` calculations for different `L`
- `submit_hz_UOU/`: fixed-`L` calculations for different `h_z`

### 2. Burst dynamics and quantum-circuit approximation

```bash
julia burst_plot_QC.jl
```

This script computes the data used for:

- Figs. 1 and 6(a): time evolution starting from the optimized initial state
- Fig. 3(b): time evolution starting from the state that approximates the optimized state obtained by a shallow quantum circuit

### 3. System-size and burst-time dependence

```bash
julia burst_Ltau_UOU.jl
```

This script computes the data used for:

- Figs. 2(a) and 6(b): dependence on system size `L` and burst time `\tau`
- Fig. 4: correlation length of the optimized initial state

### 4. Field and burst-time dependence

```bash
julia burst_hztau_UOU.jl
```

This script computes the data used for:

- Figs. 2(b) and 6(c): dependence on longitudinal field `h_z` and burst time `\tau`


### 5. Thermodynamic-limit calculation

```bash
julia burst_Linf.jl
```

This script computes the infinite-system `\tau`-dependence used for Fig. 5.


## Contact

**yamada@cat.phys.s.u-tokyo.ac.jp**
