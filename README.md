# Tunable many-body burst in isolated quantum systems

This repository contains the Julia code used for the numerical analysis in:

> S. Yamada, A. Hokkyo, and M. Ueda, *Tunable many-body burst in isolated quantum systems*  
> Paper: https://arxiv.org/abs/2602.09665

## Environment

- Julia: **1.11.6**
- Operating system: **[TODO: OS and version]**
- Package versions:
  - ITensors: **[TODO]**
  - ITensorMPS: **[TODO]**
  - JLD2: **[TODO]**
  - ProgressMeter: **[TODO]**
- Hardware used for the reported calculations: **[TODO: CPU/GPU, memory, and number of threads]**
- Approximate runtime and memory requirement: **[TODO]**
- Code version corresponding to the paper: **[TODO: Git commit, tag, or release DOI]**

A `Project.toml` and `Manifest.toml` should be used to reproduce the package environment:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Usage

Parameters are set directly in the configuration section of each script. Use `Magz` for the main-text magnetization results and `Magy` for Fig. 6. In the code, the field coefficients are entered as `hx = 0.9045/2` and `hz = h_z/2`, consistently with the factor `1/2` in the Hamiltonian used in the paper.

Create the cache directories before running the finite-system calculations:

```bash
mkdir -p submit_UOU submit_hz_UOU
```

### 1. Generate the time-evolved observable \(O(\tau)\)

Edit the system size `L`, longitudinal field `hz`, observable, and time-evolution parameters in:

```bash
julia burst_UOU.jl
```

Run the script for every required combination of `L` and `hz`. Store the generated JLD2 cache files in the locations expected by the downstream scripts:

- `submit_UOU/`: fixed-\(h_z\) calculations for different `L`
- `submit_hz_UOU/`: fixed-`L` calculations for different \(h_z\)

Exact parameter sets and cache-file naming rules: **[TODO]**

### 2. Burst dynamics and quantum-circuit approximation

```bash
julia burst_plot_QC.jl
```

This script computes the data used for:

- Fig. 1
- Fig. 3(b)
- Fig. 6(a)

Parameters for each figure and output filenames: **[TODO]**

### 3. System-size and burst-time dependence

```bash
julia burst_Ltau_UOU.jl
```

This script computes the data used for:

- Figs. 2(a) and 6(b): dependence on system size \(L\) and burst time \(\tau\)
- Fig. 4: correlation length of the optimized initial state

Parameters for each figure and output filenames: **[TODO]**

### 4. Field and burst-time dependence

```bash
julia burst_hztau_UOU.jl
```

This script computes the data used for:

- Figs. 2(b) and 6(c): dependence on longitudinal field \(h_z\) and burst time \(\tau\)

Parameters for each figure and output filenames: **[TODO]**

### 5. Thermodynamic-limit calculation

```bash
julia burst_Linf.jl
```

This script computes the infinite-system \(\tau\)-dependence used for Fig. 5.

Set `CALC_MODE = :z` or `:y` as needed. Parameters and output filenames: **[TODO]**

## Outputs and plotting

The scripts save numerical results primarily as JLD2 files. The following information should be added for complete reproduction:

- Location of the numerical data used in the published figures: **[TODO]**
- Commands or scripts used to generate the final figure files: **[TODO]**
- Expected numerical values or checksums for a small reference run: **[TODO]**
- Treatment of random seeds and the number of DMRG trials: **[TODO]**
- Convergence criteria and accepted truncation-error thresholds: **[TODO]**

## Citation

```bibtex
[TODO: BibTeX entry]
```

## License

**[TODO: license]**

## Contact

**[TODO: contact name and email]**
