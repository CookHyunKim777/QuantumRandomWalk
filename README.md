# Universal scaling and protocol-dependent amplitude in hybrid quantum walks

This repository contains the simulation and analysis code for two quantum--classical hybrid walks on a one-dimensional ring:

- **QRW-A (additive protocol):** at every time step, the density matrix is updated by the convex combination
  \(\rho_{t+1}=\alpha\mathcal{U}[\rho_t]+(1-\alpha)\mathcal{T}[\rho_t]\).
- **QRW-M (multiplicative protocol):** each cycle applies `n_qw` quantum steps followed by one classical step.

The code records the mean squared displacement (MSD), total variation distance (TVD), final probability distributions, and selected probability snapshots. The accompanying notebook extracts the diffusion coefficient and evaluates the analytical QRW-A truncations and exact QRW-M recurrence used in the manuscript.

## Repository contents

| File | Purpose |
| --- | --- |
| `run_qrw.py` | Pure-Python command-line runner |
| `run_qrw_c.py` | Command-line runner with the C backend and Python fallback |
| `qrw_utils.py` | Graph construction, walk maps, observables, and Python reference implementation |
| `qrw_core.c`, `qrw_core.h` | C simulation core |
| `qrw_c_bridge.py` | `ctypes` bridge between Python and the C core |
| `submit_jobs.ipynb` | Generation and submission of the QRW-A and QRW-M SLURM parameter sweeps |
| `submit_qrw_a.sh`, `submit_qrw_m.sh` | SLURM array scripts |
| `diffusion_coefficient.ipynb` | Numerical extraction and analytical diffusion coefficients |

The simulations reported in the manuscript use a ring with `N=2501`, a Hadamard coin, and `T=50000`. Additional graph constructors remain available in `qrw_utils.py` for extensions.

## Installation

Python 3.10 or later and a C compiler are recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
make
```

`make` builds `libqrw.so` on Linux or `libqrw.dylib` on macOS. The compiled library is intentionally not tracked by Git.

## Quick start

Run QRW-A on a small ring:

```bash
python run_qrw_c.py \
  --graph_type cycle --graph_size 101 --coin_type H \
  --walk_type QRW_A --alpha 0.90 \
  --time_length 1000 --save_interval 100 \
  --output_dir results/qrw_a
```

Run QRW-M with nine quantum steps followed by one classical step:

```bash
python run_qrw_c.py \
  --graph_type cycle --graph_size 101 --coin_type H \
  --walk_type QRW_M --n_qw 9 --n_rw 1 \
  --time_length 1000 --save_interval 100 \
  --output_dir results/qrw_m
```

Force the Python reference backend by adding `--use_python`. To compare the C and Python implementations on small test cases, run:

```bash
make verify
```

## Reproducing the manuscript calculations

1. Start Jupyter from the repository root.

   ```bash
   jupyter notebook
   ```

2. Open `submit_jobs.ipynb`. It generates the manuscript sweeps:

   - QRW-M: `n_qw = 0, ..., 19` and `n_rw = 1`;
   - QRW-A: `alpha = 0.00, 0.05, ..., 0.95`.

   The notebook creates `params/`, `results/`, and `logs/` under the repository root. Review the requested SLURM time before executing the submission cells.

3. After the simulations finish, open `diffusion_coefficient.ipynb`. The numerical procedure divides the MSD time series into consecutive 200-step blocks, retains blocks satisfying \(|\beta-1|<0.05\), and reports a diffusion coefficient only when at least three blocks are retained.

4. The same notebook evaluates the exact Laurent-polynomial recurrence for QRW-M and the `L=1` and `L=2` transfer-operator truncations for QRW-A.

## Output format

Each run writes a compressed NumPy archive (`.npz`). Important fields include:

- `msd_list`, `tvd_list`;
- `p_final`, `p_avg_final`;
- `p_snapshots`, `snapshot_times`;
- graph, protocol, and run metadata.

Load a result with:

```python
import numpy as np

data = np.load("results/qrw_a/example.npz")
msd = data["msd_list"]
```

## Computational note

The implementation propagates a dense density matrix. Its memory use therefore grows quadratically with the number of directed arcs. The manuscript-scale calculations are intended for a high-memory compute node. In QRW-M, the current implementation records complete protocol cycles; when `time_length` is not divisible by `n_qw + n_rw`, the final incomplete cycle is omitted.

## Citation

If you use this code, please cite:

Cook Hyun Kim, Jaesub Park, and Namshik Han, *Universal scaling and protocol-dependent amplitude in hybrid quantum walks*.

Publication metadata can be added to `CITATION.cff` after the article is published.
