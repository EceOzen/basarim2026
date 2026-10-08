# A Scalability Study of PCA and Neural Representations for ERA5 Compression

Code and configuration files for the BAŞARIM 2026 paper:

> Ece (Özen) İldem, *A Scalability Study of PCA and Neural Representations for ERA5 Compression*, BAŞARIM 2026, Istanbul, Türkiye.

The paper compares the computational cost of three compressors, **PCA**, **SIREN**, and a **multiresolution hash-grid** encoder, on ERA5 2 m temperature data. It covers encode/decode time, peak memory, and how each scales as the spatial domain grows on a single fixed device. Reconstruction quality is also measured (RMSE, normalized RMSE, maximum absolute error, and ClimateBenchPress error bounds).

## Main findings

- **SIREN.** Per-step cost is roughly constant (≈1.8 ms) across domain sizes. Total cost grows only with the epoch budget needed for adequate training coverage.
- **Hash-grid.** Per-step cost grows linearly with the number of levels `L` (435 CUDA kernel launches per level per step) and does not depend on table size. The GPU is busy for only 22–27% of each step, which makes this unfused PyTorch implementation launch/latency-bound.
- **PCA.** Inside a container, OpenBLAS detects the host's 48 cores but the pod has only 6 vCPUs. The resulting thread oversubscription slows PCA by 20–115×. Setting the BLAS thread count to 1 removes the slowdown.

## Repository layout

```
.
├── run_experiment.py        # Runs one (domain, method) config for n repeats; appends results to CSV
├── run_all.sh               # Runs the full 3 domains × 3 methods matrix unattended
├── configs/                 # One YAML per domain–method combination (9 files)
├── methods/
│   ├── base.py              # Common interface: fit / reconstruct / compression_ratio
│   ├── pca.py               # PCA (scikit-learn)
│   ├── siren.py             # SIREN (PyTorch)
│   └── hashgrid.py          # Multiresolution hash encoding + MLP head (pure PyTorch, unfused)
├── utils/
│   ├── timing.py            # Wall-clock and peak GPU memory measurement
│   ├── era5.py              # ERA5 download (CDS API) and caching
│   ├── climatebench.py      # Error bounds and quality metrics
│   └── report.py            # Summary tables from the results CSV
├── hashgrid_cost_sweep.py   # Hash-grid micro-benchmark (Fig. 1, Sec. VI-B)
├── pca_memory.py            # PCA peak memory, ms-level timing, field std (Tables III/V, Sec. VI-C)
├── make_fig.py              # Produces Fig. 1 from the sweep results
└── visualize_superres.py    # Continuous-query visualization (not used in the paper)
```

## Setup

Python 3.12, PyTorch 2.8.0 with CUDA 12.8. The paper's experiments ran on a single NVIDIA L4 GPU (24 GB) on a RunPod instance with 6 vCPUs and 55 GB RAM.

```bash
pip install torch numpy scikit-learn pyyaml xarray netCDF4 h5netcdf cdsapi matplotlib psutil threadpoolctl
```

Data is downloaded from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/), which requires a CDS API key in `~/.cdsapirc`. Downloads are cached in `era5_cache/`.

## Domains

All domains use hourly 2 m temperature for June 2025 (720 timesteps) at 0.25°.

| Domain | Latitude | Longitude | Grid | Points |
|---|---|---|---|---|
| Small (Marmara) | 39.5–41.5°N | 26.0–30.0°E | 9 × 17 | 110,160 |
| Medium (Türkiye) | 36.0–42.0°N | 26.0–45.0°E | 25 × 77 | 1,386,000 |
| Large (E. Med. + Türkiye) | 30.0–42.0°N | 19.0–45.0°E | 49 × 105 | 3,704,400 |

## Reproducing the paper

### 1. Main experiment (Tables III–V)

Run a single configuration:

```bash
python run_experiment.py --config configs/small_siren.yaml
```

Run the full matrix of 9 configurations × 5 repeats, ideally inside `tmux`:

```bash
bash run_all.sh
python -m utils.report
```

Results are appended to `results/experiment_results.csv`. The script appends rather than overwrites, so an interrupted run can be resumed.

### 2. Hash-grid micro-benchmark (Fig. 1, Sec. VI-B)

This script varies `L`, `T`, and `N_max` one at a time and measures per-step cost and CUDA kernel counts. It uses synthetic data of the small-domain shape, since per-step cost does not depend on data values. No ERA5 download is needed.

```bash
python hashgrid_cost_sweep.py --quick   # ~1 min smoke test
python hashgrid_cost_sweep.py           # full sweep, ~20–30 min on an L4
python make_fig.py                      # Fig. 1 (edit the input paths at the top if needed)
```

Outputs are written to `results/cost_sweep/`: `summary.txt`, `sweep_raw.csv`, `sweep_summary.csv`, `profile.csv`, and the figure.

### 3. PCA memory and BLAS threading (Sec. VI-C)

Do **not** run this at the same time as a GPU job: CPU load distorts the kernel-launch timings.

```bash
python pca_memory.py                                       # default BLAS threading
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python pca_memory.py --out results/pca_1thread.csv     # single BLAS thread
```

To check how many threads BLAS uses in your environment:

```bash
nproc
python -c "from threadpoolctl import threadpool_info; import numpy, pprint; pprint.pprint(threadpool_info())"
```

If `num_threads` is larger than the number of vCPUs allocated to your container, set `OMP_NUM_THREADS` accordingly before benchmarking CPU baselines.

## Notes

- **Timing on shared cloud instances.** Run-to-run variation of a fixed configuration is about 10%, and absolute timings differed by up to 1.5× between cloud instances. Ratios and slopes are more reliable than absolute values.
- **Hash-grid implementation.** The encoder is a deliberately unfused, pure-PyTorch implementation. Its cost characterizes this implementation, not grid-based representations in general. Fused implementations such as [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) are expected to be substantially faster.
- **Spectral error.** `results/experiment_results.csv` contains a `spectral_error` column. It is an unnormalized RAPSD difference dominated by the zero-frequency (mean) term, and it is not reported in the paper.
- **ω₀ = 5.** SIREN uses ω₀ = 5 instead of the default 30, because ω₀ = 30 led to unstable training at this depth (5 layers, width 96).

## Citation

```bibtex
@inproceedings{ildem2026scalability,
  author    = {{\"O}zen {\.I}ldem, Ece},
  title     = {A Scalability Study of {PCA} and Neural Representations for {ERA5} Compression},
  booktitle = {BA{\c{S}}ARIM 2026},
  address   = {Istanbul, T{\"u}rkiye},
  year      = {2026}
}
```

## Acknowledgments

This work used ERA5 reanalysis data produced by ECMWF and provided by the Copernicus Climate Change Service (C3S) Climate Data Store. Experiments were run on cloud GPU resources (RunPod).
