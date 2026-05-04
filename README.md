# PhaseDenoising — MT-InSAR Denoising with Deep Learning

> Implementation of **"A novel lightweight 3D CNN for accurate deformation time series retrieval in MT-InSAR"**  
> Abdallah, M., Ding, X., Younis, S., & Wu, S. (2025). *Science of Remote Sensing*, 11, 100206.  
> [https://doi.org/10.1016/j.srs.2025.100206](https://doi.org/10.1016/j.srs.2025.100206)

---

## Overview

Multi-temporal InSAR (MT-InSAR) deformation time series are contaminated by several unwanted phase signals — spatially correlated atmospheric phase screen (APS) effects, topographically correlated APS, orbital error ramps, thermal noise, and decorrelation artefacts.  Accurately separating the deformation signal from these noise sources is the central challenge this repository addresses.

**PhaseDenoising** proposes `UNet-3D`: a lightweight three-dimensional encoder-decoder that processes the full spatiotemporal volume of time-accumulated interferograms in a single forward pass, jointly exploiting spatial and temporal phase correlations.  A spatiotemporal decorrelation mask is concatenated to the input, ensuring the model focuses recovery efforts on missing pixels caused by coherence loss.

### Key Results (from the paper)

| Metric | Improvement over UNet-2D baseline |
|--------|-----------------------------------|
| MSE | **+25.0 %** |
| SSIM | **+1.8 %** |
| SNR | **+0.2 %** |
| Computational cost (separable conv) | **−80 %** |

Real-data validation over **Fernandina Volcano, Galápagos Islands** (Sentinel-1 / SBAS):

| Metric | Value |
|--------|-------|
| Correlation with MintPy SBAS | **0.91** |

---

## Repository Structure

```
PhaseDenoising/
│
├── Data/                          # Synthetic dataset generation
│   ├── bin/
│   │   └── data.py                # CLI — batch dataset generation script
│   └── src/
│       ├── __init__.py            # Public API: Stack, TimeSeries, Ifg, Ifgs, Dem, Heights, Water
│       ├── base/
│       │   ├── single.py          # Single — one wrapped interferogram with all components
│       │   └── stack.py           # Stack — full MT-InSAR simulation factory (Stack.create)
│       ├── timeseries/
│       │   └── timeseries.py      # TimeSeries — extends Stack; adds getIfgs(), plot()
│       ├── ifg/
│       │   ├── ifg.py             # Ifg — single-interferogram factory (Ifg.create)
│       │   └── ifgs.py            # Ifgs — stack of interferograms derived from TimeSeries
│       ├── slc/
│       │   ├── slc.py             # Slc — single SAR image
│       │   └── slcs.py            # Slcs — stack of SAR images
│       ├── dem/
│       │   ├── dem.py             # Dem — SRTM DEM download + water-mask composite
│       │   ├── height.py          # Heights — SRTM tile download and resampling
│       │   ├── water.py           # Water — GSHHS shoreline rasterisation
│       │   ├── waterlines.py      # GSHHS vector processing
│       │   ├── raster.py          # DemRaster — unified height + mask array
│       │   ├── geolocation.py     # Geolocation utilities
│       │   ├── srtm.py            # SRTM tile fetching
│       │   └── gshhs.py           # GSHHS data handling
│       ├── defo/
│       │   ├── model.py           # Analytic source models: Cone, Peak, Mogi, Okada
│       │   ├── source.py          # Source — selects and instantiates a geophysical model
│       │   ├── pattern.py         # Pattern — spatial deformation template
│       │   ├── trend.py           # Trend — temporal evolution (linear, sinusoidal, coseismic…)
│       │   ├── wrapper.py         # LOS projection wrapper
│       │   ├── complex.py         # Complex (composite) deformation scenarios
│       │   ├── kwargs.py          # SrcKwargs — parameter packaging
│       │   └── dtype.py           # DefoData typed array
│       ├── delays/
│       │   ├── tropospheric.py    # Tropospheric — stratified APS (linear / quadratic w.r.t. DEM)
│       │   ├── turblent.py        # Turblent — spatially correlated APS (FFT / covariance / fractal)
│       │   ├── topographic.py     # Topographic — DEM-correlated delay
│       │   ├── orbit.py           # Orbit — polynomial orbital ramps (2nd / 3rd / 5th degree)
│       │   └── dtype.py           # DelayData / DelayMaskedData typed arrays
│       ├── noise/
│       │   ├── noise.py           # Speckle, Thermal, Decorelation noise generators
│       │   └── dtype.py           # NoiseData typed array
│       ├── mask/
│       │   ├── coherence.py       # Coherence — spatiotemporal decorrelation mask
│       │   └── dtype.py           # MaskData typed array
│       ├── baseline/
│       │   ├── spatial.py         # Spatial baseline calculation
│       │   └── temporal.py        # Temporal baseline / time-base generation
│       ├── dtype/
│       │   ├── data.py            # Data — base ndarray wrapper (unit conversion, arithmetic)
│       │   └── vector.py          # Vector typed array
│       ├── objects/
│       │   ├── sensor.py          # Sensor — satellite parameters (wavelength, critical baselines)
│       │   ├── proj.py            # Projection — LOS unit-vector computation
│       │   ├── convert.py         # Converter — geo-to-pixel coordinate mapping
│       │   ├── params.py          # Common physical constants
│       │   └── geofile.py         # GeoTIFF I/O helpers
│       ├── utils/
│       │   ├── utils.py           # MinMax, MeanVar, MeanStd normalisation helpers
│       │   ├── scale.py           # Scale — unit-aware scaling (rad ↔ m ↔ cm)
│       │   ├── wrap.py            # wrap_phase — atan2 wrapping
│       │   └── detrend.py         # Detrend — polynomial detrending
│       └── plot/
│           └── ploter.py          # Ploter — multi-panel time-series visualisation
│
├── Network/                       # Deep learning training pipeline
│   └── src/
│       ├── __init__.py            # Public API: Trainer, Dataset
│       ├── dataset.py             # Dataset — PyTorch dataset; reads TimeSeries pickles from Data/
│       ├── network.py             # Network — model wrapper with optimiser, scheduler, loss, AMP
│       ├── trainer.py             # Trainer — full training loop: fit / train / valid / test
│       ├── metrics.py             # Metric — running mean tracker (loss accumulator)
│       ├── checkpoint.py          # CheckPoint — checkpoint directory management
│       ├── logger.py              # CSVLogger, Hyperparameters — training log writers
│       ├── callbacks.py           # Callback — early-stopping with configurable patience
│       ├── progbar.py             # ProgBar — Keras-style ASCII progress bar
│       ├── tensorboard.py         # TensorBoard — per-epoch visual comparison (stack / GT / output)
│       └── model/
│           ├── unet2d/
│           │   ├── models.py      # Unet2D — 4-level 2D encoder-decoder
│           │   ├── layers.py      # EncoderBlock, DecoderBlock, NeckBlock, InOutBlock (2D)
│           │   └── modules.py     # ConvBock, DeConvBlock, Concatenate (2D)
│           └── unet3d/
│               ├── models.py      # Unet3D — 4-level 3D encoder-decoder (paper architecture)
│               ├── layers.py      # EncoderBlock, DecoderBlock, NeckBlock, InOutBlock (3D)
│               └── modules.py     # ConvBock, DeConvBlock, Concatenate (3D)
│
├── CheckPoints/                   # Saved model weights and training logs
│   └── <strtime>/
│       ├── Unet.pth               # Model state dict (best validation loss)
│       ├── <Name>_<strtime>.csv   # Per-epoch loss / val_loss
│       ├── <Name>_<strtime>.txt   # Hyperparameter snapshot
│       └── <Name>_<strtime>_NNN.png  # Visual samples during training
│
└── Datasets/                      # Generated training / validation / test data
    └── <source>/
        ├── train/
        ├── valid/
        └── test/
```

---

## Data Package — Synthetic Dataset Generator

Training a deep learning model to separate deformation from noise requires large, diverse, *paired* datasets (noisy interferogram stack → clean deformation map) that are impossible to gather at scale from real observations.  The `Data` package solves this by **fully synthesising** time series of SAR interferograms from physical first principles.

### Simulated Phase Components

Each synthetic interferogram stack is the sum of the following components, matching the paper's Section 4.1:

| Component | Class | Notes |
|-----------|-------|-------|
| Deformation | `defo.Source` + `defo.Trend` | Mogi / Okada / Dyke / Sill / Cone / Peak; temporal evolution: linear, sinusoidal, coseismic, postseismic, etc. |
| Turbulent APS | `delays.Turblent` | Spatially correlated; generated via FFT, covariance, eigenvalue, fractal, or trend methods |
| Stratified APS | `delays.Tropospheric` | Height-correlated; 1st- or 2nd-order polynomial w.r.t. SRTM DEM |
| Topographic residual | `delays.Topographic` | Perpendicular baseline × DEM height |
| Orbital ramps | `delays.Orbit` | Linear polynomial in azimuth/range (2nd / 3rd / 5th degree) |
| Thermal noise | `noise.Thermal` | Zero-mean Gaussian scaled to target SNR |
| Decorrelation noise | `noise.Decorelation` | Modulated by spatial (critical baseline) and temporal (critical time) coherence |
| Coherence mask | `mask.Coherence` | Spatiotemporal binary mask (threshold-based) |
| Water mask | `dem.Water` | GSHHS shoreline rasterisation |

All components are expressed in **radians** internally and can be converted to metres or centimetres via `Data.tometer()` / `Data.tocm()`.

### Supported Satellite Sensors

| Code | Satellite |
|------|-----------|
| `SENTINEL` | Sentinel-1 A/B (default) |
| `ASAR` | Envisat ASAR |
| `ERS` | ERS-1 / ERS-2 |
| `ALOS1` | ALOS PALSAR-1 |
| `ALOS2` | ALOS-2 PALSAR-2 |
| `RADARSAT` | RADARSAT-2 |
| `CSK` | COSMO-SkyMed |
| `TSK` | TerraSAR-X |

### Supported Deformation Sources

| Code | Model |
|------|-------|
| `mogi` | Mogi (1958) point-source inflation/deflation |
| `quake` / `normal` / `thrust` / `left-lateral` / `right-lateral` | Okada (1985) fault plane |
| `dyke` | Opening dyke (Okada) |
| `sill` | Horizontal sill intrusion |
| `cone` | Symmetric cone (analytic) |
| `peak` | MATLAB-style peaks surface |
| `nodefo` | No deformation — noise-only training sample |

---

## Network Package — Training Pipeline

The `Network` package implements two encoder-decoder backbones and the full training loop.

### Backbones

| Backbone | Class | Input shape | Convolution |
|----------|-------|-------------|-------------|
| `unet2d` | `Unet2D` | `(B, N, H, W)` — N channels | 2D |
| `unet3d` | `Unet3D` | `(B, 1, N, H, W)` — N = temporal depth | 3D (spatiotemporal) |

### UNet-3D Architecture

Four-level symmetric encoder-decoder operating on the full spatiotemporal volume `(B, 1, D, H, W)`.  Spatial resolution is halved at each encoder stage while temporal depth `D` is preserved throughout.

```
Input  (B, 1, D, H, W)
  │
  ├─ InOutBlock          1  →  F              1×1×1 conv
  │
  ├─ EncoderBlock 1      F  →  F    + skip₁   MaxPool3d (1,2,2)
  ├─ EncoderBlock 2      F  →  2F   + skip₂   MaxPool3d (1,2,2)
  ├─ EncoderBlock 3      2F →  4F   + skip₃   MaxPool3d (1,2,2)
  │
  ├─ NeckBlock           4F →  8F             bottleneck
  │
  ├─ DecoderBlock 1      8F →  4F   ← skip₃   TranspConv3d (1,2,2) + concat
  ├─ DecoderBlock 2      4F →  2F   ← skip₂   TranspConv3d (1,2,2) + concat
  ├─ DecoderBlock 3      2F →  F    ← skip₁   TranspConv3d (1,2,2) + concat
  │
  └─ InOutBlock          F  →  1              1×1×1 conv
Output (B, D, H, W)
```

Each `ConvBock` contains two `Conv3d(kernel=(1,3,3))` layers with PReLU activation and BatchNorm3d, preserving the temporal dimension while filtering spatially.

#### Model naming convention

Variants are named `UNet-3D-{T}-{S/O}-{M}`:

| Token | Meaning |
|-------|---------|
| `T` ∈ {1, 3, 5} | Number of temporal filters in the depth-wise kernel |
| `S` / `O` | Separable or original convolution |
| `M` | Decorrelation mask concatenated to input |

### Optimisation

| Setting | Default |
|---------|---------|
| Optimiser | Adam |
| Learning rate | `2e-4` |
| Adam β₁ / β₂ | `0.5` / `0.999` |
| LR scheduler | `ReduceLROnPlateau` (`factor=0.1`, `patience=5`) |
| Loss function | L1 (MAE) |
| Mixed precision | Optional (PyTorch AMP) |
| Multi-GPU | Optional (`nn.DataParallel`) |
| Early stopping | `patience=10`, `min_delta=1e-5` on `val_loss` |

---

## Quick Start

### 1 — Generate a dataset

```python
from Data.src import TimeSeries

ts = TimeSeries.create(
    ndates    = 20,
    shape     = (48, 48),
    location  = {'west': -91.7, 'east': -91.4, 'south': -0.5, 'north': -0.25},
    platform  = 'SENTINEL',
    source    = 'mogi',
    disp      = 'linear',
    method    = 'fft',
    polydeg   = 'second',
    threshold = 0.2,
    snr       = 2.0,
)

noisy  = ts.getNoisy()   # (H × W × N) [rad]  — noisy time-accumulated interferograms
clean  = ts.getTrend()   # (H × W × N) [rad]  — noise-free deformation maps
mask   = ts.getMask()    # (H × W)     [bool] — spatiotemporal decorrelation mask
```

### 2 — Generate a batch dataset from the CLI

```bash
python Data/bin/data.py \
    --dir        ./Datasets \
    --phase      train \
    --ndata      20000 \
    --nifgs      20 \
    --resolution 48 48 \
    --platform   SENTINEL \
    --source     mogi \
    --trend      linear \
    --method     fft \
    --orbitdeg   second \
    --snr        2.0 \
    --cohthreshold 0.2
```

Writes compressed pickle archives to `Datasets/<source>/<phase>/stack_<source>_NNNNNN.pkl`.

### 3 — Train

```python
from Network.src import Trainer

trainer = Trainer(
    datadir        = './Datasets',
    source         = 'mogi',
    backbone       = 'unet3d',
    dim            = 64,
    mode           = 'ts',
    ismasked       = True,
    augment        = True,
    batch_size     = 32,
    lr             = 2e-4,
    checkpointsdir = './CheckPoints',
    seed           = 42,
)

trainer.fit(epochs=100)
```

### 4 — Resume training

```python
trainer.load()
trainer.fit(epochs=50)
```

---

## Key CLI Arguments (`Data/bin/data.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `-d` / `--dir` | `./Datasets` | Output directory |
| `-ph` / `--phase` | `train` | Dataset split (`train` / `valid` / `test`) |
| `-nd` / `--ndata` | `100 000` | Number of stacks to generate |
| `-n` / `--nifgs` | `10` | Number of interferograms per stack |
| `-r` / `--resolution` | `48 48` | Spatial dimensions `(H, W)` |
| `-p` / `--platform` | `SENTINEL` | SAR sensor |
| `-s` / `--source` | random | Deformation source type |
| `-tr` / `--trend` | random | Temporal deformation evolution |
| `-m` / `--turbmethod` | random | Turbulent APS method (`fft`, `cov`, `eig`, `fractal`, `trend`) |
| `-od` / `--orbitdeg` | random | Orbital ramp degree (`second`, `third`, `fifth`) |
| `-ct` / `--cohthreshold` | `0.20` | Coherence threshold for decorrelation mask |
| `-sn` / `--snr` | `2.0` | Target signal-to-noise ratio |
| `-lf` / `--lfile` | `None` | CSV file of geo-locations (random row per sample) |

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- torchvision
- numpy, scipy, matplotlib, Pillow
- gdal / rasterio
- requests

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{abdallah2025unet3d,
  title   = {A novel lightweight {3D} {CNN} for accurate deformation time series
             retrieval in {MT-InSAR}},
  author  = {Abdallah, Mahmoud and Ding, Xiaoli and Younis, Samaa and Wu, Songbo},
  journal = {Science of Remote Sensing},
  volume  = {11},
  pages   = {100206},
  year    = {2025},
  doi     = {10.1016/j.srs.2025.100206},
  url     = {https://www.sciencedirect.com/science/article/pii/S2666017225000124}
}
```

---

## References

- Abdallah, M., Ding, X., Younis, S., & Wu, S. (2025). [A novel lightweight 3D CNN for accurate deformation time series retrieval in MT-InSAR](https://doi.org/10.1016/j.srs.2025.100206). *Science of Remote Sensing*, 11, 100206.
- Mogi, K. (1958). Relations between the eruptions of various volcanoes and the deformations of the ground surfaces around them. *Bull. Earthq. Res. Inst.*, 36, 99–134.
- Okada, Y. (1985). Surface deformation due to shear and tensile faults in a half-space. *Bull. Seismol. Soc. Am.*, 75(4), 1135–1154.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). *MICCAI*.
- Sun, Y. et al. (2020). Deep learning for atmospheric correction in MT-InSAR. *IEEE TGRS*, 58(6), 4070–4083.
- Rouet-Leduc, B. et al. (2021). [Autonomous extraction of millimeter-scale deformation in InSAR time series using deep learning](https://doi.org/10.1038/s41467-021-26254-3). *Nature Communications*, 12, 6079.
- Hanssen, R. F. (2001). *Radar Interferometry: Data Interpretation and Error Analysis*. Springer.

---

## License

MIT
