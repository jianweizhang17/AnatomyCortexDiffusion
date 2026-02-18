# AnatomyCortexDiffusion

Implementation for the paper [“Anatomy-Guided Surface Diffusion Model for Alzheimer’s Disease Normative Modeling”](https://openreview.net/forum?id=PWw0GoQUXV).

## Overview
This codebase trains and samples a cortical surface diffusion model using surface feature files in **FreeSurfer `.curv`** format (e.g., thickness / curvature / tau) defined on an icosphere mesh (default: ico-6, 40962 vertices).

## Data format
- **Surface feature files**: FreeSurfer `*.curv` files.
- **Metadata CSV**: a CSV listing samples and the corresponding file paths.
  - See `example_data.csv` for the expected header style.
- **Data directory**: pass the folder containing your `.curv` files with `--data_dir`.

## Train
Use `run_train.sh` as an example entry point (edit arguments as needed):

```bash
bash run_train.sh
```

Key arguments used by `run_train.sh`:
- `--data_dir`: directory containing `.curv` files
- `--data_info_csv`: CSV listing data ids and `.curv` paths
- `--input_channel` / `--output_channel`: number of channels
- `--norm_mode`: normalization mode (e.g. `standard`)

## Test / sampling
Use `run_test.sh` to generate samples and/or harmonized outputs:

```bash
bash run_test.sh
```

You will need to set:
- `--checkpoint_file`: path to a trained checkpoint
- `--sample_steps`, `--noise_step`: sampling settings

## Aux data (mesh neighborhood orders)
The mesh neighborhood / ordering data is loaded from `aux_data/ic*.mat`.

If your `ic*.mat` files are not located in `aux_data/`, set:

```bash
export ANATOMY_CORTEXDIFFUSION_AUX_DATA_DIR="/path/to/aux_data"
```
