# MRI registration, denoising, and feature preservation

This software baseline uses one 2-D MRI slice and a simulated bright feature. All four conditions receive the same noisy, translated image. Prior-guided enhancement is a future experiment.

## Setup

Use Python 3.14 and run from the project directory.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data
curl --fail --location --output data/sub-03286_ses-NOT1ACH001_T1w.nii.gz \
  https://s3.amazonaws.com/openneuro.org/ds004712/sub-03286/ses-NOT1ACH001/anat/sub-03286_ses-NOT1ACH001_T1w.nii.gz
```

## Run

```sh
python mri_baseline.py
```

The command writes `results/comparison.png`, `results/metrics.csv`, and `results/experiment.json`. If the download is unavailable, append `--synthetic --output results/phantom` to run on a labeled synthetic phantom. After downloading the volume, use the command above to rerun on MRI.

## Source and simulation

The source is the CC0, anonymized and defaced ON-Harmony T1-weighted volume for subject `sub-03286`, session `ses-NOT1ACH001`. The selected slice is axial index **94** (zero-based) along axis 2 after canonical RAS reorientation. The slice is transposed into a 227-row × 183-column image and divided by the 99th percentile of its positive voxel intensities. The volume SHA-256 used for the checked results is `9b539655095f4007181d117ca1574e7f4dfe06f5c86a117100890d79eb0a6af4`. [OpenNeuro dataset](https://doi.org/10.18112/openneuro.ds004712.v2.0.1) · [Warrington et al., Scientific Data (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11992253/)

Coordinates are zero-based (row, column), increasing down and right. A disk centered at (115, 115), radius 2.5 pixels, adds 0.18 normalized intensity units to form the clean target. The simulation moves it by (+7, −5) pixels with zero padding, then adds independent zero-mean Gaussian noise with standard deviation 0.06, seeded with 20261115. The simulation keeps intensities unclipped. The JSON records these parameters and slice provenance.

## Methods and scoring

- **Registration.** The peak of the full linear cross-correlation against the original feature-free slice gives the integer correction to apply to the degraded image. Both registered conditions use that same estimate. Integer shifts copy pixels and pad with zeros. The translation columns give the applied correction and the remaining displacement (simulated displacement plus correction), including its Euclidean magnitude.
- **Denoising.** Both denoised conditions apply the same Gaussian filter in the moving frame, with sigma 0.85 pixels and a four-pixel kernel radius. The combined condition then applies the estimated correction.
- **Common frame.** Reference-grid metrics compare each output as produced with the clean target at matching array indices, retaining any residual displacement. Pose-corrected metrics first undo each output's known residual translation for evaluation only. Both comparisons use the same intersection of valid reference-grid pixels across all four conditions and both scoring modes. Validity masks follow every shift, and the denoised masks exclude pixels whose filter footprint touches padding or the array edge.
- **Metrics and figure.** RMSE uses normalized intensities. Feature contrast is the mean intensity in the target disk minus the mean in the concentric 4–7 pixel annulus. Contrast retention is pose-corrected contrast divided by clean target contrast. Both masks lie inside the shared valid region. The figure shows the reference grid above and the pose-corrected feature zoom below, with grayscale limits of 0–1 throughout.

## Adding a physical phantom

Use a new loader and acquisition-pair preparation function in place of `load_reference` and `simulate`. Prepare co-sampled 2-D arrays for the feature-free reference, clean feature-bearing target, and moving acquisition, plus its support mask and feature and background masks in reference coordinates. Record acquisition IDs, slice orientation, voxel spacing, intensity scaling, and how the known displacement was measured. The processing stage accepts the reference, moving image, and support mask directly. Pass the measured displacement to the scoring stage. The current estimator and scoring shifts assume an integer in-plane translation.
