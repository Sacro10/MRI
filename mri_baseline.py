"""Translation, denoising, and feature preservation on a single 2-D image."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from scipy import ndimage, signal

FILENAME = "sub-03286_ses-NOT1ACH001_T1w.nii.gz"
CENTER = (115, 115)
RADIUS = 2.5
ANNULUS = (4, 7)
AMPLITUDE = 0.18
SEED = 20261115
DISPLACEMENT = np.array([7, -5])
NOISE_SIGMA = 0.06
FILTER_SIGMA = 0.85
FILTER_RADIUS = 4


def load_reference(path, synthetic):
    if synthetic:
        rows, cols = np.indices((227, 183))
        x, y = (cols - 91) / 72, (rows - 112) / 89
        radius = x**2 + y**2
        image = np.select([radius < 0.8, radius < 0.95, radius < 1], [0.65, 0.8, 1])
        image[((x + 0.28) / 0.16)**2 + ((y - 0.12) / 0.34)**2 < 1] = 0.35
        return image, {"type": "synthetic phantom"}

    import nibabel as nib

    volume = nib.as_closest_canonical(nib.load(path))
    image = np.asarray(volume.dataobj[:, :, 94], dtype=float).T
    scale = float(np.percentile(image[image > 0], 99))
    return image / scale, {
        "type": "public defaced T1-weighted MRI",
        "dataset": "OpenNeuro ds004712",
        "filename": path.name,
        "canonical_axes": nib.aff2axcodes(volume.affine),
        "canonical_axis_2_index_zero_based": 94,
        "canonical_volume_shape": volume.shape,
        "canonical_affine": volume.affine.tolist(),
        "intensity_divisor": scale,
    }


def translate(image, offset, fill=0):
    """Apply an integer (row, column) translation with constant padding."""
    return ndimage.shift(image, offset, order=0, mode="constant", cval=fill)


def estimate_correction(reference, moving):
    correlation = signal.correlate(reference, moving, mode="full", method="fft")
    peak = np.unravel_index(correlation.argmax(), correlation.shape)
    return np.asarray(peak) - (np.asarray(moving.shape) - 1)


def simulate(reference):
    rows, cols = np.indices(reference.shape)
    distance = np.hypot(rows - CENTER[0], cols - CENTER[1])
    disk = distance <= RADIUS
    annulus = (distance >= ANNULUS[0]) & (distance <= ANNULUS[1])
    target = reference + AMPLITUDE * disk
    rng = np.random.default_rng(SEED)
    moving = translate(target, DISPLACEMENT) + rng.normal(
        0, NOISE_SIGMA, size=target.shape
    )
    valid = translate(np.ones_like(disk), DISPLACEMENT)
    return target, moving, valid, disk, annulus


def process(reference, moving, valid):
    correction = estimate_correction(reference, moving)
    filtered = ndimage.gaussian_filter(moving, FILTER_SIGMA, radius=FILTER_RADIUS)
    filter_valid = ndimage.binary_erosion(
        valid, structure=np.ones((2 * FILTER_RADIUS + 1,) * 2, dtype=bool)
    )
    zero = np.zeros(2, dtype=int)
    return {
        "Degraded input": (moving, valid, zero),
        "Registration only": (translate(moving, correction), translate(valid, correction), correction),
        "Denoising only": (filtered, filter_valid, zero),
        "Registration + denoising": (
            translate(filtered, correction), translate(filter_valid, correction), correction
        ),
    }


def score(target, conditions, disk, annulus, displacement):
    common = np.ones_like(disk)
    aligned = {}
    for name, (image, valid, correction) in conditions.items():
        residual = displacement + correction
        common &= valid & translate(valid, -residual)
        aligned[name] = translate(image, -residual)

    if not np.all(common[disk | annulus]):
        raise ValueError("The feature and its annulus must fit inside the shared valid region.")

    def contrast(image):
        return float(image[disk].mean() - image[annulus].mean())

    clean_contrast = contrast(target)
    records = []
    for name, (image, _, correction) in conditions.items():
        residual = displacement + correction
        compensated = aligned[name]
        records.append({
            "condition": name,
            "correction_row_px": int(correction[0]),
            "correction_col_px": int(correction[1]),
            "residual_row_px": int(residual[0]),
            "residual_col_px": int(residual[1]),
            "residual_magnitude_px": float(np.linalg.norm(residual)),
            "valid_pixels": int(common.sum()),
            "rmse_reference_grid": float(np.sqrt(np.mean((image[common] - target[common])**2))),
            "rmse_pose_corrected": float(np.sqrt(np.mean((compensated[common] - target[common])**2))),
            "contrast_clean_target": clean_contrast,
            "contrast_reference_grid": contrast(image),
            "contrast_pose_corrected": contrast(compensated),
            "contrast_retained_pct": 100 * contrast(compensated) / clean_contrast,
        })
    return records, aligned


def save_figure(path, target, conditions, aligned, source):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6.1), layout="constrained")
    fig.suptitle(f"MRI registration and denoising baseline | {source}")
    row, col = CENTER
    half_width = 12
    settings = {"cmap": "gray", "vmin": 0, "vmax": 1, "interpolation": "nearest"}
    full_images = [target, *(item[0] for item in conditions.values())]
    zoom_images = [target, *aligned.values()]
    for index, label in enumerate(["Clean target", *conditions]):
        full, zoom = axes[:, index]
        full.imshow(full_images[index], **settings)
        full.set_title(label, fontsize=10)
        full.add_patch(Rectangle(
            (col - half_width, row - half_width), 2 * half_width, 2 * half_width,
            fill=False, edgecolor="cyan", linewidth=0.8,
        ))
        shown = zoom.imshow(zoom_images[index], **settings)
        zoom.set_xlim(col - half_width, col + half_width)
        zoom.set_ylim(row + half_width, row - half_width)
        zoom.add_patch(Circle(
            (col, row), RADIUS, fill=False, edgecolor="cyan", linewidth=0.8
        ))
        for axis in (full, zoom):
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_ylabel("Reference grid")
    axes[1, 0].set_ylabel("Pose-corrected zoom")
    fig.colorbar(shown, ax=axes, shrink=0.75, label="Normalized intensity")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=Path, default=Path("data") / FILENAME)
    parser.add_argument("--synthetic", action="store_true", help="Run on a synthetic phantom.")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    reference, source = load_reference(args.volume, args.synthetic)
    target, moving, valid, disk, annulus = simulate(reference)
    conditions = process(reference, moving, valid)
    records, aligned = score(target, conditions, disk, annulus, DISPLACEMENT)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "source": source,
        "coordinates": "zero-based (row, column); positive shifts down and right",
        "seed": SEED,
        "applied_translation_px": DISPLACEMENT.tolist(),
        "noise_distribution": "additive zero-mean Gaussian",
        "noise_sigma": NOISE_SIGMA,
        "feature": {
            "shape": "disk",
            "center_row_col_px": CENTER,
            "radius_px": RADIUS,
            "contrast_annulus_inner_outer_radii_px": ANNULUS,
            "added_intensity": AMPLITUDE,
        },
        "gaussian_filter_sigma_px": FILTER_SIGMA,
        "gaussian_filter_radius_px": FILTER_RADIUS,
    }
    (args.output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    save_figure(args.output / "comparison.png", target, conditions, aligned, source["type"])
    print(f"Saved metrics.csv, experiment.json, and comparison.png in {args.output}")


if __name__ == "__main__":
    main()
