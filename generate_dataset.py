"""Generate aligned RAW56 arrays directly from BSRApp DNG captures."""

import argparse
import csv
import pickle
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import colour_demosaicing
import cv2
import numpy as np
import rawpy
from tqdm import tqdm

from functions import (
    METRIC_COLUMNS, OUTPUT_FILES, SEARCH_REGION, align_ecc, camera_frames, camera_metadata,
    check_sensor, contains_dng, estimate_hr_transform, find_maximum_rect, frame_metadata,
    linear_to_srgb, translate_matrix, update_registration_csv, warp_hr_crop,
)


def analyze_folder(folder, out_folder=None):

    print(f"Processing folder {folder.name}...")

    margin = 150
    context_crop = tuple(slice(axis.start - margin, axis.stop + margin) for axis in SEARCH_REGION)
    inner_crop = slice(margin, -margin), slice(margin, -margin)

    frames = camera_frames(folder)
    if out_folder is None:
        with rawpy.imread(str(frames["2"][0])) as raw:
            check_sensor(raw, frames["2"][0], "RGGB")
            reference_rgb = raw.postprocess()[context_crop][inner_crop].copy()
        with rawpy.imread(str(frames["7"][0])) as raw:
            check_sensor(raw, frames["7"][0], "GRBG")
            hr_rgb = raw.postprocess()
        _, metrics = estimate_hr_transform(reference_rgb, hr_rgb, return_metrics=True)
        return {"scene": folder.name, **metrics}

    lr_raw = []
    lr_linear = []

    for index, path in enumerate(frames["2"]):
        with rawpy.imread(str(path)) as raw:
            check_sensor(raw, path, "RGGB")
            mosaic = raw.raw_image
            if any(axis.stop > size for axis, size in zip(context_crop, mosaic.shape)):
                raise ValueError(f"{path.name} is smaller than the calibrated LR crop")
            info = camera_metadata(raw)
            if index == 0:
                lr_metadata = info
                reference_rgb = raw.postprocess()[context_crop][inner_crop].copy()
            elif any(not np.array_equal(value, lr_metadata[key]) for key, value in info.items()):
                raise ValueError(f"Inconsistent camera 2 metadata in {path.name}")
            raw_img = np.maximum(mosaic[context_crop].astype(np.float32) - 63, 0).astype(np.uint16)
        lr_raw.append(raw_img)
        lr_linear.append(colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(raw_img, pattern="RGGB"))

    with rawpy.imread(str(frames["7"][0])) as raw:
        check_sensor(raw, frames["7"][0], "GRBG")
        hr_metadata = camera_metadata(raw)
        mosaic = np.maximum(raw.raw_image.astype(np.float32) - 63, 0).astype(np.uint16)
        hr_linear = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(mosaic, pattern="GRBG")
        hr_rgb = raw.postprocess()

    matrix_lr_hr, metrics = estimate_hr_transform(reference_rgb, hr_rgb, return_metrics=True)

    # Find optimal rectangular FOV
    mask = cv2.warpPerspective(np.ones(hr_linear.shape[:2], dtype=np.float64),
                               matrix_lr_hr.astype(np.float64), reference_rgb.shape[1::-1])

    mask_slice = find_maximum_rect(mask)
    if any(axis.stop <= axis.start for axis in mask_slice):
        raise ValueError("Aligned images have no usable common crop")

    hr_image_8x = warp_hr_crop(hr_linear, matrix_lr_hr, reference_rgb.shape[:2], mask_slice)
    np.save(out_folder / 'hr.npy', (hr_image_8x / 960).astype(np.float32))

    homography_matrices_norm = [np.eye(3, dtype=matrix_lr_hr.dtype)]
    reference_srgb = linear_to_srgb(lr_linear[0], lr_metadata)
    lr_image_raws_norm = [lr_raw[0][inner_crop][mask_slice]]

    ref_image_norm = cv2.cvtColor(reference_srgb, cv2.COLOR_RGB2GRAY)

    print("LR homographies...")
    H = None

    for i in range(1, len(lr_linear)):
        img = lr_linear[i]
        img_norm = cv2.cvtColor(linear_to_srgb(img, lr_metadata), cv2.COLOR_RGB2GRAY)

        H = align_ecc(img_norm, ref_image_norm, cv2.MOTION_EUCLIDEAN, H)

        H_t = translate_matrix(H, inner_crop[0].start + mask_slice[0].start, inner_crop[1].start + mask_slice[1].start)

        # mult2 integer alignment, fix the H matrix
        x0, y0 = H_t[0, 2], H_t[1, 2]
        x_new = int(x0 / 2) * 2
        y_new = int(y0 / 2) * 2
        T = [[1, 0, x_new],
             [0, 1, y_new],
             [0, 0, 1]]

        T_inv = np.linalg.inv(T)
        H_norm = H_t @ T_inv

        homography_matrices_norm.append(H_norm)

        img_raw_norm = np.roll(lr_raw[i], shift=(y_new, x_new), axis=(0, 1))[inner_crop][mask_slice]
        lr_image_raws_norm.append(img_raw_norm)

    print("Saving results...")

    np.save(out_folder / 'h_norm.npy', np.array(homography_matrices_norm))
    np.save(out_folder / 'lr.npy', lr_image_raws_norm)

    with (out_folder / 'meta_lr.pkl').open('wb') as file:
        pickle.dump(frame_metadata(frames["2"][0], lr_metadata), file)
    with (out_folder / 'meta_hr.pkl').open('wb') as file:
        pickle.dump(frame_metadata(frames["7"][0], hr_metadata), file)
    return {"scene": folder.name, **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path(r"G:\indng"))
    parser.add_argument("output", nargs="?", type=Path, default=Path(r"G:\outdng2"))
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if not source.is_dir():
        parser.error(f"Input directory does not exist: {source}")
    if source == output or source in output.parents:
        parser.error("Choose an output directory outside the input tree")

    folders = [source] if contains_dng(source) else sorted(
        folder for folder in source.iterdir() if folder.is_dir() and contains_dng(folder))
    if not folders:
        parser.error(f"No DNG captures found in {source} or its immediate subdirectories")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "registration_per_scene.csv"
    records = {}
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(METRIC_COLUMNS):
                parser.error(f"Unexpected CSV columns in {csv_path}")
            records = {row["scene"]: row for row in reader}
    failed = False
    for folder in tqdm(folders):
        destination = output / folder.name
        try:
            if destination.exists():
                if all((destination / name).is_file() for name in OUTPUT_FILES):
                    if folder.name not in records:
                        records[folder.name] = analyze_folder(folder)
                        update_registration_csv(csv_path, records)
                    print(f"Skipping existing capture {destination}")
                    continue
                raise ValueError(f"Expected {', '.join(OUTPUT_FILES)} at {destination}. Choose a new output directory or move the existing capture aside.")
            with TemporaryDirectory(dir=output) as temp:
                staging = Path(temp) / "capture"
                staging.mkdir()
                record = analyze_folder(folder, staging)
                staging.rename(destination)
            records[folder.name] = record
            update_registration_csv(csv_path, records)
        except Exception as error:
            failed = True
            print(f"{folder}: {error}", file=sys.stderr)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
