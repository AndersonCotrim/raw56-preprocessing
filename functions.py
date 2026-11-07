"""Shared helpers for RAW56 DNG preprocessing and dataset verification."""

import csv
import pickle
import re

METRIC_COLUMNS = ("scene", "sift_rmse_px", "ecc_rmse_px", "grad_corr_sift", "grad_corr_ecc")

OUTPUT_FILES = ("lr.npy", "hr.npy", "h_norm.npy", "meta_lr.pkl", "meta_hr.pkl")
# Even-aligned bounds of camera 7's field of view in camera 2 coordinates.
SEARCH_REGION = (slice(1380, 1586), slice(1872, 2140))


def contains_dng(folder):
    return any(p.suffix.lower() == ".dng" and p.is_file() for p in folder.iterdir())


def camera_frames(folder):
    frames = {"2": {}, "7": {}}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() != ".dng":
            continue
        match = re.search(r"_(\d+)_(\d+)$", path.stem)
        if not match:
            raise ValueError(f"Unexpected DNG filename: {path.name}")
        camera, sequence = match.groups()
        if camera in frames:
            index = int(sequence)
            if index in frames[camera]:
                raise ValueError(f"Duplicate frame {index} for camera {camera}")
            frames[camera][index] = path
    for camera, paths in frames.items():
        if not paths:
            raise ValueError(f"No DNG frames for camera {camera}")
    return {camera: [paths[i] for i in sorted(paths)] for camera, paths in frames.items()}


def camera_metadata(raw):
    return {
        "black_level": raw.black_level_per_channel,
        "cam_wb": raw.camera_whitebalance,
        "daylight_wb": raw.daylight_whitebalance,
        "color_matrix": raw.color_matrix.copy(),
        "rgb_xyz_matrix": raw.rgb_xyz_matrix.copy(),
        "norm_factor": raw.white_level,
    }


def frame_metadata(path, info):
    import exifread

    with path.open("rb") as stream:
        exif = exifread.process_file(stream)
    # The legacy exporter stored per-frame timestamps inside the discarded Array Info.
    for tag in ("Image DateTime", "Image DateTimeOriginal", "Array Info"):
        exif.pop(tag, None)
    return {"info": info, "exif": exif}


def check_sensor(raw, path, expected_pattern):
    pattern = "".join(chr(raw.color_desc[i]) for i in raw.raw_colors_visible[:2, :2].flat)
    if pattern != expected_pattern or raw.white_level != 1023 or any(
        level != 63 for level in raw.black_level_per_channel
    ):
        raise ValueError(f"{path.name} does not match the {expected_pattern}, 63–1023 calibration")


def compare_file(generated, reference):
    for path in (generated, reference):
        if not path.is_file():
            return f"missing file: {path}"

    size = generated.stat().st_size
    reference_size = reference.stat().st_size
    if size != reference_size:
        return f"size differs (generated {size} bytes, reference {reference_size} bytes)"

    with generated.open("rb") as left, reference.open("rb") as right:
        while True:
            block = left.read(1024 * 1024)
            if block != right.read(1024 * 1024):
                return "contents differ"
            if not block:
                return None


def compare_metadata(generated, reference_folder):
    """Compare info and exif against trusted split or combined reference metadata."""
    try:
        with generated.open("rb") as stream:
            metadata = pickle.load(stream)
        reference_path = reference_folder / generated.name
        if reference_path.is_file():
            with reference_path.open("rb") as stream:
                reference = pickle.load(stream)
        else:
            with (reference_folder / "meta.pkl").open("rb") as stream:
                reference = pickle.load(stream)[generated.stem.removeprefix("meta_")]
        if set(metadata) != {"info", "exif"}:
            return "expected info and exif"
        for section in ("info", "exif"):
            actual = metadata[section]
            expected = reference[section]
            if actual.keys() != expected.keys():
                return f"{section}: fields differ"
            for key in expected:
                if pickle.dumps(actual[key]) != pickle.dumps(expected[key]):
                    return f"{section}.{key}: values differ"
    except (OSError, pickle.UnpicklingError, EOFError, KeyError, TypeError, AttributeError, ImportError) as error:
        return f"could not read metadata: {error}"
    return None


def linear_to_srgb(rgb, meta):
    import numpy as np

    rgb = rgb.astype(np.float32)

    rgb /= 960.0

    wb = np.array(meta['cam_wb'][:3], dtype=np.float32)
    wb /= wb[1]
    rgb[..., 0] *= wb[0]
    rgb[..., 2] *= wb[2]

    rgb = np.clip(rgb, 0, 1)

    rgb = np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(rgb, 1 / 2.4) - 0.055
    )

    return np.clip(rgb, 0, 1)


def estimate_hr_transform(reference_rgb, hr_rgb, *, return_metrics=False):
    import cv2
    import numpy as np

    ref_points, ref_features = cv2.SIFT_create().detectAndCompute(reference_rgb, None)
    blurred = cv2.GaussianBlur(hr_rgb, (51, 51), 0)
    hr_points, hr_features = cv2.SIFT_create().detectAndCompute(blurred, None)
    if ref_features is None or hr_features is None:
        raise ValueError("No SIFT descriptors for HR alignment")
    pairs = cv2.BFMatcher().knnMatch(ref_features, hr_features, k=2)
    matches = []
    for pair in pairs:
        if len(pair) == 2:
            best, second = pair
            if best.distance < 0.5 * second.distance:
                matches.append(best)
    if len(matches) < 5:
        raise ValueError("Too few SIFT matches for HR alignment")
    ref_xy = np.float32([ref_points[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    hr_xy = np.float32([hr_points[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, _ = cv2.estimateAffinePartial2D(hr_xy, ref_xy, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("Could not estimate HR alignment")
    initial = np.eye(3, dtype=np.float32)
    initial[:2] = matrix
    refined = align_ecc(cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY),
                        cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY),
                        cv2.MOTION_AFFINE, initial.copy())

    if not return_metrics:
        return refined

    metrics = dict.fromkeys(METRIC_COLUMNS[1:])
    try:
        metrics["sift_rmse_px"], metrics["ecc_rmse_px"] = registration_rmse(
            ref_points, hr_points, pairs, initial, refined)
    except (ValueError, cv2.error) as error:
        print(f"Registration RMSE unavailable: {error}")
    try:
        metrics["grad_corr_sift"], metrics["grad_corr_ecc"] = gradient_correlation(
            reference_rgb, blurred, initial, refined)
    except (ValueError, cv2.error) as error:
        print(f"Gradient correlation unavailable: {error}")
    return refined, metrics


def align_ecc(template, image, motion, initial=None):
    import cv2
    import numpy as np

    if initial is None:
        initial = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
    _, affine = cv2.findTransformECC(template, image, initial[:2], motion, criteria,
                                     inputMask=None, gaussFiltSize=5)
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2] = affine
    return matrix


def find_maximum_rect(mask):
    import cv2
    import largestinteriorrectangle as lir
    import numpy as np

    mask = mask.astype("uint8") * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Aligned images have no overlap")
    x, y, width, height = map(int, lir.lir(mask.astype(np.bool_), contours[0][:, 0, :]))
    # Keep the first even coordinate strictly above origin + 1 on each axis.
    rows = slice(((y + 1) // 2 + 1) * 2, (y + height - 1) // 2 * 2)
    cols = slice(((x + 1) // 2 + 1) * 2, (x + width - 1) // 2 * 2)
    return rows, cols


def warp_hr_crop(image, matrix, reference_shape, crop, scale=8):
    import cv2
    import numpy as np

    height, width = reference_shape
    scaled = np.diag([float(scale), float(scale), 1.0]) @ matrix
    scaled[:2, 2] += 0.5 * (scale - 1)
    warped = cv2.warpPerspective(image, scaled, (width * scale, height * scale))
    rows, cols = crop
    return warped[rows.start * scale:rows.stop * scale, cols.start * scale:cols.stop * scale]


def translate_matrix(matrix, y0, x0):

    import numpy as np

    T_in = np.array([
        [1, 0, x0],
        [0, 1, y0],
        [0, 0, 1]
    ], dtype=np.float32)

    T_out = np.array([
        [1, 0, -x0],
        [0, 1, -y0],
        [0, 0, 1]
    ], dtype=np.float32)

    return T_out @ matrix @ T_in


def registration_rmse(ref_points, hr_points, pairs, initial, refined):
    """Evaluate both HR-to-LR transforms on the same SIFT-selected correspondences."""
    import cv2
    import numpy as np

    matches = []
    for pair in pairs:
        if len(pair) == 2:
            best, second = pair
            if best.distance < 0.75 * second.distance:
                matches.append(best)
    if len(matches) < 4:
        raise ValueError("Insufficient SIFT matches")

    ref_xy = np.float32([ref_points[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    hr_xy = np.float32([hr_points[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(hr_xy, initial.astype(np.float64))
    errors = np.linalg.norm(projected - ref_xy, axis=2).ravel()
    median = np.median(errors)
    mad = np.median(np.abs(errors - median))
    inliers = np.isfinite(errors) & (errors <= max(3.0, median + 3.0 * 1.4826 * mad))
    if np.count_nonzero(inliers) < 4:
        raise ValueError("Insufficient inlier correspondences")

    results = []
    for matrix in (initial, refined):
        projected = cv2.perspectiveTransform(hr_xy[inliers], matrix.astype(np.float64))
        distances = np.linalg.norm(projected - ref_xy[inliers], axis=2).ravel()
        if not np.isfinite(distances).all():
            raise ValueError("Non-finite reprojection errors")
        results.append(float(np.sqrt(np.mean(distances.astype(np.float64) ** 2))))
    return tuple(results)


def gradient_correlation(ref_rgb, hr_blur, H_sift, H_ecc):
    """Sobel gradient cosine similarity for SIFT and ECC on identical LR pixels.

    Uses the same LR reference and blurred HR image as cross-camera registration.
    Higher values indicate more consistent gradient direction and structure.
    """
    import cv2
    import numpy as np

    ref = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hr = cv2.cvtColor(hr_blur, cv2.COLOR_RGB2GRAY).astype(np.float32)
    height, width = ref.shape

    def warp(H):
        M = np.asarray(H, dtype=np.float64)
        aligned = cv2.warpPerspective(hr, M, (width, height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpPerspective(np.ones(hr.shape, dtype=np.uint8), M,
                                   (width, height), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return aligned, mask

    aligned_sift, mask_sift = warp(H_sift)
    aligned_ecc, mask_ecc = warp(H_ecc)

    # Both methods are evaluated on the same valid pixels. Erosion avoids
    # border/interpolation artifacts entering the 3x3 Sobel neighborhoods.
    valid = cv2.erode(mask_sift & mask_ecc,
                      np.ones((5, 5), dtype=np.uint8)) > 0
    if np.count_nonzero(valid) < 100:
        raise ValueError("Insufficient common image overlap")

    ref_gx = cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=3)
    ref_gy = cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=3)
    ref_x = ref_gx[valid].astype(np.float64)
    ref_y = ref_gy[valid].astype(np.float64)
    ref_energy = float(np.dot(ref_x, ref_x) + np.dot(ref_y, ref_y))
    if ref_energy < 1e-8:
        raise ValueError("Insufficient reference gradient energy")

    def correlation(aligned):
        gx = cv2.Sobel(aligned, cv2.CV_32F, 1, 0, ksize=3)[valid].astype(np.float64)
        gy = cv2.Sobel(aligned, cv2.CV_32F, 0, 1, ksize=3)[valid].astype(np.float64)
        energy = float(np.dot(gx, gx) + np.dot(gy, gy))
        if energy < 1e-8:
            raise ValueError("Insufficient warped gradient energy")
        return float((np.dot(ref_x, gx) + np.dot(ref_y, gy)) /
                     np.sqrt(ref_energy * energy))

    return correlation(aligned_sift), correlation(aligned_ecc)


def update_registration_csv(path, records):
    """Replace the CSV atomically after each completed scene."""
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(dir=path.parent) as temporary:
        from pathlib import Path

        staging = Path(temporary) / path.name
        with staging.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=METRIC_COLUMNS)
            writer.writeheader()
            for name in sorted(records):
                row = dict(records[name])
                for column in METRIC_COLUMNS[1:]:
                    value = row.get(column)
                    row[column] = "" if value is None or value == "" else "%.6f" % float(value)
                writer.writerow(row)
        staging.replace(path)
