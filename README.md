# RAW-56 Preprocessing

This is an implementation of the dataset processing for the paper:
> Cotrim, A. N., Engler, C., Barbosa, G., Santos, C. A. N., & Pedrini, H. (2026). A real-world raw 56-frame burst dataset for 8× super-resolution. SSRN. https://doi.org/10.2139/ssrn.7389288

Generate an aligned low-resolution (LR) burst and an 8× high-resolution (HR) target from DNG files captured from [raw56-capture](https://github.com/AndersonCotrim/raw56-capture).

## Requirements

- Python 3.10

```bash
python -m pip install -r requirements.txt
```

## Usage

Organize the input files in one subdirectory per capture. The pipeline uses camera `2` for the low-resolution (LR) burst and camera `7` for the HR reference; other camera IDs are ignored.

```text
captures/
└── 20240111_080439_408/
    ├── IMG_20240111_080441_020_2_0.dng
    ├── IMG_20240111_080441_020_7_0.dng
    └── ...
```

Process all captures or a single capture, specifying an output directory outside the input tree:

```bash
python generate_dataset.py captures processed
python generate_dataset.py captures/20240111_080439_408 processed
```

The first camera `2` frame is the LR reference; the first available camera `7` frame is the HR target.

## Output

Each processed scene is saved in `processed/<capture-name>/`. Here, `N` is the number of LR frames and `H × W` is the LR crop size.

| File | Description |
| --- | --- |
| `lr.npy` | `(N, H, W)` Bayer mosaics (`uint16`), black-level subtracted and shifted by even pixel offsets. |
| `hr.npy` | `(8H, 8W, 3)` aligned linear RGB target (`float32`), black-level subtracted and normalized by 960. |
| `h_norm.npy` | `(N, 3, 3)` residual LR alignment transforms; the reference-frame entry is the identity matrix. |
| `meta_lr.pkl`, `meta_hr.pkl` | Camera calibration (`info`) and EXIF (`exif`) dictionaries from the first LR and HR DNG files. |

The output root also contains `registration_per_scene.csv` with the columns `scene`, `sift_rmse_px`, `ecc_rmse_px`, `grad_corr_sift`, and `grad_corr_ecc`.

## Related Resources

- **Dataset:** [Hugging Face](https://huggingface.co/datasets/andersoncotrim/raw-56-burst-super-resolution) ([DOI](https://doi.org/10.57967/hf/10146))
- **Capture:** [raw56-capture](https://github.com/AndersonCotrim/raw56-capture) — Android RAW burst acquisition.
- **Training and Evaluation:** [SBFBurst](https://github.com/AndersonCotrim/SBFBurst) — baseline training and evaluation scripts.

## Citation

Cotrim, A. N., Engler, C., Barbosa, G., Santos, C. A. N., & Pedrini, H. (2026). *A real-world raw 56-frame burst dataset for 8× super-resolution*. SSRN. [DOI: 10.2139/ssrn.7389288](https://doi.org/10.2139/ssrn.7389288).

## License

The code is distributed under the Apache License 2.0. See `LICENSE`.
