import os
import csv
import math
import numpy as np
from statistics import mean, stdev

# Reuse existing pipeline and helpers from Karate_Posture.py
from Karate_Posture import (
    run_analysis,
    calculate_leg_spread_angle,
    compute_pelvis_lateral_offset,
    compute_support_triangle_area,
    compute_pelvis_height_ratio,
    compute_leg_symmetry_ratio,
    compute_stability_index,
    get_reference_path,
)

# Configuration
ROOT_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(ROOT_DIR, "Test Dataset")
REF_DIR = os.path.join(ROOT_DIR, "Reference Stances")
OUTPUT_CSV = os.path.join(ROOT_DIR, "evaluation_results.csv")
SUMMARY_CSV = os.path.join(ROOT_DIR, "evaluation_summary.csv")

# Known stance names (same as main)
STANCE_NAMES = [
    "neko_ashi_dachi",
    "zenkutsu_dachi_front",
    "siko_dachi",
    "han_zenkutsu_dachi_side",
    "heiko_dachi",
    "han_zenkutsu_dachi_front",
    "zenkutsu_dachi_side",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def guess_stance_from_filename(filename):
    name = filename.lower()
    for s in STANCE_NAMES:
        if s in name:
            return s
    return None


def safe_div(a, b):
    try:
        return a / b
    except Exception:
        return float('nan')


def normalize_by_height(value, silhouette_height):
    if silhouette_height is None or silhouette_height <= 0:
        return float('nan')
    return safe_div(float(value), float(silhouette_height))


def collect_metrics_for_image(image_path):
    try:
        # run_analysis does not require a valid stance_name for per-image analysis
        result = run_analysis(image_path, "unknown")
    except Exception as e:
        raise

    if not result:
        raise RuntimeError("run_analysis returned no result")

    metrics = {}
    has = result.get("has_contours", False)

    if not has:
        # If silhouette/contour missing, set NaNs
        metrics["Stance_Width"] = float('nan')
        metrics["Leg_Spread_Angle"] = float('nan')
        metrics["Pelvis_Lateral_Offset"] = float('nan')
        metrics["Support_Triangle_Area"] = float('nan')
        metrics["Pelvis_Height_Ratio"] = float('nan')
        metrics["Leg_Symmetry_Ratio"] = float('nan')
    else:
        silhouette_height = result.get("silhouette_height", 0)

        stance_width = result.get("stance_width", 0.0)
        metrics["Stance_Width"] = float(normalize_by_height(stance_width, silhouette_height))

        leg_angle = calculate_leg_spread_angle(
            result.get("center_node"),
            result.get("left_foot"),
            result.get("right_foot"),
        )
        metrics["Leg_Spread_Angle"] = float(leg_angle)

        pelvis_offset = compute_pelvis_lateral_offset(
            result.get("center_node"),
            result.get("left_foot"),
            result.get("right_foot"),
        )
        metrics["Pelvis_Lateral_Offset"] = float(normalize_by_height(pelvis_offset, silhouette_height))

        support_area = compute_support_triangle_area(
            result.get("center_node"),
            result.get("left_foot"),
            result.get("right_foot"),
            silhouette_height,
        )
        metrics["Support_Triangle_Area"] = float(support_area)

        pelvis_ratio = compute_pelvis_height_ratio(
            result.get("center_node"),
            result.get("silhouette_mask"),
        )
        metrics["Pelvis_Height_Ratio"] = float(pelvis_ratio)

        leg_sym = compute_leg_symmetry_ratio(
            result.get("center_node"),
            result.get("left_foot"),
            result.get("right_foot"),
        )
        metrics["Leg_Symmetry_Ratio"] = float(leg_sym)

    return result, metrics


def main():
    rows = []
    failures = []

    files = sorted(os.listdir(TEST_DATA_DIR)) if os.path.isdir(TEST_DATA_DIR) else []

    for fname in files:
        _, ext = os.path.splitext(fname)
        if ext.lower() not in IMAGE_EXTS:
            continue

        path = os.path.join(TEST_DATA_DIR, fname)
        print(f"Processing: {fname}")
        try:
            result, metrics = collect_metrics_for_image(path)
        except Exception as e:
            print(f"Failed: {fname} -> {e}")
            failures.append(fname)
            continue

        # Try to find reference stance by filename
        stance = guess_stance_from_filename(fname)
        stability_idx = float('nan')

        if stance:
            ref_path = get_reference_path(stance, REF_DIR)
            if os.path.exists(ref_path):
                try:
                    ref_res = run_analysis(ref_path, stance)
                    # compute metrics differences for stability index
                    # normalized width
                    ref_norm = normalize_by_height(
                        ref_res.get("stance_width", 0.0),
                        ref_res.get("silhouette_height", 0),
                    )
                    user_norm = normalize_by_height(
                        result.get("stance_width", 0.0),
                        result.get("silhouette_height", 0),
                    )

                    ref_angle = calculate_leg_spread_angle(
                        ref_res.get("center_node"),
                        ref_res.get("left_foot"),
                        ref_res.get("right_foot"),
                    )
                    user_angle = calculate_leg_spread_angle(
                        result.get("center_node"),
                        result.get("left_foot"),
                        result.get("right_foot"),
                    )

                    ref_offset = compute_pelvis_lateral_offset(
                        ref_res.get("center_node"),
                        ref_res.get("left_foot"),
                        ref_res.get("right_foot"),
                    )
                    user_offset = compute_pelvis_lateral_offset(
                        result.get("center_node"),
                        result.get("left_foot"),
                        result.get("right_foot"),
                    )
                    ref_offset = normalize_by_height(ref_offset, ref_res.get("silhouette_height", 0))
                    user_offset = normalize_by_height(user_offset, result.get("silhouette_height", 0))

                    ref_area = compute_support_triangle_area(
                        ref_res.get("center_node"),
                        ref_res.get("left_foot"),
                        ref_res.get("right_foot"),
                        ref_res.get("silhouette_height", 1),
                    )
                    user_area = compute_support_triangle_area(
                        result.get("center_node"),
                        result.get("left_foot"),
                        result.get("right_foot"),
                        result.get("silhouette_height", 1),
                    )

                    ref_pelvis = compute_pelvis_height_ratio(
                        ref_res.get("center_node"),
                        ref_res.get("silhouette_mask"),
                    )
                    user_pelvis = compute_pelvis_height_ratio(
                        result.get("center_node"),
                        result.get("silhouette_mask"),
                    )

                    ref_leg_sym = compute_leg_symmetry_ratio(
                        ref_res.get("center_node"),
                        ref_res.get("left_foot"),
                        ref_res.get("right_foot"),
                    )
                    user_leg_sym = compute_leg_symmetry_ratio(
                        result.get("center_node"),
                        result.get("left_foot"),
                        result.get("right_foot"),
                    )

                    diffs = {
                        "stance_width": user_norm - ref_norm,
                        "leg_spread": user_angle - ref_angle,
                        "pelvis_offset": user_offset - ref_offset,
                        "triangle_area": user_area - ref_area,
                        "pelvis_height": user_pelvis - ref_pelvis,
                        "leg_symmetry": user_leg_sym - ref_leg_sym,
                    }
                    stability_idx = compute_stability_index(diffs)
                except Exception as e:
                    print(f"Reference comparison failed for {fname}: {e}")
                    stability_idx = float('nan')
            else:
                print(f"Reference image not found for stance '{stance}' (expected {ref_path})")

        row = {
            "Image_Name": fname,
            "Stance_Width": metrics.get("Stance_Width", float('nan')),
            "Leg_Spread_Angle": metrics.get("Leg_Spread_Angle", float('nan')),
            "Pelvis_Lateral_Offset": metrics.get("Pelvis_Lateral_Offset", float('nan')),
            "Support_Triangle_Area": metrics.get("Support_Triangle_Area", float('nan')),
            "Pelvis_Height_Ratio": metrics.get("Pelvis_Height_Ratio", float('nan')),
            "Leg_Symmetry_Ratio": metrics.get("Leg_Symmetry_Ratio", float('nan')),
            "Stability_Index": float(stability_idx) if not math.isnan(stability_idx) else float('nan'),
        }
        rows.append(row)

    # Write CSV
    fieldnames = [
        "Image_Name",
        "Stance_Width",
        "Leg_Spread_Angle",
        "Pelvis_Lateral_Offset",
        "Support_Triangle_Area",
        "Pelvis_Height_Ratio",
        "Leg_Symmetry_Ratio",
        "Stability_Index",
    ]

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Summary statistics
    stats = {}
    for key in fieldnames[1:]:
        vals = [r[key] for r in rows if not (r[key] is None or (isinstance(r[key], float) and math.isnan(r[key])))]
        if len(vals) == 0:
            stats[key] = (float('nan'), float('nan'), float('nan'), float('nan'))
        else:
            arr = np.array(vals, dtype=float)
            stats[key] = (float(np.mean(arr)), float(np.min(arr)), float(np.max(arr)), float(np.std(arr, ddof=0)))

    # Print summary
    for key, (m, mn, mx, sd) in stats.items():
        print("# ==================================================")
        print(key.replace('_', ' ').upper())
        print(f"Mean: {m:.2f}" if not math.isnan(m) else "Mean: NA")
        print(f"Min : {mn:.2f}" if not math.isnan(mn) else "Min : NA")
        print(f"Max : {mx:.2f}" if not math.isnan(mx) else "Max : NA")
        print(f"Std : {sd:.2f}" if not math.isnan(sd) else "Std : NA")
        print("")

    # Optional summary CSV
    with open(SUMMARY_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "Min", "Max", "Std"])
        for key, (m, mn, mx, sd) in stats.items():
            writer.writerow([key, m, mn, mx, sd])

    if failures:
        print("Processing completed with failures for files:")
        for fn in failures:
            print(f" - {fn}")


if __name__ == '__main__':
    main()
