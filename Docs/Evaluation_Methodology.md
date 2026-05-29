# Evaluation Methodology

## 1. Introduction

This document describes the evaluation methodology employed by the Karate Posture Evaluation System. The system uses classical digital image processing techniques (no machine learning) to extract geometric features from a user-supplied image and compare them to WKF reference stances. The methodology defines how features are computed, normalized, compared, aggregated into a Stability Index, and converted into textual recommendations. The descriptions include exact formulas and thresholds as implemented in the codebase.

---

## 2. Feature Extraction Process

For each feature below, the Purpose, Calculation Method (exact formula), and Importance in stance evaluation are provided. All coordinates use image pixel space with origin at the top-left, x increasing rightwards and y increasing downwards.

### 2.1 Pelvis Center
- Purpose: Provide a central anatomical reference used for vector formation to each foot and for computing pelvis-related metrics.
- Calculation Method: The pelvis center is estimated by scanning the silhouette mask from top to bottom and selecting the first row that contains foreground pixels. The pelvis x-coordinate is the mean x of foreground pixels in that row; the full-image y-coordinate is the row index plus the ROI vertical offset `roi_y`.
- Importance: Acts as the geometric origin for angle, symmetry, and offset calculations; its accuracy directly influences downstream metrics.

### 2.2 Foot Points
- Purpose: Determine left and right base-of-support endpoints necessary for distances, area, and symmetry calculations.
- Calculation Method: Using silhouette pixel coordinates (ys, xs) in ROI space and `waist_x` (pelvis x in ROI):
  1. Split pixels into left (xs < waist_x) and right (xs > waist_x).
  2. For each side, find the maximal y (bottom-most) and select candidates in the bottom band (y >= bottom_y - 10).
  3. For left foot select the candidate with minimum x; for right foot select the candidate with maximum x.
  4. Convert chosen y coordinates to full-image coordinates by adding `roi_y`.
- Importance: Foot points define stance width, support triangle, and are critical for reliable symmetry and offset measures.

### 2.3 Stance Width
- Purpose: Measure the Euclidean distance between foot endpoints — primary measure of stance breadth.
- Calculation Method: If left foot (x_L, y_L) and right foot (x_R, y_R):
  - D = sqrt((x_R - x_L)^2 + (y_R - y_L)^2)
- Importance: Higher D indicates a wider stance. Normalization by silhouette height makes D comparable across images.

### 2.4 Leg Spread Angle
- Purpose: Quantifies the included angle at the pelvis between vectors to left and right feet.
- Calculation Method: With vectors v_L = left - pelvis and v_R = right - pelvis:
  - cos(theta) = (v_L . v_R) / (||v_L|| * ||v_R||)
  - theta = arccos( clip(cos(theta), -1, 1) ) in radians
  - theta_deg = theta * (180 / pi)
- Importance: Characterizes leg openness; compared against reference angle to detect under/over spread.

### 2.5 Pelvis Lateral Offset
- Purpose: Measure horizontal displacement of pelvis from the midpoint of feet — indicator of lateral imbalance.
- Calculation Method:
  - x_mid = (x_L + x_R) / 2
  - O = |x_P - x_mid|
  - Normalized: O_norm = O / H (where H is silhouette height)
- Importance: Larger offset indicates pelvis shifted away from center of the support base.

### 2.6 Support Triangle Area
- Purpose: Compute base-of-support area formed by pelvis and both feet; normalized for scale invariance.
- Calculation Method (shoelace/determinant):
  - A = 0.5 * | x_P (y_L - y_R) + x_L (y_R - y_P) + x_R (y_P - y_L) |
  - A_norm = A / H^2
- Importance: Larger normalized area generally correlates with greater static stability.

### 2.7 Pelvis Height Ratio
- Purpose: Quantify pelvis vertical placement within silhouette bounds as a normalized fraction.
- Calculation Method:
  - H = y_bottom - y_top (silhouette height)
  - R_P = (y_P - y_top) / H
- Importance: Indicates posture depth and vertical alignment relative to reference.

### 2.8 Leg Symmetry Ratio
- Purpose: Measure left-right distance symmetry from pelvis to each foot.
- Calculation Method:
  - left_dist = ||v_L||, right_dist = ||v_R||
  - S = left_dist / right_dist (returns 0 if right_dist is 0 to avoid division by zero)
- Importance: Values near 1 indicate balanced limb distribution; deviations indicate asymmetry.

---

## 3. Reference Comparison Method

### 3.1 WKF Reference Image Selection
Reference images are stored in `Reference Stances/` with filenames of the form `REF_{stance}.png`. For a given user-selected stance, the code constructs the matching reference path and attempts to analyze the reference image with the same pipeline.

### 3.2 User Image Processing
The user image is processed by `run_analysis()` which executes the full pipeline (preprocessing, ROI, thresholding, silhouette building, contour extraction, pelvis/feet detection, and metric computation) returning both visualization images and numeric metrics.

### 3.3 Feature Vector Generation
For both reference and user images the pipeline yields a consistent set of features: normalized stance width, leg spread angle (degrees), pelvis lateral offset (optionally normalized by H), support triangle normalized area, pelvis height ratio, and leg symmetry ratio. These are collected into a feature vector per image.

### 3.4 Feature Matching Process
Matching is a direct numeric comparison: for each metric, compute the user value and reference value and derive signed differences (user - reference). Normalizations are applied (e.g., divide widths and offsets by silhouette height) to ensure scale invariance.

---

## 4. Difference Calculation

### 4.1 Reference Value vs User Value
The code records both the reference and user values in the `metrics_data` table and uses their difference as the primary quantity for evaluation.

### 4.2 Difference Computation
For a metric m, difference is computed as:
- diff_m = user_m - ref_m
The implementation often works with normalized values (e.g., normalized stance width) and computes absolute or signed differences depending on context.

### 4.3 Percentage Deviation
When appropriate (e.g., stance width), a percentage deviation is computed as:
- percent_diff = (|diff| / ref_norm) * 100  (guarding against ref_norm=0)
Used in `compare_stance_width()` to determine Good/Warning/Critical categories.

### 4.4 Normalization Methods
Normalization used in the code:
- Width and offsets: divided by silhouette height H (to produce unitless comparators)
- Triangle area: divided by H^2
- Pelvis height: expressed as fraction (y_P - y_top) / H
These normalizations remove scale effects from camera distance and image resolution.

---

## 5. Stability Index Methodology

### 5.1 Inputs Used
The Stability Index aggregates deviations for the following metrics (as implemented):
- stance_width (normalized)
- leg_spread (degrees)
- pelvis_offset (normalized or units as computed)
- triangle_area (normalized)
- pelvis_height (ratio)
- leg_symmetry (ratio)

Implementation expects a dictionary `metrics_dictionary` with keys:
`stance_width`, `leg_spread`, `pelvis_offset`, `triangle_area`, `pelvis_height`, `leg_symmetry`
Each value is the signed difference (user minus reference) appropriate for the metric.

### 5.2 Weighting Mechanism
Per-metric weights used in the implementation (sum = 1.0):
- w_stance_width = 0.20
- w_leg_spread  = 0.20
- w_pelvis_offset = 0.15
- w_triangle_area = 0.15
- w_pelvis_height = 0.15
- w_leg_symmetry = 0.15

### 5.3 Thresholds (for penalty scaling)
Per-metric thresholds Ti used to convert absolute differences into a bounded penalty pi:
- T_stance_width = 0.25
- T_leg_spread = 25.0 (degrees)
- T_pelvis_offset = 35.0
- T_triangle_area = 0.18
- T_pelvis_height = 0.18
- T_leg_symmetry = 0.25

### 5.4 Calculation Procedure
For each metric i:
1. Compute absolute normalized difference: |d_i|.
2. Compute bounded penalty: p_i = min(|d_i| / T_i, 1.0)
3. Compute weighted penalty contribution: w_i * p_i
After summation:
- weighted_penalty = sum_i w_i * p_i
- total_weight = sum_i w_i (defensive code computes this, typically 1.0)
- stability_index = 100.0 * (1.0 - (weighted_penalty / total_weight))
- stability_index is clamped to [0, 100]

### 5.5 Interpretation of Results
- SI near 100: user metrics closely match reference (high stability).
- SI in middle ranges indicates moderate deviations.
- SI small (close to 0): major deviations in one or more weighted metrics.
The code then applies categorical labels via `evaluate_stability_index()`:
- SI >= 80 → Good
- 60 <= SI < 80 → Warning
- SI < 60 → Critical

The weights reflect relative importance; thresholds determine sensitivity.

---

## 6. Status Classification

The system maps numeric deviations to three qualitative classes: Good, Warning, Critical. The selection rules vary per metric but follow the same pattern of absolute-difference thresholds. Representative rules (from code):

- Stance Width (percent deviation):
  - Good: percent_diff <= 10%
  - Warning: percent_diff <= 25%
  - Critical: percent_diff > 25%

- Leg Spread Angle (degrees):
  - Good: abs_diff <= 10°
  - Warning: abs_diff <= 25°
  - Critical: abs_diff > 25°

- Pelvis Lateral Offset:
  - Good: abs_diff <= 15
  - Warning: abs_diff <= 35
  - Critical: abs_diff > 35

- Support Triangle Area:
  - Good: abs_diff <= 0.08
  - Warning: abs_diff <= 0.18
  - Critical: abs_diff > 0.18

- Pelvis Height Ratio: same thresholds as triangle area.

- Leg Symmetry Ratio: deviation-based thresholds (Good <= 0.20, Warning <= 0.40, else Critical)

These thresholds are heuristics chosen for interpretability and tuned empirically for the dataset; they are implemented directly in the comparison/evaluation functions.

---

## 7. Feedback Generation

Feedback is generated from three sources:
1. Metric values (absolute or normalized) recorded for reference and user.
2. Deviation values (signed/absolute differences) used to determine the status class.
3. The Stability Index which aggregates deviations into a single percentage and qualitative label.

Mechanism:
- For each metric, the evaluation function computes status according to thresholds and selects a short feedback string describing corrective action (e.g., "Increase leg spread", "Slightly narrow stance").
- For stability, `evaluate_stability_index()` returns an overall message such as "Stable stance posture" (Good), "Moderate posture instability" (Warning), or "Poor stance stability" (Critical).
- The resulting `metrics_data` list contains structured entries with `feature`, `reference`, `user`, `difference`, `status`, and `feedback` which are rendered in the table and report.

Rationale: rule-based feedback provides actionable guidance and preserves interpretability for academic evaluation.

---

## 8. Batch Evaluation Methodology

### 8.1 Dataset Iteration
`batch_evaluation.py` enumerates files in `Test Dataset/` and filters by supported image extensions. For each image it attempts the following:
- Run `run_analysis()` to extract geometry and metrics.
- Guess the stance name by substring matching of known stance identifiers in the filename.
- If a matching reference image exists, run analysis on the reference and compute differences and the Stability Index.

### 8.2 Metric Collection
For each image, metrics collected include:
- Stance_Width (normalized by silhouette height)
- Leg_Spread_Angle (degrees)
- Pelvis_Lateral_Offset (normalized by silhouette height)
- Support_Triangle_Area (normalized)
- Pelvis_Height_Ratio
- Leg_Symmetry_Ratio
- Stability_Index (if reference exists and comparison completed)

If silhouette/contours are missing for an image metrics may be set to NaN for robustness.

### 8.3 CSV Generation
- Per-image metrics are written to `evaluation_results.csv` with fieldnames matching the metrics above.
- A summary CSV `evaluation_summary.csv` is created containing mean, min, max, and standard deviation for each metric across processed images.

### 8.4 Summary Report Generation
`batch_evaluation.py` also prints per-metric summary statistics to console (Mean/Min/Max/Std) and writes the summary CSV for later analysis.

---

## 9. Evaluation Limitations

The methodology has the following limitations which should be acknowledged:
- **Sensitivity to segmentation quality:** All feature computations rely on accurate silhouette extraction; errors in thresholding or morphological cleanup directly produce erroneous metrics.
- **Viewpoint variation:** The pipeline assumes near-frontal or near-lateral views; oblique camera angles distort angular and distance measures.
- **Lighting conditions:** Extreme illumination or strong shadows can break thresholding; CLAHE mitigates but does not eliminate illumination sensitivity.
- **Reference image dependence:** Performance and meaningfulness of differences depend on the representativeness and correctness of reference images (framing, scale, and pose).
- **Heuristic thresholds and weights:** The thresholds for status classification and weights for the Stability Index are heuristic and dataset-dependent; they require calibration for broader deployment.

---

## 10. Conclusion

The Karate Posture Evaluation System applies well-established classical DIP operations to produce interpretable geometric features and a composite stability assessment. The methodology emphasizes transparency (explicit formulas, thresholds, and weights) and reproducibility for academic evaluation. While robust for controlled datasets, the method requires careful preprocessing and dataset curation to produce reliable results in more varied real-world conditions.


*End of Evaluation Methodology.*
