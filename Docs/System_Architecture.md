# Karate Posture Evaluation System Architecture

## 1\. Introduction

The Karate Posture Evaluation System is a classical digital image processing (DIP) application designed to evaluate a user's karate stances by comparing geometric posture features extracted from a user-supplied image against WKF (World Karate Federation) reference stances. The system is implemented without machine learning or pose-estimation libraries and produces per-image metrics, qualitative feedback, visual reports (PNG), and batch CSV exports for dataset evaluation.

Primary goals:

* Provide an interpretable, deterministic posture evaluation pipeline.
* Support single-image interactive analysis and automated batch evaluation.
* Produce visual reports and CSV summaries suitable for academic submission and grading.

## 2\. Overall Architecture

The system is organized into the following logical layers. Each layer focuses on a specific set of responsibilities and exposes inputs/outputs used by downstream layers.

1. Input Layer
2. Image Preprocessing Layer
3. ROI Extraction Layer
4. Image Segmentation Layer
5. Silhouette Generation Layer
6. Geometric Feature Extraction Layer
7. WKF Reference Comparison Layer
8. Stability Evaluation Layer
9. Feedback Generation Layer
10. Visualization and Reporting Layer

## 3\. Architectural Workflow

The following describes the data flow and responsibilities for each layer.

### 3.1 Input Layer

* Purpose: Acquire image data for analysis and provide CLI or interactive entry points.
* Input: File path supplied via CLI (`--image`) or interactive prompt.
* Processing: Validate path, read image using OpenCV (`cv2.imread`), produce a working copy.
* Output: BGR image (`input\_bgr`) and a copy (`input\_bgr\_raw`) for annotation and further processing.

Key functions: `load\_input\_image()` in `Karate\_Posture.py`.

### 3.2 Image Preprocessing Layer

* Purpose: Prepare image for robust segmentation by removing confounding background colors and improving contrast.
* Input: Raw BGR image from Input Layer.
* Processing:

  * Remove tatami/mat colors (red/blue) in HSV color space using color-range masking.
  * Apply CLAHE on the LAB color L-channel for local contrast enhancement.
* Output: Cleaned BGR image suitable for grayscale conversion and thresholding; an RGB copy for visualization.

Key functions: `remove\_mat\_colors()`, `enhance\_image()`.

### 3.3 ROI Extraction Layer

* Purpose: Restrict processing to the lower-body region where pelvis and feet appear, reducing noise and computation.
* Input: Enhanced BGR image.
* Processing: Convert to grayscale and compute vertical ROI start `roi\_y = int(height \* roi\_ratio)`; slice image and grayscale into ROI and full-frame variants.
* Output: Full grayscale image, ROI grayscale slice, ROI BGR slice, and `roi\_y` offset.

Key function: `compute\_roi()`.

### 3.4 Image Segmentation Layer

* Purpose: Produce binary masks separating foreground (subject) from background.
* Input: Blurred grayscale images (full and ROI).
* Processing:

  * Gaussian blur to reduce noise.
  * Otsu thresholding for automatic binary segmentation.
* Output: Binary masks for full frame and ROI.

Key functions: `denoise\_images()`, `otsu\_threshold()`.

### 3.5 Silhouette Generation Layer

* Purpose: Refine binary masks into a single, robust silhouette representing the subject's lower body.
* Input: Binary ROI mask.
* Processing:

  * Morphological closing and opening to fill holes and remove noise.
  * Larger closing and connected-component area filtering to retain only sufficiently large components.
* Output: Final silhouette mask and optionally a cleaned full-frame mask.

Key functions: `clean\_masks()`, `build\_silhouette()`.

### 3.6 Geometric Feature Extraction Layer

* Purpose: Extract geometric landmarks and compute metrics required for stance evaluation.
* Input: Silhouette mask and ROI mapping (`roi\_y`).
* Processing:

  * Contour extraction for diagnostics.
  * Pelvis center estimation by scanning first occupied silhouette row and computing mean x.
  * Foot point detection by splitting silhouette pixels about the waist x-coordinate, selecting bottom-band extreme x points per side.
  * Compute stance width (Euclidean distance), silhouette height, leg spread angle (vector dot product and arccos), pelvis lateral offset (horizontal distance from feet midpoint), support triangle area (shoelace/determinant formula normalized by height squared), pelvis height ratio (normalized vertical position), and leg symmetry ratio (distance ratio).
* Output: Coordinates for pelvis and feet, numeric metrics (stance width, normalized metrics, angles, areas), masks and annotated visualization images.

Key functions: `extract\_contours()`, `find\_pelvis\_center()`, `detect\_feet()`, `calculate\_leg\_spread\_angle()`, `calculate\_silhouette\_height()`, `compute\_pelvis\_height\_ratio()`, `compute\_pelvis\_lateral\_offset()`, `compute\_support\_triangle\_area()`, `compute\_leg\_symmetry\_ratio()`.

### 3.7 WKF Reference Comparison Layer

* Purpose: Compare user metrics against WKF reference stance metrics to compute differences.
* Input: Metric values from the Geometric Feature Extraction Layer for both the user image and the reference image.
* Processing: Normalize stance width and offsets by silhouette height where applicable, compute signed differences for each metric.
* Output: Metric differences and per-metric comparative rows (reference value, user value, difference).

Key functions: `get\_reference\_path()`, `compare\_stance\_width()`, `compare\_leg\_spread\_angle()`, `evaluate\_pelvis\_offset()`, `evaluate\_support\_triangle\_area()`, `evaluate\_pelvis\_height\_ratio()`, `evaluate\_leg\_symmetry\_ratio()`.

### 3.8 Stability Evaluation Layer

* Purpose: Aggregate per-metric deviations into a single, interpretable Stability Index.
* Input: Signed metric differences for the chosen set of metrics.
* Processing: For each metric, compute a bounded penalty = min(abs(diff)/threshold, 1.0), multiply by per-metric weight, sum weighted penalties, compute stability index as 100\*(1 - weighted\_penalty / total\_weight), and clamp to \[0,100].
* Output: Stability Index (percentage) and classification label (Good/Warning/Critical).

Key functions: `compute\_stability\_index()`, `evaluate\_stability\_index()`.

### 3.9 Feedback Generation Layer

* Purpose: Translate numeric deviations and stability scores into textual feedback and status categories for user guidance.
* Input: Per-metric comparison rows and stability index.
* Processing: Map deviation magnitudes to discrete categories using implemented threshold bands; generate context-specific advice strings.
* Output: `metrics\_data` list of dictionaries with fields {feature, reference, user, difference, status, feedback}.

Key functions: comparison/evaluation functions listed in Section 3.7 and `evaluate\_stability\_index()`.

### 3.10 Visualization and Reporting Layer

* Purpose: Present results visually (multi-panel diagnostics), save a composed PNG report, and support batch CSV exports.
* Input: Annotated images, `metrics\_data`, stability classification, and paths for saving.
* Processing: Compose side-by-side images and metrics table using Matplotlib, save figures as timestamped PNG, and write CSVs for batch runs.
* Output: On-screen figures (interactive), PNG evaluation report files, CSV files (`evaluation\_results.csv`, `evaluation\_summary.csv`).

Key functions: `draw\_contours()`, `draw\_stance\_output()`, `show\_plots()`, `draw\_metrics\_table()`, `show\_template\_figure()`, `save\_evaluation\_report()`. Batch export in `batch\_evaluation.py` writes CSVs.

## 4\. Module Description

### 4.1 `Karate\_Posture.py`

**Responsibility:** Core implementation of the image-processing pipeline, single-image interactive CLI, metric computation, visualization, and report saving.

Key behaviors:

* Accepts CLI args or interactive input.
* Runs deterministic DIP pipeline (color removal → enhance → ROI → blur → otsu → cleanup → silhouette → geometry).
* Compares user against reference if available and computes `metrics\_data` and `stability\_index`.
* Produces on-screen diagnostic plots and saves a PNG evaluation report.

Important implementation notes:

* A `--ref-dir` CLI argument is supported to locate reference images; default is the `Reference Stances` folder next to the script.
* Error handling: missing or unreadable images throw descriptive errors; missing reference images are warned and do not crash the interactive flow.

### 4.2 `batch\_evaluation.py`

**Responsibility:** Automate evaluation over a directory of test images, produce a consolidated CSV of per-image metrics and a summary statistics CSV.

Key behaviors:

* Enumerates images in `Test Dataset/`.
* For each image, runs `run\_analysis()` from `Karate\_Posture.py` to extract metrics.
* Attempts to identify stance from filename and runs reference comparison if reference images exist.
* Writes `evaluation\_results.csv` with per-image metrics and `evaluation\_summary.csv` with mean/min/max/std values.

Important implementation notes:

* Batch mode is robust to individual failures: exceptions for an image are logged and processing continues.
* Normalization and metric calculations mirror `Karate\_Posture.py` so single-image and batch outputs are consistent.

## 5\. Feature Extraction Architecture

This section details how each key feature is computed and the downstream usage of each metric.

### 5.1 Pelvis Center

* Computation: Find the first row (from the top) in the silhouette that contains foreground pixels in ROI coordinates; compute mean x of foreground pixels in that row; convert to full-image coordinates by adding `roi\_y`.
* Use: Serves as the geometric origin for vectors to each foot and as a reference for pelvis height ratio and lateral offset.

### 5.2 Foot Points

* Computation: Split silhouette pixels by pelvis x (waist\_x) into left and right sets; identify bottom-most band and select extreme x within that band (left: min x, right: max x) and convert to full-image coordinates.
* Use: Defines the base-of-support endpoints; used in stance width, triangle area, symmetry, and offset calculations.

### 5.3 Stance Width

* Computation: Euclidean distance between detected left and right foot points: ( D = \\sqrt{(x\_R - x\_L)^2 + (y\_R - y\_L)^2} ).
* Use: Primary measure of stance breadth; normalized by silhouette height for comparisons.

### 5.4 Leg Spread Angle

* Computation: Vector from pelvis to each foot; angle computed via dot product and arccos: ( \\theta = \\arccos\\left(\\frac{\\mathbf{v}\_L\\cdot\\mathbf{v}\_R}{|\\mathbf{v}\_L|,|\\mathbf{v}\_R|}\\right) ), reported in degrees.
* Use: Describes openness of stance and is compared to reference angle to provide corrective feedback.

### 5.5 Pelvis Lateral Offset

* Computation: Horizontal distance between pelvis x and feet midpoint x: ( O = |x\_P - \\frac{x\_L+x\_R}{2}| ). Often normalized by silhouette height.
* Use: Indicates lateral balance of pelvis relative to foot base.

### 5.6 Support Triangle Area

* Computation: Exact triangle area via determinant (shoelace) formula normalized by silhouette height squared: ( A = \\frac{1}{2}|x\_P(y\_L - y\_R) + x\_L(y\_R - y\_P) + x\_R(y\_P - y\_L)| ), then ( A\_{norm} = A/H^2 ).
* Use: Measure of base-of-support size relative to subject scale; informs stability evaluation.

### 5.7 Pelvis Height Ratio

* Computation: Vertical position of pelvis relative to silhouette top divided by silhouette height: ( R\_P = (y\_P - y\_{top}) / H ).
* Use: Gives a normalized indicator of vertical pelvis placement indicating posture depth.

### 5.8 Leg Symmetry Ratio

* Computation: Ratio of pelvis-to-left distance to pelvis-to-right distance: ( S = \\frac{|\\mathbf{v}\_L|}{|\\mathbf{v}\_R|} ).
* Use: Quantifies left-right asymmetry; feeds into stability calculation and feedback.

### 5.9 Stability Index

* Computation: For each metric compute bounded penalty pi = min(|di| / Ti, 1.0) then weighted sum Pw = sum(wi \* pi); stability index SI = 100\*(1 - Pw / sum(wi)).
* Use: Aggregate measure for grading and succinct feedback; used to compute Good/Warning/Critical labels.

## 6\. Output Architecture

### 6.1 Evaluation Table

* Composition: Per-metric rows with reference value, user value, difference, status, and textual feedback.
* Generation: Built from `metrics\_data` and rendered into figures and also saved into reports.

### 6.2 Feedback Generation

* Mechanism: Per-metric thresholds map numeric deviations to `Good`, `Warning`, or `Critical` with short advice text. Stability index receives analogous categorization.
* Rationale: Provide actionable and interpretable feedback for users without exposing raw numeric thresholds.

### 6.3 PNG Report Export

* Format: Single PNG with side-by-side reference and user images and the metrics table in a lower panel.
* Generation: `save\_evaluation\_report()` builds the Matplotlib figure and writes a timestamped PNG to disk.

### 6.4 CSV Batch Evaluation Export

* Files: `evaluation\_results.csv` (per-image metrics) and `evaluation\_summary.csv` (mean/min/max/std per metric).
* Generation: `batch\_evaluation.py` runs over `Test Dataset/` and writes CSV outputs for downstream analysis.

## 7\. System Workflow Diagram Reference

\[Insert Workflow Diagram Here]

(Recommended diagram: a flowchart showing layers 1→10, with `Karate\_Posture.py` and `batch\_evaluation.py` call relationships annotated. Include data artifacts: PNG, CSV.)

## 8\. Evaluation Framework Diagram Reference

\[Insert Evaluation Framework Diagram Here]

(Recommended diagram: architecture showing metric computations feeding into stability aggregation, depiction of weights/thresholds, and outputs mapped to report/CSV.)

## 9\. Operational Considerations

* Dependencies: OpenCV (cv2), NumPy, Matplotlib.
* Execution modes: Interactive single-image mode (`Karate\_Posture.py`) and automated batch mode (`batch\_evaluation.py`).
* Portability: `batch\_evaluation.py` uses project-relative `Reference Stances` path; `Karate\_Posture.py` supports `--ref-dir` to avoid hardcoded absolute paths.
* Failure modes: Absent or unreadable images will be reported; missing silhouettes produce NaN metrics; batch mode continues on individual failures.

## 10\. Extensibility and Future Work

* Replace pelvis/foot heuristics with a landmark detector (if ML permitted) to improve robustness.
* Add configuration file for thresholds and weights to allow calibration without code changes.
* Implement automated unit tests for geometric functions and integration tests for end-to-end pipeline.
* Produce HTML/PDF reports with richer metadata and multi-image galleries for batch analysis.

## 11\. Appendix — Key Files

* `Karate\_Posture.py` — Core pipeline and single-image workflow.
* `batch\_evaluation.py` — Batch runner and CSV exporter.
* `Reference Stances/` — WKF reference images named as `REF\_{stance}.png`.
* `Test Dataset/` — Directory containing test/user images used for batch evaluation.
* `docs/` — Generated documentation (`Equations\_and\_Mathematical\_Analysis.Rmd`, `Function\_Documentation.txt`, `Viva\_Preparation\_Notes.txt`, `System\_Architecture.md`).



\---



