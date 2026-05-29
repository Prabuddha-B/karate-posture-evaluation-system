# Dataset Description

## 1. Overview

This dataset supports the "Karate Posture Evaluation System" — a classical digital image processing pipeline for assessing karate stances by comparing user images against WKF (World Karate Federation) reference stances. The dataset is composed of two parts:

- **Reference Stances**: Canonically framed and labeled WKF reference images used as ground-truth posture exemplars.
- **Test Dataset**: User or test images captured under diverse conditions used to validate the pipeline and compute evaluation metrics.

The dataset is intended for geometric feature extraction, reference comparison, and stability assessment; it is not for training machine learning models.

---

## 2. Reference Stance Dataset

The Reference Stances directory contains one canonical image per stance named according to the convention `REF_{stance}.png`. Each reference image represents the desired posture geometry for a specific WKF stance. The system uses these images to compute reference geometric features (stance width, leg spread angle, pelvis position, support triangle area, etc.) against which user images are compared.

Reference stances included in the project:

- **Heiko Dachi**
  - Purpose in karate: A neutral parallel stance used for basic posture and transitions; feet are typically shoulder-width and parallel.
  - Usage: Provides baseline values for stance width and leg symmetry for neutral posture comparisons.

- **Siko Dachi**
  - Purpose in karate: A wider, more rooted stance (often spelled "Shiko Dachi"), used for stable grounding and certain techniques.
  - Usage: Serves as a reference for wide stances, informing expected stance width, support area, and pelvis placement.

- **Neko Ashi Dachi**
  - Purpose in karate: A cat-stance with weight on one leg and the other leg partially raised, emphasizing balance and weight distribution.
  - Usage: Reference for asymmetric stance evaluation and leg symmetry expectations.

- **Han Zenkutsu Dachi (Front)**
  - Purpose in karate: Half-forward stance oriented to the front; variant of zenkutsu emphasizing forward weight distribution and a certain leg spread.
  - Usage: Reference for forward-oriented stance geometry (normalized widths and pelvis height ratios).

- **Han Zenkutsu Dachi (Side)**
  - Purpose in karate: Half-forward stance viewed from the side; emphasizes depth and forward-rear foot relation.
  - Usage: Useful when comparing side-view test images; informs pelvis height and depth-related metrics.

- **Zenkutsu Dachi (Front)**
  - Purpose in karate: A full-forward stance used for powerful forward techniques; large stance width and specific leg spread angle.
  - Usage: Reference for deep forward stances with characteristic stance width and leg spread.

- **Zenkutsu Dachi (Side)**
  - Purpose in karate: Side view of the forward stance; emphasizes vertical alignment, pelvis height, and support base from the lateral perspective.
  - Usage: Reference for comparing side-view images and evaluating support base and pelvis vertical placement.

Each reference image is treated as the canonical exemplar for the named stance; the evaluation system computes geometric features from these references for comparison with user images.

---

## 3. Test Dataset

The Test Dataset directory contains user-uploaded or test images intended to exercise the evaluation pipeline. Characteristics:

- **User uploaded stance images:** Photographs of practitioners performing stances; may include different body sizes and clothing.
- **Various viewing angles:** Images may be captured from slightly different viewpoints (frontal and side are the primary intended perspectives), to test the pipeline's robustness to viewpoint variation.
- **Various performers:** Different persons with varying heights, proportions, and execution quality to assess real-world variability.

The test dataset is used both interactively (single image analysis) and in bulk via `batch_evaluation.py` to generate per-image metrics and summary statistics.

---

## 4. Dataset Structure

Top-level structure (project-relative):

```
Reference Stances/
Test Dataset/
```

- `Reference Stances/` — contains reference images named `REF_{stance}.png`, where `{stance}` is the canonical identifier used by the code (e.g., `REF_heiko_dachi.png`).
- `Test Dataset/` — contains test/user images. Filenames often include the stance identifier and an index (e.g., `han_zenkutsu_dachi_front_001.png`).

**Naming convention:**
- Reference images: `REF_{stance}.png` (PNG format assumed).
- Test images: `{stance}_{index}.{ext}` or descriptive names containing the stance identifier; `batch_evaluation.py` attempts to infer stance by substring matching of known stance names.

This naming convention enables automated discovery and pairing of user images with their corresponding reference stance for comparison.

---

## 5. Image Characteristics

- **Image format:** The project expects common raster image formats supported by OpenCV: PNG, JPG/JPEG, BMP, TIFF. Reference images are named with `.png` by convention.
- **Resolution variability:** Images may vary in resolution and aspect ratio; the pipeline normalizes geometric measures by silhouette height where appropriate to reduce scale sensitivity.
- **Lighting variability:** Test images include varied lighting conditions. The pipeline applies CLAHE-based enhancement to mitigate uneven illumination effects, but extreme lighting can still impair segmentation.
- **Background conditions:** Images are typically captured on tatami mats (red or blue) which can introduce high-saturation backgrounds; the pipeline specifically removes red/blue mat colors in HSV before segmentation. Other background clutter may still affect silhouette extraction.

---

## 6. Preprocessing Considerations

The dataset is prepared primarily by algorithmic preprocessing rather than manual curation. Key preprocessing steps applied during evaluation:

- **Tatami color removal:** Many images are shot on colored mats. The pipeline uses HSV-range masking to remove red and blue hues that would otherwise be mistaken for foreground during thresholding.
- **ROI extraction:** The pipeline focuses on a lower-body ROI (default `roi_ratio=0.45`) under the assumption that pelvis and feet lie in the lower portion of the frame; this reduces false positives from upper-body and background.
- **Thresholding:** Otsu's method automatically selects grayscale thresholds for binarization. Gaussian blur is applied beforehand to stabilize threshold selection.
- **Silhouette generation:** Morphological closing/opening and connected component filtering are applied to obtain a single contiguous silhouette representing the subject's lower body.

These preprocessing steps are critical to ensure reliable geometric feature extraction from test images that vary in lighting, resolution, and background.

---

## 7. Dataset Limitations

It is important to acknowledge known dataset limitations that affect pipeline performance and generalizability:

- **Limited number of images:** The dataset is not a large-scale collection; small sample sizes reduce statistical power for threshold calibration or broad generalization.
- **Lighting variations:** Extreme lighting (very low or very high exposure) can still compromise segmentation and lead to incorrect feature detection.
- **Perspective distortions:** The pipeline assumes near-orthogonal frontal or lateral views; strong perspective or camera tilt can distort geometric measures like angles and distances.
- **Clothing variations:** Dark or highly patterned clothing can reduce contrast between body and background; color-based mat removal may also unintentionally affect clothing with similar hues.
- **Single-person assumption:** The pipeline assumes one primary subject per image; multiple people will confound silhouette extraction.

These limitations should be documented in reports and considered when interpreting quantitative results.

---

## 8. Intended Use

The dataset is intended to support the following research and evaluation activities:

- **Geometric feature extraction:** Derive reproducible metrics (stance width, leg spread angle, pelvis metrics, support triangle area, symmetry) from images using non-ML DIP techniques.
- **WKF reference comparison:** Provide per-image comparisons to canonical WKF stances to generate structured feedback on stance correctness.
- **Stability evaluation:** Compute a composite Stability Index from multiple geometric deviations to summarize stance quality and produce categorical feedback (Good/Warning/Critical).

This dataset and pipeline are suitable for academic demonstration, controlled experimental validation, and instructor feedback. They are not designed for large-scale deployment without further robustness improvements and broader dataset collection.

---

*End of Dataset Description.*
