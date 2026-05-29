import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loads input image for processing.
def load_input_image(image_path):
    input_bgr = cv2.imread(image_path)
    if input_bgr is None:
        raise ValueError(f"ERROR: Could not load image:\n{image_path}")
    return input_bgr, input_bgr.copy()


# Builds the WKF reference stance image path.
def get_reference_path(stance_name, ref_dir):
    filename = f"REF_{stance_name}.png"
    return os.path.join(ref_dir, filename)


# Enhances contrast without gamma correction.
def enhance_image(input_bgr, clip_limit=1.2, tile_grid=(8, 8)):
    lab = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# Removes red/blue mat colors from the input image.
def remove_mat_colors(input_bgr):
    hsv = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    full_mask = cv2.bitwise_or(mask_red, mask_blue)
    inv_mask = cv2.bitwise_not(full_mask)

    input_bgr = cv2.bitwise_and(input_bgr, input_bgr, mask=inv_mask)

    input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
    return input_bgr, input_rgb


# Computes full-frame grayscale and lower-body ROI slices.
def compute_roi(input_bgr, roi_ratio=0.45):
    gray_full = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
    height_full = gray_full.shape[0]
    roi_y = int(height_full * roi_ratio)
    gray_roi = gray_full[roi_y:, :]
    roi_bgr = input_bgr[roi_y:, :]
    return gray_full, gray_roi, roi_bgr, roi_y


# Applies Gaussian blur for noise reduction.
def denoise_images(gray_full, gray_roi):
    blurred_full = cv2.GaussianBlur(gray_full, (5, 5), 0)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    return blurred_full, blurred_roi


# Performs Otsu thresholding without inversion.
def otsu_threshold(blurred_full, blurred_roi):
    _, binary_full = cv2.threshold(
        blurred_full,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    _, binary_roi = cv2.threshold(
        blurred_roi,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary_full, binary_roi


# Cleans the binary masks with morphological operations.
def clean_masks(binary_full, binary_roi):
    kernel_small = np.ones((3, 3), np.uint8)

    closed_roi = cv2.morphologyEx(
        binary_roi,
        cv2.MORPH_CLOSE,
        kernel_small,
        iterations=2
    )

    clean_roi = cv2.morphologyEx(
        closed_roi,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    closed_full = cv2.morphologyEx(
        binary_full,
        cv2.MORPH_CLOSE,
        kernel_small,
        iterations=2
    )

    clean_full = cv2.morphologyEx(
        closed_full,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    return clean_full, clean_roi


# Builds the silhouette mask from the cleaned ROI mask.
def build_silhouette(clean_roi):
    silhouette_mask = clean_roi.copy()
    kernel_large = np.ones((7, 7), np.uint8)

    silhouette_mask = cv2.morphologyEx(
        silhouette_mask,
        cv2.MORPH_CLOSE,
        kernel_large,
        iterations=3
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        silhouette_mask,
        connectivity=8
    )
    if num_labels > 1:
        min_component_area = int(0.002 * silhouette_mask.size)
        cleaned_mask = np.zeros_like(silhouette_mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
                cleaned_mask[labels == label] = 255
        silhouette_mask = cleaned_mask

    return silhouette_mask


# Extracts a cleaned contour mask and external contours.
def extract_contours(silhouette_mask):
    if silhouette_mask[0, 0] == 255:
        contour_mask = cv2.bitwise_not(silhouette_mask)
    else:
        contour_mask = silhouette_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        contour_mask,
        connectivity=8
    )
    if num_labels > 1:
        min_component_area = int(0.002 * contour_mask.size)
        cleaned_mask = np.zeros_like(contour_mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
                cleaned_mask[labels == label] = 255
        contour_mask = cleaned_mask

    contours, _ = cv2.findContours(
        contour_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contour_mask, contours


# Finds pelvis center using the first body row in the silhouette.
def find_pelvis_center(silhouette_mask, roi_y):
    mask_points = np.column_stack(
        np.where(silhouette_mask > 0)
    )
    if len(mask_points) == 0:
        raise ValueError("ERROR: Silhouette mask is empty.")

    waist_y = None
    for row in range(silhouette_mask.shape[0]):
        row_pixels = np.where(silhouette_mask[row] > 0)[0]
        if len(row_pixels) > 0:
            waist_y = row
            waist_x = int(np.mean(row_pixels))
            break

    if waist_y is None:
        raise ValueError("ERROR: Could not find waist row in silhouette mask.")

    return (waist_x, waist_y + roi_y), waist_x


# Detects left/right foot points separately using bottom bands.
def detect_feet(silhouette_mask, waist_x, roi_y):
    ys, xs = np.where(silhouette_mask > 0)
    if len(xs) == 0:
        raise ValueError("ERROR: No silhouette pixels found.")

    left_mask = xs < waist_x
    right_mask = xs > waist_x

    left_xs = xs[left_mask]
    left_ys = ys[left_mask]

    right_xs = xs[right_mask]
    right_ys = ys[right_mask]

    if len(left_xs) == 0 or len(right_xs) == 0:
        raise ValueError("ERROR: Could not split silhouette into left/right regions.")

    left_bottom_y = np.max(left_ys)
    left_band = left_ys >= (left_bottom_y - 10)
    left_candidate_xs = left_xs[left_band]
    left_candidate_ys = left_ys[left_band]

    left_index = np.argmin(left_candidate_xs)
    left_foot_point = (
        int(left_candidate_xs[left_index]),
        int(left_candidate_ys[left_index])
    )

    right_bottom_y = np.max(right_ys)
    right_band = right_ys >= (right_bottom_y - 10)
    right_candidate_xs = right_xs[right_band]
    right_candidate_ys = right_ys[right_band]

    right_index = np.argmax(right_candidate_xs)
    right_foot_point = (
        int(right_candidate_xs[right_index]),
        int(right_candidate_ys[right_index])
    )

    left_foot_point_full = (
        left_foot_point[0],
        left_foot_point[1] + roi_y
    )

    right_foot_point_full = (
        right_foot_point[0],
        right_foot_point[1] + roi_y
    )

    return left_foot_point_full, right_foot_point_full


# Draws contour overlays on the original image.
def draw_contours(input_rgb_raw, contours, roi_y):
    contour_output = input_rgb_raw.copy()
    contours_full = []
    for contour in contours:
        contour_full = contour.copy()
        contour_full[:, 0, 1] += roi_y
        contours_full.append(contour_full)

    cv2.drawContours(
        contour_output,
        contours_full,
        -1,
        (0, 255, 0),
        4
    )

    return contour_output


# Draws stance geometry and annotations on the output image.
def draw_stance_output(
    input_rgb_raw,
    center_node,
    left_foot_point_full,
    right_foot_point_full,
    stance_width
):
    stance_output = input_rgb_raw.copy()
    node_color = (255, 0, 0)
    line_color = (0, 0, 255)

    cv2.circle(stance_output, center_node, 8, node_color, -1)
    cv2.circle(stance_output, left_foot_point_full, 8, node_color, -1)
    cv2.circle(stance_output, right_foot_point_full, 8, node_color, -1)

    cv2.putText(
        stance_output,
        "Left Foot",
        (left_foot_point_full[0] - 20, left_foot_point_full[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        node_color,
        1
    )

    cv2.putText(
        stance_output,
        "Right Foot",
        (right_foot_point_full[0] - 20, right_foot_point_full[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        node_color,
        1
    )

    cv2.line(stance_output, center_node, left_foot_point_full, line_color, 3)
    cv2.line(stance_output, center_node, right_foot_point_full, line_color, 3)
    cv2.line(stance_output, left_foot_point_full, right_foot_point_full, line_color, 3)

    cv2.putText(
        stance_output,
        "Center of Mass",
        (center_node[0] + 10, center_node[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        node_color,
        1
    )

    cv2.putText(
        stance_output,
        f"Stance Width: {stance_width:.2f}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    return stance_output


# Shows diagnostic and final stance visualizations.
def show_plots(
    input_rgb_raw,
    input_rgb,
    enhanced_rgb,
    gray_roi,
    binary_mask,
    clean_mask,
    contour_mask,
    contour_output,
    stance_output,
    has_contours
):
    plt.figure(figsize=(16, 9))

    plt.subplot(2, 4, 1)
    plt.imshow(input_rgb_raw)
    plt.title("Original Input")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(input_rgb)
    plt.title("After Red/Blue Removal")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(enhanced_rgb)
    plt.title("After Removal + Enhanced")
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(gray_roi, cmap='gray')
    plt.title("Lower-Body ROI (Gray)")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Otsu Threshold")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(clean_mask, cmap='gray')
    plt.title("Morphological Cleaning")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(contour_mask, cmap='gray')
    plt.title("Final Silhouette Mask")
    plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.imshow(contour_output)
    plt.title("Geometric Contour Extraction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



# Renders the geometric metrics table.
def draw_metrics_table(ax, metrics_data):
    ax.axis("off")

    headers = ["Feature", "WKF Reference", "User Value", "Difference", "Status", "Feedback"]
    rows = [
        [
            item.get("feature", ""),
            item.get("reference", ""),
            item.get("user", ""),
            item.get("difference", ""),
            item.get("status", ""),
            item.get("feedback", "")
        ]
        for item in metrics_data
    ]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    status_colors = {
        "good": "#b6e3b6",
        "warning": "#ffd89e",
        "critical": "#f5b5b5"
    }

    for idx, item in enumerate(metrics_data, start=1):
        status = str(item.get("status", "")).strip().lower()
        color = status_colors.get(status)
        if color is not None:
            table[(idx, 4)].set_facecolor(color)

        if str(item.get("feature", "")).strip().lower() == "stability index":
            for col in range(len(headers)):
                if col == 4:
                    continue
                table[(idx, col)].set_facecolor("#55B5CF")


# Calculates the leg spread angle using pelvis and foot points.
def calculate_leg_spread_angle(pelvis_point, left_foot_point, right_foot_point):
    pelvis = np.array(pelvis_point, dtype=float)
    left_foot = np.array(left_foot_point, dtype=float)
    right_foot = np.array(right_foot_point, dtype=float)

    vec_left = left_foot - pelvis
    vec_right = right_foot - pelvis

    left_norm = np.linalg.norm(vec_left)
    right_norm = np.linalg.norm(vec_right)
    if left_norm == 0 or right_norm == 0:
        return 0.0

    cos_angle = np.dot(vec_left, vec_right) / (left_norm * right_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))


# Calculates silhouette height from the mask bounding box.
def calculate_silhouette_height(silhouette_mask):
    ys, xs = np.where(silhouette_mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return 0
    top_y = int(np.min(ys))
    bottom_y = int(np.max(ys))
    return max(bottom_y - top_y, 0)


# Calculates pelvis height ratio within the silhouette bounds.
def compute_pelvis_height_ratio(pelvis_point, silhouette_mask):
    ys, xs = np.where(silhouette_mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return 0.0
    top_y = float(np.min(ys))
    bottom_y = float(np.max(ys))
    height = bottom_y - top_y
    if height <= 0:
        return 0.0
    return float(pelvis_point[1] - top_y) / height


# Compares stance width values and returns a metrics table row.
def compare_stance_width(ref_width, user_width, ref_height, user_height):
    ref_norm = (ref_width / ref_height) if ref_height > 0 else 0.0
    user_norm = (user_width / user_height) if user_height > 0 else 0.0
    diff = user_norm - ref_norm
    if ref_norm == 0:
        abs_diff = abs(diff)
        percent_diff = abs_diff
    else:
        percent_diff = (abs(diff) / ref_norm) * 100.0

    if percent_diff <= 10:
        status = "Good"
    elif percent_diff <= 25:
        status = "Warning"
    else:
        status = "Critical"

    if status == "Good":
        feedback = "Proper stance width"
    elif status == "Warning":
        if user_width < ref_width:
            feedback = "Slightly narrow stance"
        elif user_width > ref_width:
            feedback = "Slightly wide stance"
        else:
            feedback = "Proper stance width"
    else:
        feedback = "Adjust stance width significantly"

    return {
        "feature": "Stance Width",
        "reference": f"{ref_norm:.2f}",
        "user": f"{user_norm:.2f}",
        "difference": f"{diff:.2f}",
        "status": status,
        "feedback": feedback
    }


# Compares leg spread angles and returns a metrics table row.
def compare_leg_spread_angle(ref_angle, user_angle):
    diff = user_angle - ref_angle
    abs_diff = abs(diff)

    if abs_diff <= 10:
        status = "Good"
    elif abs_diff <= 25:
        status = "Warning"
    else:
        status = "Critical"

    if status == "Good":
        feedback = "Proper leg spread"
    elif status == "Warning":
        if user_angle < ref_angle:
            feedback = "Increase leg spread"
        elif user_angle > ref_angle:
            feedback = "Reduce leg spread"
        else:
            feedback = "Proper leg spread"
    else:
        feedback = "Correct leg spread immediately"

    return {
        "feature": "Leg Spread Angle",
        "reference": f"{ref_angle:.1f}°",
        "user": f"{user_angle:.1f}°",
        "difference": f"{diff:.1f}°",
        "status": status,
        "feedback": feedback
    }


# Computes lateral pelvis offset from the foot midpoint.
def compute_pelvis_lateral_offset(pelvis_point, left_foot_point, right_foot_point):
    midpoint_x = (left_foot_point[0] + right_foot_point[0]) / 2.0
    return abs(pelvis_point[0] - midpoint_x)


# Compares pelvis lateral offsets and returns a metrics table row.
def evaluate_pelvis_offset(reference_offset, user_offset):
    diff = user_offset - reference_offset
    abs_diff = abs(diff)

    if abs_diff <= 15:
        status = "Good"
    elif abs_diff <= 35:
        status = "Warning"
    else:
        status = "Critical"

    if status == "Good":
        feedback = "Balanced pelvis alignment"
    elif status == "Warning":
        feedback = "Slight pelvis imbalance"
    else:
        feedback = "Shift pelvis toward center"

    return {
        "feature": "Pelvis Lateral Offset",
        "reference": f"{reference_offset:.2f}",
        "user": f"{user_offset:.2f}",
        "difference": f"{diff:.2f}",
        "status": status,
        "feedback": feedback
    }


# Computes normalized support triangle area from pelvis and feet points.
def compute_support_triangle_area(
    pelvis_point,
    left_foot_point,
    right_foot_point,
    silhouette_height
):
    if silhouette_height <= 0:
        return 0.0
    x1, y1 = pelvis_point
    x2, y2 = left_foot_point
    x3, y3 = right_foot_point
    area = 0.5 * abs(
        x1 * (y2 - y3) +
        x2 * (y3 - y1) +
        x3 * (y1 - y2)
    )
    return float(area) / (float(silhouette_height) ** 2)


# Compares support triangle areas and returns a metrics table row.
def evaluate_support_triangle_area(reference_area, user_area):
    diff = user_area - reference_area
    abs_diff = abs(diff)

    if abs_diff <= 0.08:
        status = "Good"
    elif abs_diff <= 0.18:
        status = "Warning"
    else:
        status = "Critical"

    if status == "Good":
        feedback = "Stable support base"
    elif status == "Warning":
        feedback = "Moderate support imbalance"
    else:
        feedback = "Adjust stance stability"

    return {
        "feature": "Support Triangle Area",
        "reference": f"{reference_area:.2f}",
        "user": f"{user_area:.2f}",
        "difference": f"{diff:.2f}",
        "status": status,
        "feedback": feedback
    }


# Compares pelvis height ratios and returns a metrics table row.
def evaluate_pelvis_height_ratio(reference_ratio, user_ratio):
    diff = user_ratio - reference_ratio
    abs_diff = abs(diff)

    if abs_diff <= 0.08:
        status = "Good"
    elif abs_diff <= 0.18:
        status = "Warning"
    else:
        status = "Critical"

    if status == "Good":
        feedback = "Proper pelvis height"
    elif status == "Warning":
        feedback = "Slight pelvis height deviation"
    else:
        feedback = "Adjust stance depth"

    return {
        "feature": "Pelvis Height Ratio",
        "reference": f"{reference_ratio:.2f}",
        "user": f"{user_ratio:.2f}",
        "difference": f"{diff:.2f}",
        "status": status,
        "feedback": feedback
    }


# Computes left-right leg symmetry ratio.
def compute_leg_symmetry_ratio(pelvis_point, left_foot_point, right_foot_point):
    left_dist = np.linalg.norm(
        np.array(left_foot_point, dtype=float) -
        np.array(pelvis_point, dtype=float)
    )
    right_dist = np.linalg.norm(
        np.array(right_foot_point, dtype=float) -
        np.array(pelvis_point, dtype=float)
    )
    if right_dist == 0:
        return 0.0
    return float(left_dist / right_dist)


# Compares leg symmetry ratios and returns a metrics table row.
def evaluate_leg_symmetry_ratio(reference_ratio, user_ratio):
    if reference_ratio == 0:
        symmetry_deviation = 0.0
    else:
        symmetry_deviation = abs(1 - (user_ratio / reference_ratio))

    if symmetry_deviation <= 0.20:
        status = "Good"
    elif symmetry_deviation <= 0.40:
        status = "Warning"
    else:
        status = "Critical"

    if status == "Good":
        feedback = "Balanced leg distribution"
    elif status == "Warning":
        feedback = "Slight leg imbalance"
    else:
        feedback = "Correct weight distribution"

    return {
        "feature": "Leg Symmetry Ratio",
        "reference": f"{reference_ratio:.2f}",
        "user": f"{user_ratio:.2f}",
        "difference": f"{symmetry_deviation:.2f}",
        "status": status,
        "feedback": feedback
    }


# Computes stability index from metric differences.
def compute_stability_index(metrics_dictionary):
    # Thresholds align with the warning bounds for each metric.
    thresholds = {
        "stance_width": 0.25,
        "leg_spread": 25.0,
        "pelvis_offset": 35.0,
        "triangle_area": 0.18,
        "pelvis_height": 0.18,
        "leg_symmetry": 0.25
    }

    weights = {
        "stance_width": 0.20,
        "leg_spread": 0.20,
        "pelvis_offset": 0.15,
        "triangle_area": 0.15,
        "pelvis_height": 0.15,
        "leg_symmetry": 0.15
    }

    total_weight = 0.0
    weighted_penalty = 0.0

    for key, weight in weights.items():
        diff = metrics_dictionary.get(key, 0.0)
        threshold = thresholds.get(key, 1.0)
        penalty = min(abs(diff) / threshold, 1.0) if threshold > 0 else 0.0
        weighted_penalty += penalty * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    stability_index = 100.0 * (1.0 - (weighted_penalty / total_weight))
    return max(min(stability_index, 100.0), 0.0)


# Evaluates the stability index and returns a metrics table row.
def evaluate_stability_index(stability_index):
    if stability_index >= 80:
        status = "Good"
        feedback = "Stable stance posture"
    elif stability_index >= 60:
        status = "Warning"
        feedback = "Moderate posture instability"
    else:
        status = "Critical"
        feedback = "Poor stance stability"

    return {
        "feature": "Stability Index",
        "reference": "100.00%",
        "user": f"{stability_index:.2f}%",
        "difference": f"{100.0 - stability_index:.2f}%",
        "status": status,
        "feedback": feedback
    }


# Shows the final comparison template.
def show_template_figure(ref_rgb, user_rgb, stance_label, metrics_data):
    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 2])
    ax_ref = fig.add_subplot(grid[0, 0])
    ax_user = fig.add_subplot(grid[0, 1])
    ax_table = fig.add_subplot(grid[1, :])

    common_h = min(ref_rgb.shape[0], user_rgb.shape[0])
    ref_w = int(ref_rgb.shape[1] * (common_h / ref_rgb.shape[0]))
    user_w = int(user_rgb.shape[1] * (common_h / user_rgb.shape[0]))
    ref_view = cv2.resize(ref_rgb, (ref_w, common_h), interpolation=cv2.INTER_AREA)
    user_view = cv2.resize(user_rgb, (user_w, common_h), interpolation=cv2.INTER_AREA)

    ax_ref.imshow(ref_view)
    ax_ref.set_title("WKF Reference Stance")
    ax_ref.axis("off")

    ax_user.imshow(user_view)
    ax_user.set_title("User Uploaded Stance")
    ax_user.axis("off")

    fig.text(
        0.5,
        0.585,
        "Geometric Calculation",
        ha="center",
        va="center",
        fontsize=12
    )
    fig.text(
        0.5,
        0.56,
        f"Stance: {stance_label}",
        ha="center",
        va="center",
        fontsize=11
    )
    draw_metrics_table(ax_table, metrics_data)

    fig.suptitle("Karate Posture Evaluation System", fontsize=16)
    plt.tight_layout()
    plt.show()


def save_evaluation_report(stance_name, ref_rgb, user_rgb, metrics_data, out_dir=None):
   
    try:
        if out_dir is None:
            out_dir = os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        fig = plt.figure(figsize=(18, 10))
        grid = fig.add_gridspec(2, 2, height_ratios=[3, 2])
        ax_ref = fig.add_subplot(grid[0, 0])
        ax_user = fig.add_subplot(grid[0, 1])
        ax_table = fig.add_subplot(grid[1, :])

        # Resize images to a common height to keep layout consistent
        common_h = min(ref_rgb.shape[0], user_rgb.shape[0])
        ref_w = int(ref_rgb.shape[1] * (common_h / ref_rgb.shape[0]))
        user_w = int(user_rgb.shape[1] * (common_h / user_rgb.shape[0]))
        ref_view = cv2.resize(ref_rgb, (ref_w, common_h), interpolation=cv2.INTER_AREA)
        user_view = cv2.resize(user_rgb, (user_w, common_h), interpolation=cv2.INTER_AREA)

        ax_ref.imshow(ref_view)
        ax_ref.set_title("WKF Reference Stance")
        ax_ref.axis("off")

        ax_user.imshow(user_view)
        ax_user.set_title("User Uploaded Stance")
        ax_user.axis("off")

        fig.text(0.5, 0.585, "Geometric Calculation", ha="center", va="center", fontsize=12)
        fig.text(0.5, 0.56, f"Stance: {stance_name}", ha="center", va="center", fontsize=11)

        draw_metrics_table(ax_table, metrics_data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = stance_name.replace(" ", "_")
        filename = f"{safe_name}_{timestamp}.png"
        out_path = os.path.join(out_dir, filename)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_path
    except Exception:
        try:
            plt.close('all')
        except Exception:
            pass
        return None


# Runs the full stance analysis pipeline.
def run_analysis(image_path, stance_name):
    input_bgr, input_bgr_raw = load_input_image(image_path)
    input_bgr, input_rgb = remove_mat_colors(input_bgr)
    input_bgr = enhance_image(input_bgr)
    enhanced_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
    input_rgb_raw = cv2.cvtColor(input_bgr_raw, cv2.COLOR_BGR2RGB)

    gray_full, gray_roi, _, roi_y = compute_roi(input_bgr)
    blurred_full, blurred_roi = denoise_images(gray_full, gray_roi)
    binary_full, binary_roi = otsu_threshold(blurred_full, blurred_roi)
    _, clean_roi = clean_masks(binary_full, binary_roi)

    silhouette_mask = build_silhouette(clean_roi)
    contour_mask, contours = extract_contours(silhouette_mask)
    silhouette_height = calculate_silhouette_height(silhouette_mask)

    contour_output = input_rgb_raw.copy()
    stance_output = input_rgb_raw.copy()
    stance_width = 0.0
    center_node = (0, 0)
    left_foot_point_full = (0, 0)
    right_foot_point_full = (0, 0)

    if len(contours) > 0:
        contour_output = draw_contours(input_rgb_raw, contours, roi_y)
        center_node, waist_x = find_pelvis_center(silhouette_mask, roi_y)
        left_foot_point_full, right_foot_point_full = detect_feet(
            silhouette_mask,
            waist_x,
            roi_y
        )
        stance_width = np.sqrt(
            (right_foot_point_full[0] - left_foot_point_full[0]) ** 2 +
            (right_foot_point_full[1] - left_foot_point_full[1]) ** 2
        )
        stance_output = draw_stance_output(
            input_rgb_raw,
            center_node,
            left_foot_point_full,
            right_foot_point_full,
            stance_width
        )

    return {
        "stance_name": stance_name,
        "stance_width": stance_width,
        "center_node": center_node,
        "left_foot": left_foot_point_full,
        "right_foot": right_foot_point_full,
        "stance_output": stance_output,
        "contour_output": contour_output,
        "contour_mask": contour_mask,
        "silhouette_mask": silhouette_mask,
        "silhouette_height": silhouette_height,
        "input_rgb": input_rgb,
        "enhanced_rgb": enhanced_rgb,
        "input_rgb_raw": input_rgb_raw,
        "gray_roi": gray_roi,
        "binary_mask": binary_roi,
        "clean_mask": clean_roi,
        "has_contours": len(contours) > 0
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Karate stance analysis")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument(
        "--stance",
        choices=[
            "neko_ashi_dachi",
            "zenkutsu_dachi_front",
            "siko_dachi",
            "han_zenkutsu_dachi_side",
            "heiko_dachi",
            "han_zenkutsu_dachi_front",
            "zenkutsu_dachi_side"
        ],
        help="Stance name"
    )
    parser.add_argument(
        "--ref-dir",
        help="Path to reference stances directory (optional). If omitted, uses the Reference Stances folder next to this script."
    )
    return parser.parse_args()


def prompt_if_missing(args):
    image_path = args.image or input("Image path: ").strip()
    image_path = image_path.strip('"')
    if args.stance:
        stance_name = args.stance
    else:
        print("Select stance:")
        print("1) Neko Ashi Dachi")
        print("2) Zenkutsu Dachi (Front)")
        print("3) Zenkutsu Dachi (Side)")
        print("4) Han Zenkutsu Dachi (Front)")
        print("5) Han Zenkutsu Dachi (Side)")
        print("6) Siko Dachi")
        print("7) Heiko Dachi")
        choice = input("Enter 1-7: ").strip()
        stance_map = {
            "1": "neko_ashi_dachi",
            "2": "zenkutsu_dachi_front",
            "3": "zenkutsu_dachi_side",
            "4": "han_zenkutsu_dachi_front",
            "5": "han_zenkutsu_dachi_side",
            "6": "siko_dachi",
            "7": "heiko_dachi"
        }
        stance_name = stance_map.get(choice, "")

    if stance_name not in {
        "neko_ashi_dachi",
        "zenkutsu_dachi_front",
        "siko_dachi",
        "han_zenkutsu_dachi_side",
        "heiko_dachi",
        "han_zenkutsu_dachi_front",
        "zenkutsu_dachi_side"
    }:
        raise ValueError("ERROR: Invalid stance name.")
    return image_path, stance_name


def main():
    args = parse_args()
    image_path, stance_name = prompt_if_missing(args)

    result = run_analysis(image_path, stance_name)

    show_plots(
        result["input_rgb_raw"],
        result["input_rgb"],
        result["enhanced_rgb"],
        result["gray_roi"],
        result["binary_mask"],
        result["clean_mask"],
        result["contour_mask"],
        result["contour_output"],
        result["stance_output"],
        result["has_contours"]
    )

    # Determine reference directory: prefer CLI arg, else use project-relative folder
    if getattr(args, "ref_dir", None):
        ref_dir = args.ref_dir
    else:
        ref_dir = os.path.join(os.path.dirname(__file__), "Reference Stances")

    ref_path = get_reference_path(stance_name, ref_dir)
    ref_result = None
    if os.path.exists(ref_path):
        try:
            ref_result = run_analysis(ref_path, stance_name)
        except Exception as e:
            print(f"Warning: Failed to analyze reference image: {e}")
            ref_result = None
    else:
        print(f"Warning: Reference image not found at {ref_path}")

    # Fallback: if reference analysis not available, use the user's input image for display
    if ref_result is not None:
        ref_rgb = ref_result["stance_output"] if ref_result.get("has_contours") else ref_result["input_rgb_raw"]
    else:
        ref_rgb = result["input_rgb_raw"]
    user_rgb = result["stance_output"] if result["has_contours"] else result["input_rgb_raw"]
    stance_label = stance_name.replace("_", " ")
    metrics_data = []
    if ref_result["has_contours"] and result["has_contours"]:
        metrics_data.append(
            compare_stance_width(
                ref_result["stance_width"],
                result["stance_width"],
                ref_result["silhouette_height"],
                result["silhouette_height"]
            )
        )

        ref_norm_width = (
            ref_result["stance_width"] / ref_result["silhouette_height"]
            if ref_result["silhouette_height"] > 0 else 0.0
        )
        user_norm_width = (
            result["stance_width"] / result["silhouette_height"]
            if result["silhouette_height"] > 0 else 0.0
        )

        ref_angle = calculate_leg_spread_angle(
            ref_result["center_node"],
            ref_result["left_foot"],
            ref_result["right_foot"]
        )
        user_angle = calculate_leg_spread_angle(
            result["center_node"],
            result["left_foot"],
            result["right_foot"]
        )
        metrics_data.append(compare_leg_spread_angle(ref_angle, user_angle))

        ref_offset = compute_pelvis_lateral_offset(
            ref_result["center_node"],
            ref_result["left_foot"],
            ref_result["right_foot"]
        )
        user_offset = compute_pelvis_lateral_offset(
            result["center_node"],
            result["left_foot"],
            result["right_foot"]
        )
        metrics_data.append(evaluate_pelvis_offset(ref_offset, user_offset))

        ref_area = compute_support_triangle_area(
            ref_result["center_node"],
            ref_result["left_foot"],
            ref_result["right_foot"],
            ref_result["silhouette_height"]
        )
        user_area = compute_support_triangle_area(
            result["center_node"],
            result["left_foot"],
            result["right_foot"],
            result["silhouette_height"]
        )
        metrics_data.append(evaluate_support_triangle_area(ref_area, user_area))

        ref_pelvis_ratio = compute_pelvis_height_ratio(
            ref_result["center_node"],
            ref_result["silhouette_mask"]
        )
        user_pelvis_ratio = compute_pelvis_height_ratio(
            result["center_node"],
            result["silhouette_mask"]
        )
        metrics_data.append(
            evaluate_pelvis_height_ratio(ref_pelvis_ratio, user_pelvis_ratio)
        )

        ref_leg_sym = compute_leg_symmetry_ratio(
            ref_result["center_node"],
            ref_result["left_foot"],
            ref_result["right_foot"]
        )
        user_leg_sym = compute_leg_symmetry_ratio(
            result["center_node"],
            result["left_foot"],
            result["right_foot"]
        )
        metrics_data.append(
            evaluate_leg_symmetry_ratio(ref_leg_sym, user_leg_sym)
        )

        stability_inputs = {
            "stance_width": user_norm_width - ref_norm_width,
            "leg_spread": user_angle - ref_angle,
            "pelvis_offset": user_offset - ref_offset,
            "triangle_area": user_area - ref_area,
            "pelvis_height": user_pelvis_ratio - ref_pelvis_ratio,
            "leg_symmetry": user_leg_sym - ref_leg_sym
        }
        stability_index = compute_stability_index(stability_inputs)
        metrics_data.append(evaluate_stability_index(stability_index))

    report_path = save_evaluation_report(stance_name, ref_rgb, user_rgb, metrics_data)
    if report_path:
        print(f"Saved evaluation report: {report_path}")

    show_template_figure(ref_rgb, user_rgb, stance_label, metrics_data)


if __name__ == "__main__":
    main()
