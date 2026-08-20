# === SAHI LABEL REMOVED FROM UI (accurate: full-frame VisDrone, no tiling) - Aug 19 ===
"""
Drone-Based Parking Occupancy Monitoring — Streamlit version
--------------------------------------------------------------
Uses the ACTUAL project scripts verbatim, adapted only to run on an in-memory
uploaded image instead of a file-range batch job:

  - Detection + deduplication: ported directly from aerial_batch_run_detector_.py
    (VisDrone-trained YOLOv11x via SAHI's AutoDetectionModel + get_prediction --
    full-frame, non-sliced inference; exact dedupe_within_groups logic)
  - Occupancy calculation: ported directly from compute_availability.py
    (exact clip_polygon / polygon_area / overlap_fraction / load_space_polygons)
  - Bay-boundary transfer: ported directly from transfer_polygons.py
    (SIFT + RANSAC homography, same validity check)

Folder layout expected:
    model/best.pt
    annotations/<SiteName>.json   (COCO export with segmentation polygons)
    sample_images/*.JPG
"""

import json
import math
import os
import glob
import tempfile
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_prediction

import gps_utils
import site_geo

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "model/best.pt"
ANNOTATIONS_DIR = "annotations"
SAMPLE_IMAGES_DIR = "sample_images"
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.50
IOS_THRESHOLD = 0.75
CENTRE_FACTOR = 0.65
IMGSZ = 640
OCC_FRAC_DEFAULT = 0.12

# Motor deliberately excluded (false positives on people) -- same as thesis pipeline
VEHICLE_CLASSES = {"car", "van", "truck", "bus"}
PERSON_CLASSES = {"pedestrian", "people"}
ALLOWED_CLASSES = VEHICLE_CLASSES | PERSON_CLASSES
CLASS_PRIORITY = {"car": 6, "van": 5, "truck": 5, "bus": 5, "pedestrian": 4, "people": 3}

REFERENCE_IMAGES = {
    "CarPark2": os.path.join(SAMPLE_IMAGES_DIR, "DJI_0028.JPG"),
    "Foundation": os.path.join(SAMPLE_IMAGES_DIR, "DJI_0079.JPG"),
    "Foundation118m": os.path.join(SAMPLE_IMAGES_DIR, "DJI_0084.JPG"),
    "KBS": os.path.join(SAMPLE_IMAGES_DIR, "DJI_0098.JPG"),
}


# ---------------------------------------------------------------------------
# Detection: ported verbatim from aerial_batch_run_detector_.py
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    source_image: str
    image_number: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    centre_x: float
    centre_y: float
    kept: bool = False
    status: str = ""
    duplicate_of: str = ""

    @property
    def category_group(self) -> str:
        if self.class_name in VEHICLE_CLASSES:
            return "vehicle"
        if self.class_name in PERSON_CLASSES:
            return "person"
        return "other"

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def short_side(self) -> float:
        return min(self.width, self.height)

    @property
    def long_side(self) -> float:
        return max(self.width, self.height)

    @property
    def identity(self) -> str:
        return (
            f"{self.category_group}:{self.class_name}:{self.confidence:.3f}:"
            f"{self.x1:.1f},{self.y1:.1f},{self.x2:.1f},{self.y2:.1f}"
        )


def to_detection(prediction: Any, source_image: str, image_number: int) -> Detection:
    x1, y1, x2, y2 = [float(v) for v in prediction.bbox.to_xyxy()]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return Detection(
        source_image=source_image, image_number=image_number,
        class_name=str(prediction.category.name).strip().lower(),
        confidence=float(prediction.score.value),
        x1=x1, y1=y1, x2=x2, y2=y2, width=width, height=height,
        centre_x=(x1 + x2) / 2.0, centre_y=(y1 + y2) / 2.0,
    )


def extract_detections(result: Any, image_path: str, image_number: int, confidence: float) -> list:
    detections = []
    for prediction in result.object_prediction_list:
        detection = to_detection(prediction, source_image=str(image_path), image_number=image_number)
        if detection.class_name not in ALLOWED_CLASSES:
            continue
        if detection.confidence < confidence:
            continue
        if detection.width <= 1.0 or detection.height <= 1.0:
            continue
        detections.append(detection)
    return detections


def intersection_metrics(a: Detection, b: Detection):
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0, 0.0
    union = a.area + b.area - intersection
    smaller_area = min(a.area, b.area)
    iou = intersection / union if union > 0.0 else 0.0
    ios = intersection / smaller_area if smaller_area > 0.0 else 0.0
    return iou, ios


def centre_distance(a: Detection, b: Detection) -> float:
    return math.hypot(a.centre_x - b.centre_x, a.centre_y - b.centre_y)


def detection_priority(detection: Detection):
    return (detection.confidence, CLASS_PRIORITY.get(detection.class_name, 1))


def dedupe_within_groups(detections, iou_threshold, ios_threshold, centre_factor):
    """Deduplicate vehicles against vehicles and people against people.
    A person box overlapping a car is not removed merely because the boxes overlap."""
    kept_all, removed_all = [], []
    for group_name in ("vehicle", "person"):
        group = [d for d in detections if d.category_group == group_name]
        ordered = sorted(group, key=detection_priority, reverse=True)
        kept = []
        for candidate in ordered:
            duplicate = None
            for accepted in kept:
                iou, ios = intersection_metrics(candidate, accepted)
                distance = centre_distance(candidate, accepted)
                centre_limit = centre_factor * min(candidate.long_side, accepted.long_side)
                if iou >= iou_threshold or (ios >= ios_threshold and distance <= centre_limit):
                    duplicate = accepted
                    break
            if duplicate is None:
                candidate.kept = True
                candidate.status = f"{group_name} retained"
                kept.append(candidate)
                kept_all.append(candidate)
            else:
                candidate.kept = False
                candidate.status = f"{group_name} duplicate removed"
                candidate.duplicate_of = duplicate.identity
                removed_all.append(candidate)
    kept_all.sort(key=lambda d: (d.category_group, -d.confidence))
    return kept_all, removed_all


@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}.")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=device,
        image_size=IMGSZ,
    )


def run_detection_on_image(image_rgb: np.ndarray):
    """Save the in-memory image to a temp file (get_prediction expects a path,
    matching the exact usage in aerial_batch_run_detector_.py) and run detection."""
    detection_model = get_model()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        tmp_path = tmp.name
    try:
        result = get_prediction(image=tmp_path, detection_model=detection_model, verbose=0)
        raw = extract_detections(result, image_path=tmp_path, image_number=0, confidence=CONFIDENCE_THRESHOLD)
        kept, removed = dedupe_within_groups(raw, IOU_THRESHOLD, IOS_THRESHOLD, CENTRE_FACTOR)
    finally:
        os.unlink(tmp_path)
    return kept, removed


def draw_detections(image_rgb: np.ndarray, detections: list):
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for d in detections:
        color = (80, 200, 120) if d.category_group == "vehicle" else (200, 160, 60)
        cv2.rectangle(img_bgr, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 3)
        cv2.putText(img_bgr, f"{d.class_name} {d.confidence:.2f}", (int(d.x1), max(15, int(d.y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Occupancy: ported verbatim from compute_availability.py
# ---------------------------------------------------------------------------
def polygon_area(poly):
    """Shoelace formula. poly = [(x,y), (x,y), ...]"""
    n = len(poly)
    a = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def clip_polygon(subject, clip):
    """Sutherland-Hodgman polygon clipping"""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersect(p1, p2, a, b):
        x1, y1 = p1; x2, y2 = p2
        x3, y3 = a; x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output = list(subject)
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        if not output:
            break
        inp = output
        output = []
        for j in range(len(inp)):
            cur = inp[j]
            prv = inp[j - 1]
            if inside(cur, a, b):
                if not inside(prv, a, b):
                    output.append(intersect(prv, cur, a, b))
                output.append(cur)
            elif inside(prv, a, b):
                output.append(intersect(prv, cur, a, b))
    return output


def overlap_fraction(bay_poly, car_box):
    x1, y1, x2, y2 = car_box
    car_poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    inter = clip_polygon(bay_poly, car_poly)
    if len(inter) < 3:
        return 0.0
    bay_area = polygon_area(bay_poly)
    if bay_area <= 0:
        return 0.0
    return polygon_area(inter) / bay_area


def load_space_polygons_from_json(coco_dict):
    img = coco_dict["images"][0]
    coco_w, coco_h = img["width"], img["height"]
    polys = []
    n_from_seg = n_from_bbox = 0
    for a in coco_dict["annotations"]:
        seg = a.get("segmentation")
        if seg and len(seg) > 0 and len(seg[0]) >= 6:
            flat = seg[0]
            pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
            polys.append(pts)
            n_from_seg += 1
        else:
            x, y, w, h = a["bbox"]
            polys.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            n_from_bbox += 1
    return polys, (coco_w, coco_h), (n_from_seg, n_from_bbox)


def rescale_poly(poly, sx, sy):
    return [(px * sx, py * sy) for (px, py) in poly]


def list_annotation_sites():
    if not os.path.isdir(ANNOTATIONS_DIR):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(ANNOTATIONS_DIR) if f.lower().endswith(".json"))


def load_site_coco(site_name):
    path = os.path.join(ANNOTATIONS_DIR, f"{site_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_occupancy_map(image_rgb, bay_polys, vehicle_boxes, occ_frac):
    """Exact logic from compute_availability.py's main(), returning an
    annotated image (green=free, red=occupied) plus per-space results."""
    per_space = []
    occupied = 0
    for bp in bay_polys:
        best = 0.0
        for cb in vehicle_boxes:
            f = overlap_fraction(bp, cb)
            if f > best:
                best = f
        is_occ = best >= occ_frac
        if is_occ:
            occupied += 1
        per_space.append({"status": "occupied" if is_occ else "available", "overlap_fraction": round(best, 3)})

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for bp, ps in zip(bay_polys, per_space):
        color = (0, 0, 255) if ps["status"] == "occupied" else (0, 255, 0)  # BGR: red / green
        pts = np.array([[int(x), int(y)] for x, y in bp], dtype=np.int32)
        cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=4)
    occ_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    capacity = len(bay_polys)
    available = capacity - occupied
    return occ_img_rgb, occupied, available, capacity


# ---------------------------------------------------------------------------
# Bay-boundary transfer: ported verbatim from transfer_polygons.py
# ---------------------------------------------------------------------------
DETECT_MAX_DIM = 2000
MIN_MATCH_COUNT = 15
RANSAC_REPROJ_THRESHOLD = 5.0
MIN_AREA_FRACTION = 0.15


def _to_gray_downscaled(img_bgr, max_dim):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else img_bgr
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), scale


def compute_homography(ref_gray, target_gray, scale_ref, scale_target):
    sift = cv2.SIFT_create(nfeatures=8000)
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)
    if des1 is None or des2 is None:
        raise RuntimeError("No SIFT features found in one of the images.")
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < MIN_MATCH_COUNT:
        raise RuntimeError(f"Only {len(good)} good matches found (need >= {MIN_MATCH_COUNT}).")
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / scale_ref
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) / scale_target
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None:
        raise RuntimeError("Homography estimation failed.")
    return H, inliers, len(good)


def validate_homography(H, ref_shape, target_shape, min_area_fraction=MIN_AREA_FRACTION):
    h_ref, w_ref = ref_shape
    h_tgt, w_tgt = target_shape
    corners = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    x, y = warped[:, 0], warped[:, 1]
    area = 0.5 * abs(sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4)))
    target_area = w_tgt * h_tgt
    area_fraction = area / target_area if target_area else 0.0
    return area_fraction >= min_area_fraction, area_fraction


def warp_polygon(flat_poly, H):
    pts = np.array(flat_poly, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [(float(p[0]), float(p[1])) for p in warped]


def transfer_annotation(reference_site, target_image_rgb):
    ref_path = REFERENCE_IMAGES.get(reference_site)
    if not ref_path or not os.path.exists(ref_path):
        return None, f"No reference image found for site '{reference_site}'."

    ref_img = cv2.imread(ref_path)
    target_img = cv2.cvtColor(target_image_rgb, cv2.COLOR_RGB2BGR)
    ref_gray, scale_ref = _to_gray_downscaled(ref_img, DETECT_MAX_DIM)
    target_gray, scale_target = _to_gray_downscaled(target_img, DETECT_MAX_DIM)

    try:
        H, inliers, good_matches = compute_homography(ref_gray, target_gray, scale_ref, scale_target)
    except RuntimeError as e:
        return None, f"Homography failed: {e}"

    ref_h, ref_w = ref_img.shape[:2]
    tgt_h, tgt_w = target_img.shape[:2]
    valid, area_fraction = validate_homography(H, (ref_h, ref_w), (tgt_h, tgt_w))
    if not valid:
        return None, (
            f"Rejected: homography is degenerate despite {inliers}/{good_matches} reported inliers "
            f"(warped reference footprint covers only {area_fraction:.1%} of the target image, "
            f"below the {MIN_AREA_FRACTION:.0%} validity threshold). This usually means SIFT matched "
            f"repetitive features (e.g. parallel bay lines) rather than true corresponding points."
        )

    coco = load_site_coco(reference_site)
    if not coco:
        return None, f"No bay annotation found for site '{reference_site}'."
    ref_bays, (coco_w, coco_h), _ = load_space_polygons_from_json(coco)
    sx, sy = ref_w / coco_w, ref_h / coco_h
    ref_bays = [rescale_poly(p, sx, sy) for p in ref_bays]

    warped_bays = []
    for bay in ref_bays:
        flat = [c for pt in bay for c in pt]
        warped_bays.append(warp_polygon(flat, H))

    status = (
        f"Homography accepted: {inliers}/{good_matches} RANSAC inliers, "
        f"warped reference footprint covers {area_fraction:.1%} of the target image. "
        f"Transferred {len(warped_bays)} bay polygons from '{reference_site}' without manual re-annotation."
    )
    return warped_bays, status


def list_sample_images():
    if not os.path.isdir(SAMPLE_IMAGES_DIR):
        return []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(SAMPLE_IMAGES_DIR, e)))
    return sorted(files)


# ---------------------------------------------------------------------------
# GPS-based automatic site identification
# (suggested by Ben Bartlett: read image metadata and auto-select the correct
# manually annotated site, rather than requiring a manual dropdown)
#
# Reuses the existing verbatim-ported functions above unchanged:
#   - direct match: load_site_coco + load_space_polygons_from_json + rescale_poly
#   - repeat-visit match: transfer_annotation (SIFT+RANSAC homography, unchanged)
# ---------------------------------------------------------------------------
DIMENSION_MATCH_TOLERANCE = 0.02  # 2% -- treat near-identical resolutions as "same capture setup"


def resolve_annotation_candidate(candidate_site_keys, image_rgb):
    """
    Given a shortlist of annotation site_keys physically located at the GPS-matched
    car park (e.g. ["Foundation", "Foundation118m"] -- same site, two different
    capture sessions), decide which applies to this new image and how:

      1. DIRECT MATCH -- this image's resolution closely matches an existing
         annotation's own COCO resolution. No CV needed; rescale and reuse.
      2. HOMOGRAPHY TRANSFER -- resolution differs (different day / altitude /
         camera position). Try transfer_annotation() against each candidate in
         turn and keep the first one that is accepted (passes the existing
         validity check in transfer_annotation/validate_homography).

    This is the "different days of the same car park" case Ben asked the demo
    to show: GPS narrows down WHICH car park, this function works out WHICH of
    that car park's saved annotations fits this specific photo.

    Returns: (bay_polys_in_image_coords, chosen_site_key, method, message)
    """
    img_h, img_w = image_rgb.shape[:2]

    # Collect each candidate's stored COCO resolution up front, so we can tell
    # whether dimension-matching is even a meaningful disambiguator for this
    # family. If two candidates (e.g. Foundation / Foundation118m) were both
    # exported from Roboflow at the SAME fixed size (a common Roboflow
    # default such as 640x640), their stored width/height are identical --
    # dimension matching can never tell them apart, and blindly trusting
    # "first in the list" would silently always pick the wrong one for half
    # of all real photos. In that case, skip Pass 1 entirely and let Pass 2's
    # homography scoring (which looks at actual image content, not just
    # metadata numbers) make the real decision instead.
    candidate_dims = {}
    for site_key in candidate_site_keys:
        coco = load_site_coco(site_key)
        if not coco:
            continue
        try:
            _, (coco_w, coco_h), _ = load_space_polygons_from_json(coco)
        except (KeyError, IndexError):
            continue
        if coco_w and coco_h:
            candidate_dims[site_key] = (coco_w, coco_h)

    dims_are_distinguishable = len(set(candidate_dims.values())) == len(candidate_dims) and len(candidate_dims) > 1

    # --- Pass 1: direct resolution match (fast path, no CV) ---
    # Only attempted when candidates actually have distinct stored resolutions,
    # OR when there's only one candidate for this site (no ambiguity possible).
    if dims_are_distinguishable or len(candidate_site_keys) == 1:
        for site_key in candidate_site_keys:
            if site_key not in candidate_dims:
                continue
            coco_w, coco_h = candidate_dims[site_key]
            dw = abs(img_w - coco_w) / coco_w
            dh = abs(img_h - coco_h) / coco_h
            if dw <= DIMENSION_MATCH_TOLERANCE and dh <= DIMENSION_MATCH_TOLERANCE:
                coco = load_site_coco(site_key)
                bay_polys, _, (n_seg, n_box) = load_space_polygons_from_json(coco)
                sx, sy = img_w / coco_w, img_h / coco_h
                bay_polys = [rescale_poly(p, sx, sy) for p in bay_polys]
                return (
                    bay_polys,
                    site_key,
                    "direct",
                    f"Image resolution ({img_w}x{img_h}) matches the existing '{site_key}' "
                    f"annotation ({coco_w}x{coco_h}) -- reused directly, no transfer needed.",
                )

    # --- Pass 2: no direct match -- try homography transfer against EVERY
    #     candidate reference image, and keep whichever produces the BEST
    #     (highest inliers x footprint-coverage) valid result.
    #
    #     IMPORTANT: this does NOT stop at the first candidate that merely
    #     passes validate_homography's area-fraction threshold. Because
    #     "Foundation" and "Foundation118m" are the SAME physical car park
    #     just at different altitudes, SIFT can find enough real (not just
    #     repetitive-line) matches between them that a technically-valid but
    #     WRONG homography can pass the threshold for the wrong candidate.
    #     Scoring every candidate and keeping the strongest one avoids
    #     "Foundation" incorrectly winning over "Foundation118m" (or vice
    #     versa) just because it happened to be tried first. ---
    best = None  # (score, warped_bays, site_key, status_msg)
    failures = []
    for site_key in candidate_site_keys:
        ref_path = REFERENCE_IMAGES.get(site_key)
        if not ref_path or not os.path.exists(ref_path):
            failures.append(f"{site_key}: no reference image found.")
            continue

        ref_img = cv2.imread(ref_path)
        target_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        ref_gray, scale_ref = _to_gray_downscaled(ref_img, DETECT_MAX_DIM)
        target_gray, scale_target = _to_gray_downscaled(target_img, DETECT_MAX_DIM)

        try:
            H, inliers, good_matches = compute_homography(ref_gray, target_gray, scale_ref, scale_target)
        except RuntimeError as e:
            failures.append(f"{site_key}: {e}")
            continue

        ref_h, ref_w = ref_img.shape[:2]
        tgt_h, tgt_w = target_img.shape[:2]
        valid, area_fraction = validate_homography(H, (ref_h, ref_w), (tgt_h, tgt_w))
        if not valid:
            failures.append(
                f"{site_key}: homography rejected (only {area_fraction:.1%} footprint coverage, "
                f"likely matched repetitive bay-line features rather than true correspondences)."
            )
            continue

        # Score by inliers x coverage -- rewards both a well-matched geometry
        # AND a plausible footprint, so a marginal, barely-valid match from
        # the wrong candidate can't beat a strong match from the right one.
        score = inliers * area_fraction

        coco = load_site_coco(site_key)
        if not coco:
            failures.append(f"{site_key}: no bay annotation JSON found.")
            continue
        ref_bays, (coco_w, coco_h), _ = load_space_polygons_from_json(coco)
        sx, sy = ref_w / coco_w, ref_h / coco_h
        ref_bays = [rescale_poly(p, sx, sy) for p in ref_bays]
        warped_bays = [warp_polygon([c for pt in bay for c in pt], H) for bay in ref_bays]

        status_msg = (
            f"Homography accepted: {inliers}/{good_matches} RANSAC inliers, "
            f"warped reference footprint covers {area_fraction:.1%} of the target image. "
            f"Transferred {len(warped_bays)} bay polygons from '{site_key}'."
        )

        if best is None or score > best[0]:
            best = (score, warped_bays, site_key, status_msg)

    if best is not None:
        _, warped_bays, site_key, status_msg = best
        return (
            warped_bays,
            site_key,
            "homography",
            f"No matching resolution found -- transferred the '{site_key}' annotation "
            f"(best match among {len(candidate_site_keys)} candidate(s) for this site). {status_msg}",
        )

    fail_msg = "; ".join(failures) if failures else "no usable reference images for this site."
    return None, None, None, f"Could not match this image to any existing annotation. {fail_msg}"



def identify_site_from_image(pil_image, image_rgb):
    """
    Full auto-detect flow: read GPS EXIF -> match against known car park
    boundaries (point-in-polygon against the UL KML) -> resolve which of that
    site's saved annotations applies to this specific image.

    Returns: (bay_polys, chosen_site_key, kml_display_name, method, message)
    All fields except `message` are None if identification failed at any stage.
    """
    gps = gps_utils.extract_gps(pil_image)
    if gps is None:
        return None, None, None, None, (
            "No GPS metadata found in this image's EXIF. Auto-detect requires the "
            "original DJI Mini 3 photo (GPS is stripped by many messaging apps and "
            "screenshot tools, and by Streamlit's own image widget if re-saved) -- "
            "use the manual tabs instead."
        )

    lat, lon, altitude = gps
    match = site_geo.identify_site_candidates(lat, lon)
    if match is None:
        return None, None, None, None, (
            f"GPS position ({lat:.6f}, {lon:.6f}) does not fall inside any car park "
            f"with an existing bay annotation. Use the manual tabs instead."
        )

    kml_name, display_name, candidate_keys = match
    bay_polys, chosen_key, method, resolve_msg = resolve_annotation_candidate(candidate_keys, image_rgb)
    if bay_polys is None:
        return None, None, display_name, None, f"GPS matched this image to {display_name}, but {resolve_msg}"

    alt_note = f", altitude {altitude:.1f} m" if altitude else ""
    message = f"GPS ({lat:.6f}, {lon:.6f}{alt_note}) matched to {display_name}. {resolve_msg}"
    return bay_polys, chosen_key, display_name, method, message


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Drone-Based Parking Occupancy Monitoring", layout="wide")

st.markdown(
    """
    <style>
    .hero { background: linear-gradient(135deg, #1E2761 0%, #141A3D 100%);
        border-radius: 14px; padding: 26px 30px; margin-bottom: 22px; color: white; }
    .hero h1 { color: #FFFFFF; margin: 0 0 6px 0; font-size: 26px; }
    .hero p { color: #CADCFC; margin: 0; font-size: 14.5px; line-height: 1.5; }
    div.stButton > button { background-color: #1E2761; color: white; font-weight: 600; border-radius: 8px; }
    div.stButton > button:hover { background-color: #2A3583; color: white; }
    </style>
    <div class="hero">
      <h1>Drone-Based Parking Occupancy Monitoring</h1>
      <p>Runs the project's actual detection script (VisDrone-trained YOLOv11x, full-frame
      inference with the same deduplication logic) and the actual occupancy script (Sutherland-Hodgman polygon
      overlap) from the MSc thesis <i>"Low-Cost Drone-Based Parking Occupancy Monitoring for
      Smart Campus Mobility Management"</i> — University of Limerick.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab0, tab1, tab2 = st.tabs([
    "Auto-detect (GPS)",
    "Detect on a known image",
    "New photo of an existing car park (auto-transfer annotation)",
])

with tab0:
    st.markdown(
        "Upload a **DJI Mini 3 photo directly** (not a re-saved or screenshotted copy -- those "
        "strip GPS metadata). The site is identified automatically from the photo's GPS position "
        "against the UL car park boundaries -- no need to say which car park this is, or whether "
        "it needs a fresh overlay or a homography transfer from an earlier visit. This uses the "
        "same point-in-polygon method as the thesis's multi-site validation (Section 4.4)."
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### 1 · Upload a drone photo")
        auto_upload = st.file_uploader(
            "Aerial image (original file, GPS EXIF required)", type=["jpg", "jpeg", "png"], key="up0"
        )
        auto_pil_image = None
        auto_image_rgb = None
        if auto_upload is not None:
            auto_pil_image = Image.open(auto_upload)
            auto_image_rgb = np.array(auto_pil_image.convert("RGB"))
            st.image(auto_image_rgb, caption="Uploaded image", use_container_width=True)

        st.markdown("#### 2 · Configure")
        occ_frac0 = st.slider(
            "Occupancy threshold (bay is occupied if a car covers >= this fraction of the bay polygon)",
            0.05, 0.6, OCC_FRAC_DEFAULT, 0.01, key="occ0",
        )
        auto_clicked = st.button("▶  Auto-Detect & Run", key="run0")

    with col2:
        st.markdown("#### 3 · Results")
        if auto_clicked:
            if auto_image_rgb is None:
                st.warning("Please upload an image first.")
            else:
                with st.spinner("Reading GPS, identifying site, running detection..."):
                    bay_polys, site_key, display_name, method, id_message = identify_site_from_image(
                        auto_pil_image, auto_image_rgb
                    )
                    kept, removed = run_detection_on_image(auto_image_rgb)
                    det_img = draw_detections(auto_image_rgb, kept)
                    vehicles = [d for d in kept if d.category_group == "vehicle"]

                lines = [
                    id_message,
                    "",
                    f"Detected {len(kept)} kept object(s) ({len(vehicles)} vehicles), "
                    f"{len(removed)} duplicate(s) removed, at confidence >= {CONFIDENCE_THRESHOLD}.",
                ]

                occ_img = None
                if bay_polys:
                    vehicle_boxes = [[d.x1, d.y1, d.x2, d.y2] for d in vehicles]
                    occ_img, occupied, available, capacity = compute_occupancy_map(
                        auto_image_rgb, bay_polys, vehicle_boxes, occ_frac0
                    )
                    method_label = {"direct": "direct overlay", "homography": "homography transfer"}.get(
                        method, method
                    )
                    lines.append(
                        f"Bay-level occupancy via {method_label} ({capacity} bays): "
                        f"{occupied} occupied, {available} available "
                        f"(occupancy threshold = {occ_frac0})."
                    )
                else:
                    lines.append("No bay-level occupancy computed -- see message above.")

                c1, c2 = st.columns(2)
                with c1:
                    st.image(det_img, caption="Detection output", use_container_width=True)
                with c2:
                    if occ_img is not None:
                        st.image(
                            occ_img, caption="Bay-level occupancy (green=free, red=occupied)",
                            use_container_width=True,
                        )
                st.text_area("Site identification & summary", "\n".join(lines), height=140)

    st.caption(
        "No GPS in the photo, or the site has no saved annotation yet? Use the manual tabs "
        "instead -- nothing about them has changed."
    )

with tab1:
    site_choices = ["(none — detection only)"] + list_annotation_sites()
    sample_choices = list_sample_images()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### 1 · Choose an image")
        uploaded = st.file_uploader("Upload an aerial image", type=["jpg", "jpeg", "png"], key="up1")
        sample_pick = st.selectbox("...or pick a sample image", ["(none)"] + sample_choices, key="sample1")

        image_rgb = None
        if uploaded is not None:
            image_rgb = np.array(Image.open(uploaded).convert("RGB"))
        elif sample_pick != "(none)":
            img_bgr = cv2.imread(sample_pick)
            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if image_rgb is not None:
            st.image(image_rgb, caption="Selected image", use_container_width=True)

        st.markdown("#### 2 · Configure")
        site_pick = st.selectbox("Bay annotation (for occupancy calculation)", site_choices, key="site1")
        occ_frac1 = st.slider(
            "Occupancy threshold (bay is occupied if a car covers >= this fraction of the bay polygon)",
            0.05, 0.6, OCC_FRAC_DEFAULT, 0.01, key="occ1",
        )
        run_clicked = st.button("▶  Run Detection", key="run1")

    with col2:
        st.markdown("#### 3 · Results")
        if run_clicked:
            if image_rgb is None:
                st.warning("Please upload or select an image first.")
            else:
                with st.spinner("Running detection..."):
                    kept, removed = run_detection_on_image(image_rgb)
                    det_img = draw_detections(image_rgb, kept)
                    vehicles = [d for d in kept if d.category_group == "vehicle"]

                lines = [
                    f"Detected {len(kept)} kept object(s) ({len(vehicles)} vehicles), "
                    f"{len(removed)} duplicate(s) removed, at confidence >= {CONFIDENCE_THRESHOLD}."
                ]

                occ_img = None
                if site_pick != "(none — detection only)":
                    coco = load_site_coco(site_pick)
                    if not coco:
                        lines.append(f"No bay annotation found for '{site_pick}'.")
                    else:
                        bay_polys, (coco_w, coco_h), (n_seg, n_box) = load_space_polygons_from_json(coco)
                        img_h, img_w = image_rgb.shape[:2]
                        sx, sy = img_w / coco_w, img_h / coco_h
                        bay_polys = [rescale_poly(p, sx, sy) for p in bay_polys]
                        vehicle_boxes = [[d.x1, d.y1, d.x2, d.y2] for d in vehicles]
                        occ_img, occupied, available, capacity = compute_occupancy_map(
                            image_rgb, bay_polys, vehicle_boxes, occ_frac1
                        )
                        lines.append(
                            f"Bay-level occupancy ({capacity} annotated bays, {n_seg} from true polygons, "
                            f"{n_box} from bbox fallback): {occupied} occupied, {available} available "
                            f"(occupancy threshold = {occ_frac1})."
                        )
                else:
                    lines.append("No site selected — showing detection only, no bay-level occupancy computed.")

                c1, c2 = st.columns(2)
                with c1:
                    st.image(det_img, caption="Detection output", use_container_width=True)
                with c2:
                    if occ_img is not None:
                        st.image(occ_img, caption="Bay-level occupancy (green=free, red=occupied)", use_container_width=True)
                st.text_area("Summary", "\n".join(lines), height=110)

    st.caption(
        "Bay-level occupancy is only computed when a matching annotation is selected. "
        "For a new image without an existing annotation, only vehicle detection is shown."
    )

with tab2:
    st.markdown(
        "Already have a bay annotation for one of these car parks, but a **new photo** from a "
        "different visit? Rather than re-annotating from scratch, this transfers the existing "
        "annotation onto the new image automatically, using SIFT feature matching and a "
        "RANSAC-estimated homography (Section 3.9.2 of the thesis)."
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### 1 · New image")
        new_upload = st.file_uploader("Upload a new photo of the car park", type=["jpg", "jpeg", "png"], key="up2")
        new_image_rgb = np.array(Image.open(new_upload).convert("RGB")) if new_upload is not None else None
        if new_image_rgb is not None:
            st.image(new_image_rgb, caption="New image", use_container_width=True)

        st.markdown("#### 2 · Which car park is this?")
        ref_site_pick = st.selectbox(
            "Existing car park (its saved annotation will be transferred)",
            list(REFERENCE_IMAGES.keys()), key="refsite",
        )
        occ_frac2 = st.slider("Occupancy threshold", 0.05, 0.6, OCC_FRAC_DEFAULT, 0.01, key="occ2")
        transfer_clicked = st.button("▶  Transfer Annotation & Detect", key="run2")

    with col2:
        st.markdown("#### 3 · Results")
        if transfer_clicked:
            if new_image_rgb is None:
                st.warning("Please upload a new image of the car park first.")
            else:
                with st.spinner("Matching features, estimating homography, and running detection..."):
                    warped_bays, status_msg = transfer_annotation(ref_site_pick, new_image_rgb)
                    det_img = occ_img = None
                    lines = [status_msg]
                    if warped_bays is not None:
                        kept, removed = run_detection_on_image(new_image_rgb)
                        det_img = draw_detections(new_image_rgb, kept)
                        vehicles = [d for d in kept if d.category_group == "vehicle"]
                        vehicle_boxes = [[d.x1, d.y1, d.x2, d.y2] for d in vehicles]
                        occ_img, occupied, available, capacity = compute_occupancy_map(
                            new_image_rgb, warped_bays, vehicle_boxes, occ_frac2
                        )
                        lines.append(
                            f"Detected {len(kept)} kept object(s) ({len(vehicles)} vehicles), "
                            f"{len(removed)} duplicate(s) removed."
                        )
                        lines.append(
                            f"Bay-level occupancy ({capacity} transferred bays): {occupied} occupied, "
                            f"{available} available (occupancy threshold = {occ_frac2})."
                        )

                c1, c2 = st.columns(2)
                with c1:
                    if det_img is not None:
                        st.image(det_img, caption="Detection output", use_container_width=True)
                with c2:
                    if occ_img is not None:
                        st.image(occ_img, caption="Bay-level occupancy (transferred)", use_container_width=True)
                st.text_area("Summary", "\n".join(lines), height=160)
