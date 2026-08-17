"""
Drone-Based Parking Occupancy Monitoring — Live Demo
------------------------------------------------------
Runs the actual domain-matched (VisDrone-trained YOLOv11x) detector from the thesis
"Low-Cost Drone-Based Parking Occupancy Monitoring for Smart Campus Mobility Management"
on an uploaded aerial image, and computes real bay-level occupancy using the same
Sutherland-Hodgman polygon clipping + Shoelace area method described in the thesis
(Section 3.9, Occupancy and Availability Estimation).

To deploy:
1. Push this repo to a Hugging Face Space (SDK: Gradio).
2. Place your trained weights at model/best.pt
3. Place a COCO-format bay annotation JSON per reference image in annotations/
4. (Optional) Place a few sample images in sample_images/ for the dropdown gallery.
"""

import json
import os
import glob

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "model/best.pt"           # your trained VisDrone/YOLOv11x weights
ANNOTATIONS_DIR = "annotations"        # folder of COCO JSON bay annotations, one per site/image
SAMPLE_IMAGES_DIR = "sample_images"    # a few real example images for the demo gallery
CONFIDENCE_THRESHOLD = 0.45            # matches thesis Appendix B default
OVERLAP_THRESHOLD = 0.12               # matches thesis default (Car Park 2); use 0.30 for newer sites
VEHICLE_CLASSES = {"car", "van", "truck", "bus"}  # matches thesis Appendix B

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model weights not found at {MODEL_PATH}. "
                "Place your trained best.pt there before running."
            )
        _model = YOLO(MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# Geometry: Sutherland-Hodgman polygon clipping + Shoelace area
# (identical method to thesis Section 3.9)
# ---------------------------------------------------------------------------
def clip_polygon(subject, clip):
    """Clip `subject` polygon against a convex `clip` polygon (Sutherland-Hodgman)."""
    output = list(subject)
    for i in range(len(clip)):
        if not output:
            break
        A, B = clip[i], clip[(i + 1) % len(clip)]
        input_list = output
        output = []
        for j in range(len(input_list)):
            P, Q = input_list[j], input_list[(j + 1) % len(input_list)]
            side_p = (B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] - A[0])
            side_q = (B[0] - A[0]) * (Q[1] - A[1]) - (B[1] - A[1]) * (Q[0] - A[0])
            if side_p <= 0:
                output.append(P)
            if (side_p <= 0) != (side_q <= 0):
                t = side_p / (side_p - side_q) if (side_p - side_q) != 0 else 0
                output.append((P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])))
    return output


def shoelace_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def bbox_to_poly(x1, y1, x2, y2):
    return [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]


# ---------------------------------------------------------------------------
# Annotation loading (COCO JSON bay polygons)
# ---------------------------------------------------------------------------
def list_annotation_sites():
    if not os.path.isdir(ANNOTATIONS_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(ANNOTATIONS_DIR)
        if f.lower().endswith(".json")
    )


def load_bay_polygons(site_name, target_width=None, target_height=None):
    """Load bay polygons from a COCO-format annotation file (Roboflow export style).

    Roboflow exports store the image width/height it was annotated on in the
    'images' section of the COCO JSON. If the image being processed at runtime
    is a different resolution (e.g. the original full-resolution photo vs. a
    resized Roboflow export), the polygon coordinates must be scaled to match,
    or they will appear shrunk into one corner of the real image.
    """
    path = os.path.join(ANNOTATIONS_DIR, f"{site_name}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        coco = json.load(f)

    # Determine the annotation's original image size (Roboflow export resolution)
    images_meta = coco.get("images", [])
    ann_width, ann_height = None, None
    if images_meta:
        ann_width = images_meta[0].get("width")
        ann_height = images_meta[0].get("height")

    scale_x, scale_y = 1.0, 1.0
    if ann_width and ann_height and target_width and target_height:
        scale_x = target_width / ann_width
        scale_y = target_height / ann_height

    bays = []
    for ann in coco.get("annotations", []):
        seg = ann.get("segmentation")
        if not seg:
            continue
        # COCO polygon segmentation: flat list [x1, y1, x2, y2, ...] (possibly multiple parts)
        part = seg[0] if isinstance(seg[0], list) else seg
        pts = [
            (part[i] * scale_x, part[i + 1] * scale_y)
            for i in range(0, len(part), 2)
        ]
        bays.append(pts)
    return bays


def list_sample_images():
    if not os.path.isdir(SAMPLE_IMAGES_DIR):
        return []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(SAMPLE_IMAGES_DIR, e)))
    return sorted(files)


# ---------------------------------------------------------------------------
# Bay-boundary transfer across repeat visits (SIFT + RANSAC homography)
# Ported directly from the thesis's transfer_polygons.py (Section 3.9.2)
# ---------------------------------------------------------------------------
DETECT_MAX_DIM = 2000
MIN_MATCH_COUNT = 15
RANSAC_REPROJ_THRESHOLD = 5.0  # pixels, at full resolution
MIN_AREA_FRACTION = 0.15

# Each reference site's own sample image doubles as the homography reference image
REFERENCE_IMAGES = {
    "CarPark2": os.path.join(SAMPLE_IMAGES_DIR, "DJI_0028.JPG"),
    "Foundation": os.path.join(SAMPLE_IMAGES_DIR, "DJI_0079.JPG"),
    "Foundation118m": os.path.join(SAMPLE_IMAGES_DIR, "DJI_0084.JPG"),
    "KBS": os.path.join(SAMPLE_IMAGES_DIR, "DJI_0098.JPG"),
}


def _to_gray_downscaled(img_bgr, max_dim):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else img_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray, scale


def compute_homography(ref_gray, target_gray, scale_ref, scale_target):
    sift = cv2.SIFT_create(nfeatures=8000)
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)
    if des1 is None or des2 is None:
        raise RuntimeError("No SIFT features found in one of the images.")

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < MIN_MATCH_COUNT:
        raise RuntimeError(f"Only {len(good)} good matches found (need >= {MIN_MATCH_COUNT}). Images may not overlap enough.")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / scale_ref
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) / scale_target

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None:
        raise RuntimeError("Homography estimation failed (RANSAC returned None).")
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


def warp_bay_polygons(bays, H):
    warped_bays = []
    for bay in bays:
        pts = np.float32(bay).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        warped_bays.append([(float(p[0]), float(p[1])) for p in warped])
    return warped_bays


def transfer_annotation(reference_site, target_image_rgb):
    """Full pipeline: SIFT match reference site's image against a new target image,
    estimate + validate a homography, and warp that site's bay polygons onto the
    target image. Returns (warped_bays, status_message)."""
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

    ref_bays = load_bay_polygons(reference_site, target_width=ref_w, target_height=ref_h)
    if not ref_bays:
        return None, f"No bay annotation found for site '{reference_site}'."

    warped_bays = warp_bay_polygons(ref_bays, H)
    status = (
        f"Homography accepted: {inliers}/{good_matches} RANSAC inliers, "
        f"warped reference footprint covers {area_fraction:.1%} of the target image. "
        f"Transferred {len(warped_bays)} bay polygons from '{reference_site}' without manual re-annotation."
    )
    return warped_bays, status



# ---------------------------------------------------------------------------
# Core pipeline: detect -> draw boxes -> compute occupancy -> draw bay overlay
# ---------------------------------------------------------------------------
def _run_detection(img_bgr):
    """Run the domain-matched detector and return (detections, annotated_det_img_rgb)."""
    model = get_model()
    results = model.predict(img_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        if cls_name not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        detections.append((x1, y1, x2, y2, cls_name, conf))

    det_img = img_bgr.copy()
    for (x1, y1, x2, y2, cls_name, conf) in detections:
        cv2.rectangle(det_img, (int(x1), int(y1)), (int(x2), int(y2)), (80, 200, 120), 3)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(det_img, label, (int(x1), max(15, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 120), 1, cv2.LINE_AA)
    return detections, cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)


def _draw_occupancy(img_bgr, bays, detections, overlap_threshold):
    """Given bay polygons (already in this image's pixel coordinates) and detections,
    draw the occupancy overlay and return (overlay_rgb, occupied_count, total)."""
    occ_img = img_bgr.copy()
    occupied_count = 0
    for bay in bays:
        bay_area = shoelace_area(bay) or 1.0
        inter_area = 0.0
        for (x1, y1, x2, y2, _, _) in detections:
            clipped = clip_polygon(bay, bbox_to_poly(x1, y1, x2, y2))
            if len(clipped) >= 3:
                inter_area += shoelace_area(clipped)
        ratio = min(inter_area / bay_area, 1.0)
        is_occupied = ratio >= overlap_threshold
        color = (60, 90, 220) if is_occupied else (90, 190, 90)  # BGR: red if occupied, green if empty
        if is_occupied:
            occupied_count += 1
        pts = np.array(bay, dtype=np.int32).reshape((-1, 1, 2))
        overlay = occ_img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.35, occ_img, 0.65, 0, occ_img)
        cv2.polylines(occ_img, [pts], True, color, 2)

    total = len(bays)
    available = total - occupied_count
    for color, thick in [((255, 255, 255), 3), ((30, 30, 30), 1)]:
        cv2.putText(occ_img, f"Occupied: {occupied_count}/{total}  |  Available: {available}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thick, cv2.LINE_AA)
    return cv2.cvtColor(occ_img, cv2.COLOR_BGR2RGB), occupied_count, total


def run_pipeline(image, site_name, overlap_threshold):
    if image is None:
        return None, None, "Please upload or select an image first."

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detections, det_img_rgb = _run_detection(img_bgr)
    status_lines = [f"Detected {len(detections)} vehicle(s) at confidence >= {CONFIDENCE_THRESHOLD}."]

    occ_img_rgb = None
    if site_name and site_name != "(none — detection only)":
        img_h, img_w = img_bgr.shape[:2]
        bays = load_bay_polygons(site_name, target_width=img_w, target_height=img_h)
        if not bays:
            status_lines.append(f"No bay annotation found for '{site_name}'.")
        else:
            occ_img_rgb, occupied_count, total = _draw_occupancy(img_bgr, bays, detections, overlap_threshold)
            available = total - occupied_count
            status_lines.append(
                f"Bay-level occupancy ({total} annotated bays): "
                f"{occupied_count} occupied, {available} available "
                f"(overlap threshold = {overlap_threshold})."
            )
    else:
        status_lines.append("No site selected — showing detection only, no bay-level occupancy computed.")

    return det_img_rgb, occ_img_rgb, "\n".join(status_lines)


def run_transfer_pipeline(new_image, reference_site, overlap_threshold):
    """New-image-of-an-existing-car-park flow: transfer that site's bay annotation
    onto the new image via homography, then detect + compute occupancy."""
    if new_image is None:
        return None, None, "Please upload a new image of the car park first."
    if not reference_site:
        return None, None, "Please select which existing car park this new image is of."

    warped_bays, status_msg = transfer_annotation(reference_site, new_image)
    if warped_bays is None:
        return None, None, status_msg

    img_bgr = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    detections, det_img_rgb = _run_detection(img_bgr)
    occ_img_rgb, occupied_count, total = _draw_occupancy(img_bgr, warped_bays, detections, overlap_threshold)
    available = total - occupied_count

    full_status = (
        f"{status_msg}\n\n"
        f"Detected {len(detections)} vehicle(s) at confidence >= {CONFIDENCE_THRESHOLD}.\n"
        f"Bay-level occupancy ({total} transferred bays): {occupied_count} occupied, "
        f"{available} available (overlap threshold = {overlap_threshold})."
    )
    return det_img_rgb, occ_img_rgb, full_status


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
SITE_CHOICES = ["(none — detection only)"] + list_annotation_sites()
SAMPLE_CHOICES = list_sample_images()

# Custom theme matching the thesis presentation palette (navy / amber / sage)
THEME = gr.themes.Default(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
).set(
    button_primary_background_fill="#1E2761",
    button_primary_background_fill_hover="#2A3583",
    button_primary_text_color="#FFFFFF",
    block_title_text_weight="600",
    block_border_width="1px",
    block_shadow="0 1px 3px rgba(30,39,97,0.08)",
    body_background_fill="#F7F8FC",
)

CUSTOM_CSS = """
#hero {
    background: linear-gradient(135deg, #1E2761 0%, #141A3D 100%);
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 18px;
    color: white !important;
}
#hero h1 { color: #FFFFFF !important; margin: 0 0 6px 0; font-size: 26px; }
#hero p { color: #CADCFC !important; margin: 0; font-size: 14.5px; line-height: 1.5; }
.gr-button-primary { font-weight: 600 !important; }
#footer-note { color: #5B6472; font-size: 12.5px; text-align: center; margin-top: 10px; }
.gradio-container { max-width: 1200px !important; margin: auto !important; }
"""

with gr.Blocks(
    title="Drone-Based Parking Occupancy Monitoring",
) as demo:
    gr.HTML(
        """
        <div id="hero">
          <h1>Drone-Based Parking Occupancy Monitoring</h1>
          <p>Live demo of the domain-matched (VisDrone-trained YOLOv11x) detection pipeline from the
          MSc thesis <i>"Low-Cost Drone-Based Parking Occupancy Monitoring for Smart Campus Mobility
          Management"</i> — University of Limerick.</p>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("Detect on a known image"):
            with gr.Row():
                with gr.Column(scale=1, min_width=340):
                    with gr.Group():
                        gr.Markdown("### 1 · Choose an image")
                        image_input = gr.Image(type="numpy", label="Aerial image", height=260)
                        if SAMPLE_CHOICES:
                            sample_dropdown = gr.Dropdown(
                                choices=SAMPLE_CHOICES, label="...or pick a sample image", value=None
                            )
                    with gr.Group():
                        gr.Markdown("### 2 · Configure")
                        site_dropdown = gr.Dropdown(
                            choices=SITE_CHOICES,
                            value=SITE_CHOICES[0],
                            label="Bay annotation (for occupancy calculation)",
                        )
                        threshold_slider = gr.Slider(
                            minimum=0.05, maximum=0.6, value=OVERLAP_THRESHOLD, step=0.01,
                            label="Overlap threshold (thesis default: 0.12 at Car Park 2, 0.30 at newer sites)",
                        )
                    run_button = gr.Button("▶  Run Detection", variant="primary", size="lg")

                with gr.Column(scale=2, min_width=560):
                    gr.Markdown("### 3 · Results")
                    with gr.Row():
                        det_output = gr.Image(label="Detection output", height=340)
                        occ_output = gr.Image(label="Bay-level occupancy", height=340)
                    status_output = gr.Textbox(label="Summary", lines=4)

            if SAMPLE_CHOICES:
                def load_sample(path):
                    if not path:
                        return None
                    img_bgr = cv2.imread(path)
                    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                sample_dropdown.change(fn=load_sample, inputs=sample_dropdown, outputs=image_input)

            run_button.click(
                fn=run_pipeline,
                inputs=[image_input, site_dropdown, threshold_slider],
                outputs=[det_output, occ_output, status_output],
            )

            gr.HTML(
                """
                <p id="footer-note">Bay-level occupancy is only computed when a matching annotation is selected.
                For a new image without an existing annotation, only vehicle detection is shown.</p>
                """
            )

        with gr.Tab("New photo of an existing car park (auto-transfer annotation)"):
            gr.Markdown(
                "Already have a bay annotation for one of these car parks, but a **new photo** from a "
                "different visit? Rather than re-annotating from scratch, this transfers the existing "
                "annotation onto the new image automatically, using SIFT feature matching and a "
                "RANSAC-estimated homography (Section 3.9.2 of the thesis)."
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=340):
                    with gr.Group():
                        gr.Markdown("### 1 · New image")
                        transfer_image_input = gr.Image(type="numpy", label="New photo of the car park", height=260)
                    with gr.Group():
                        gr.Markdown("### 2 · Which car park is this?")
                        transfer_site_dropdown = gr.Dropdown(
                            choices=list(REFERENCE_IMAGES.keys()),
                            value=list(REFERENCE_IMAGES.keys())[0] if REFERENCE_IMAGES else None,
                            label="Existing car park (its saved annotation will be transferred)",
                        )
                        transfer_threshold_slider = gr.Slider(
                            minimum=0.05, maximum=0.6, value=OVERLAP_THRESHOLD, step=0.01,
                            label="Overlap threshold",
                        )
                    transfer_button = gr.Button("▶  Transfer Annotation & Detect", variant="primary", size="lg")

                with gr.Column(scale=2, min_width=560):
                    gr.Markdown("### 3 · Results")
                    with gr.Row():
                        transfer_det_output = gr.Image(label="Detection output", height=340)
                        transfer_occ_output = gr.Image(label="Bay-level occupancy (transferred)", height=340)
                    transfer_status_output = gr.Textbox(label="Summary", lines=6)

            transfer_button.click(
                fn=run_transfer_pipeline,
                inputs=[transfer_image_input, transfer_site_dropdown, transfer_threshold_slider],
                outputs=[transfer_det_output, transfer_occ_output, transfer_status_output],
            )



if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS)
