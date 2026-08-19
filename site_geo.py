"""
site_geo.py
-----------
Loads the UL car park KML boundaries and provides GPS point-in-polygon
matching, so a new image can be automatically assigned to the correct
annotated site rather than requiring a manual dropdown selection.

This reuses the same point-in-polygon site-identification approach used in
the thesis's multi-site validation (Section 4.4): distance-to-centroid was
found unreliable for closely situated car parks, so a true point-in-polygon
test against each site's KML boundary is used instead.
"""

import re
import os

KML_PATH = os.environ.get("UL_KML_PATH", "UL_car_parks.kml")

# ---------------------------------------------------------------------------
# Map KML placemark name -> the annotation site_key(s) physically located there.
# Verified against thesis Table 5.2 areas (KML area vs. reported area, m^2):
#   car park 1  -> 21,460.8  (thesis: Foundation, 21,461)   MATCH
#   car park 2  ->  5,298.5  (thesis: Schuman,     5,299)   MATCH
#   car park 14 ->  4,029.1  (thesis: KBS,          4,029)  MATCH
#
# A KML placemark can map to MORE than one annotation key when the same
# physical car park was annotated more than once (e.g. Foundation was
# annotated both at native resolution and again at 118.9m). GPS alone
# cannot distinguish between these — see resolve_annotation_candidate()
# in app.py, which tries each candidate and keeps whichever produces a
# valid direct match or homography transfer.
# ---------------------------------------------------------------------------
KML_NAME_TO_SITE_KEYS = {
    "car park 1": ["Foundation", "Foundation118m"],
    "car park 2": ["CarPark2"],
    "car park 14": ["KBS"],
}

KML_NAME_TO_DISPLAY = {
    "car park 1": "Foundation Car Park",
    "car park 2": "Car Park 2 (Schuman)",
    "car park 14": "Kemmy Business School Car Park",
}


def _parse_kml_polygons(kml_path):
    """Returns {kml_name: [(lon, lat), ...]} for every Placemark with coordinates."""
    with open(kml_path, "r", encoding="utf-8") as f:
        kml = f.read()

    placemarks = re.findall(r"<Placemark.*?</Placemark>", kml, re.S)
    polygons = {}
    for pm in placemarks:
        name_match = re.search(r"<name>(.*?)</name>", pm)
        coords_match = re.search(r"<coordinates>\s*(.*?)\s*</coordinates>", pm, re.S)
        if not name_match or not coords_match:
            continue
        name = name_match.group(1).strip()
        pts = []
        for tok in coords_match.group(1).split():
            parts = tok.split(",")
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                pts.append((lon, lat))
        if len(pts) >= 3:
            polygons[name] = pts
    return polygons


def _point_in_polygon(lon, lat, polygon):
    """Standard ray-casting point-in-polygon test. polygon = [(lon, lat), ...]."""
    n = len(polygon)
    inside = False
    x, y = lon, lat
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if y > min(y1, y2):
            if y <= max(y1, y2):
                if x <= max(x1, x2):
                    if y1 != y2:
                        xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    if x1 == x2 or x <= xinters:
                        inside = not inside
        x1, y1 = x2, y2
    return inside


# Load once at import time
try:
    ALL_POLYGONS = _parse_kml_polygons(KML_PATH)
except FileNotFoundError:
    ALL_POLYGONS = {}

# Only the sites we actually have annotations for, restricted from the full 17
ANNOTATED_KML_NAMES = [k for k in KML_NAME_TO_SITE_KEYS if k in ALL_POLYGONS]


def identify_site_candidates(lat, lon):
    """
    Given a GPS point, return (kml_name, display_name, candidate_site_keys)
    for whichever annotated car park boundary contains this point, checking
    ONLY the sites with existing annotations (not all 17 UL car parks) —
    if the drone photo is of a site with no annotation yet, this correctly
    returns no match rather than forcing a wrong one.

    Returns None if the point falls inside no annotated site's boundary.
    """
    for kml_name in ANNOTATED_KML_NAMES:
        polygon = ALL_POLYGONS[kml_name]
        if _point_in_polygon(lon, lat, polygon):
            return (
                kml_name,
                KML_NAME_TO_DISPLAY.get(kml_name, kml_name),
                KML_NAME_TO_SITE_KEYS[kml_name],
            )
    return None
