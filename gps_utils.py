"""
gps_utils.py
------------
Extracts GPS coordinates from a photo's EXIF metadata (DJI Mini 3 photos
write standard GPSInfo EXIF tags — this is the same metadata field Ben's
make_kml_from_exif_txt.py script reads to generate KML pins from flight
photos, confirming the tag is reliably present on this drone's output).

Note (thesis Section 3.4): the DJI Mini 3 does NOT reliably log
GimbalPitchDegree (camera tilt) — it always reports 0.00 degrees regardless
of true angle. This is a SEPARATE EXIF field from GPSInfo and is unaffected;
GPS position itself is unrelated to gimbal angle and is not subject to the
same limitation.
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def _dms_to_decimal(dms, ref):
    """Convert EXIF GPS (degrees, minutes, seconds) tuple to decimal degrees."""
    degrees, minutes, seconds = dms
    decimal = float(degrees) + float(minutes) / 60.0 + float(seconds) / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_gps(pil_image):
    """
    Returns (lat, lon, altitude_m_or_None) as decimal degrees, or None if the
    image has no GPS EXIF block at all (e.g. a screenshot, a re-saved/
    stripped copy, or a non-drone photo).
    """
    exif = pil_image.getexif() if hasattr(pil_image, "getexif") else None
    if not exif:
        return None

    gps_ifd = exif.get_ifd(0x8825) if hasattr(exif, "get_ifd") else None
    if not gps_ifd:
        # Fallback for older Pillow / non-standard EXIF layout
        raw_exif = pil_image._getexif() if hasattr(pil_image, "_getexif") else None
        if not raw_exif:
            return None
        gps_ifd = None
        for tag_id, value in raw_exif.items():
            if TAGS.get(tag_id) == "GPSInfo":
                gps_ifd = value
                break
        if not gps_ifd:
            return None

    gps_info = {}
    for tag_id, value in gps_ifd.items():
        tag = GPSTAGS.get(tag_id, tag_id)
        gps_info[tag] = value

    if "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
        return None

    lat = _dms_to_decimal(gps_info["GPSLatitude"], gps_info.get("GPSLatitudeRef", "N"))
    lon = _dms_to_decimal(gps_info["GPSLongitude"], gps_info.get("GPSLongitudeRef", "E"))

    altitude = None
    if "GPSAltitude" in gps_info:
        try:
            altitude = float(gps_info["GPSAltitude"])
        except (TypeError, ValueError):
            altitude = None

    return (lat, lon, altitude)
