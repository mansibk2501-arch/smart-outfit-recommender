import colorsys
import math
import numpy as np

COLOR_PALETTE = {
    "Pink": "#E7A1B0",
    "Blue": "#4169E1",
    "Black": "#202124",
    "White": "#FFFFFF",
    "Red": "#D62828",
    "Green": "#39805A",
    "Yellow": "#F4D35E",
    "Purple": "#9370DB",
    "Orange": "#F28C28",
    "Brown": "#8B5E3C",
    "Beige": "#D9C3A5",
    "Navy Blue": "#243B64",
    "Grey": "#A8ADB4",
    "Sage Green": "#A3B18A",
    "Maroon": "#800000",
    "Gold": "#D4AF37",
    "Teal": "#008080",
    "Lavender": "#C4A1FF",
    "Cream": "#FFF1D0",
    "Denim": "#315C8C",
}

def hex_to_rgb(hex_color):
    s = hex_color.strip().lstrip("#")
    return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    vals = [max(0, min(255, int(round(float(v))))) for v in rgb]
    return "#{:02X}{:02X}{:02X}".format(*vals)

def rgb_to_hsv(rgb):
    r, g, b = [float(v) / 255.0 for v in rgb]
    return colorsys.rgb_to_hsv(r, g, b)

def color_features(rgb1, rgb2):
    h1, s1, v1 = rgb_to_hsv(rgb1)
    h2, s2, v2 = rgb_to_hsv(rgb2)
    raw_hue = abs(h1 - h2)
    hue_dist = min(raw_hue, 1.0 - raw_hue)
    # Saturation/value distances and simple brightness contrast
    s_diff = abs(s1 - s2)
    v_diff = abs(v1 - v2)
    brightness1 = 0.299*rgb1[0] + 0.587*rgb1[1] + 0.114*rgb1[2]
    brightness2 = 0.299*rgb2[0] + 0.587*rgb2[1] + 0.114*rgb2[2]
    brightness_contrast = abs(brightness1 - brightness2) / 255.0
    neutral1 = 1.0 if s1 < 0.18 else 0.0
    neutral2 = 1.0 if s2 < 0.18 else 0.0
    neutral_pair = 1.0 if neutral1 or neutral2 else 0.0
    return [hue_dist, s_diff, v_diff, brightness_contrast, neutral_pair, min(s1, s2), min(v1, v2)]

def harmony_rule_score(rgb1, rgb2):
    hue_dist, s_diff, v_diff, contrast, neutral_pair, min_sat, min_val = color_features(rgb1, rgb2)
    # Rule-based pseudo-label source: neutrals pair broadly; otherwise prefer
    # analogous, complementary, and triadic-ish hue separations with some contrast.
    if neutral_pair:
        score = 0.78 + 0.15 * contrast
    else:
        # Hue distance measured as fraction of full wheel; 0.0 same, 0.5 opposite.
        peaks = [0.0, 1/12, 1/6, 1/3, 1/2]
        closeness = max(math.exp(-((hue_dist - p) / 0.075) ** 2) for p in peaks)
        score = 0.25 + 0.55 * closeness + 0.15 * contrast - 0.10 * s_diff
    return float(max(0.0, min(1.0, score)))

def nearest_palette_name(rgb):
    # Euclidean RGB distance is sufficient for a readable prototype.
    arr = np.array(rgb, dtype=float)
    best_name, best_dist = None, float("inf")
    for name, hx in COLOR_PALETTE.items():
        dist = float(np.linalg.norm(arr - np.array(hex_to_rgb(hx), dtype=float)))
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name
