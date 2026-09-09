import json
import os

COLOR_ABOVE = "#2a78d6"
COLOR_BELOW = "#e34948"
COLOR_ERRORBAR = "#52514e"
COLOR_BASELINE = "#c3c2b7"
COLOR_GRID = "#e1e0d9"
COLOR_TEXT_MUTED = "#898781"
COLOR_TEXT_PRIMARY = "#0b0b0b"
SURFACE = "#ffffff"


def load_display_names(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def resolve_display(name: str, display_names: dict) -> str:
    return display_names.get(name, name)


def named_row(raw_name, value, std, display_names: dict) -> dict:
    return {"name": resolve_display(raw_name, display_names), "value": float(value), "std": float(std)}


def task_short_label(name: str) -> str:
    return name.replace("_", " ")
