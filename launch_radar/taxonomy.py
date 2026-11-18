"""Closed taxonomy for the open-data prototype."""

from __future__ import annotations

TOPICS: dict[str, tuple[str, ...]] = {
    "navigation": (
        "lane",
        "localization",
        "position",
        "drift",
        "map",
        "boundary",
        "steering",
    ),
    "perception_safety": (
        "collision",
        "crash",
        "brake",
        "pedestrian",
        "obstacle",
        "unsafe",
        "takeover",
        "emergency",
    ),
    "connectivity": (
        "bluetooth",
        "wifi",
        "offline",
        "connection",
        "cloud",
        "pairing",
    ),
    "hardware": (
        "battery",
        "charging",
        "sensor",
        "camera",
        "wheel",
        "overheat",
    ),
    "ux": (
        "screen",
        "button",
        "confusing",
        "interface",
        "app",
        "instruction",
    ),
    "support": (
        "support",
        "warranty",
        "dealer",
        "ticket",
        "no response",
    ),
}

SAFETY_TERMS = (
    "collision",
    "crash",
    "pedestrian",
    "unsafe",
    "emergency",
    "takeover",
    "brake failure",
)

SEVERITY_WEIGHT = {
    "critical": 5,
    "high": 3,
    "medium": 2,
    "low": 1,
}
