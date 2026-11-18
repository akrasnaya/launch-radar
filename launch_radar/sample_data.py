"""Small hand-written corpus. Cohort and build fields are synthetic."""

from __future__ import annotations

from launch_radar.models import FeedbackRecord


def sample_records() -> list[FeedbackRecord]:
    rows = [
        ("fb-001", "After build 4.8 the vehicle lost its lane and required a manual takeover.", "pilot", "4.8.12"),
        ("fb-002", "Localization drifted near the barrier and the driver took over.", "pilot", "4.8.12"),
        ("fb-003", "Lane position jumped during the pilot route; emergency takeover.", "pilot", "4.8.12"),
        ("fb-004", "One lane drift on the previous build, no takeover needed.", "holdout", "4.8.11"),
        ("fb-005", "The app screen is confusing and the close button is hard to find.", "pilot", "4.8.12"),
        ("fb-006", "Confusing interface: I cannot find the close button on the screen.", "pilot", "4.8.12"),
        ("fb-007", "The close button in the app interface is confusing.", "holdout", "4.8.11"),
        ("fb-008", "Bluetooth pairing fails and the robot stays offline.", "pilot", "4.8.12"),
        ("fb-009", "Charging stopped twice and the battery sensor reported an error.", "pilot", "4.8.12"),
        ("fb-010", "Support sent the same troubleshooting steps and gave no response after that.", "holdout", "4.8.11"),
        ("fb-011", "Brake failure warning appeared before an unsafe stop.", "pilot", "4.8.12"),
        ("fb-012", "Navigation map looks good and the route works as expected.", "holdout", "4.8.11"),
    ]
    return [
        FeedbackRecord(
            record_id=record_id,
            text=text,
            source="sample",
            cohort=cohort,
            build=build,
            synthetic=True,
        )
        for record_id, text, cohort, build in rows
    ]
