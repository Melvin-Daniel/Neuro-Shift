"""
Rule-based sleep tips from lifestyle inputs (not used as ML features).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any


def _hours_between_bed_and_wake(bedtime: time, wake: time) -> float:
    """Approximate hours in bed (handles crossing midnight)."""
    today = datetime(2000, 1, 1)
    t_bed = datetime.combine(today, bedtime)
    t_wake = datetime.combine(today, wake)
    if t_wake <= t_bed:
        t_wake += timedelta(days=1)
    return (t_wake - t_bed).total_seconds() / 3600.0


def collect_tips(
    *,
    sleep_duration_h: float,
    stress: int,
    exercise_min: int,
    screen_min: int,
    caffeine: str,
    mood: str,
    interruptions: str,
    bedtime: time | None,
    wake: time | None,
    predicted_quality: str | None = None,
) -> list[str]:
    tips: list[str] = []
    seen: set[str] = set()

    def add(msg: str) -> None:
        if msg not in seen:
            seen.add(msg)
            tips.append(msg)

    if sleep_duration_h < 6:
        add("Aim for 7–9 hours of sleep when possible; short sleep often tracks with lower recovery.")
    elif sleep_duration_h > 9.5:
        add("Very long time in bed can sometimes reflect poor sleep efficiency; note if you still feel tired.")

    if stress >= 7:
        add("High stress can fragment sleep; try 5 minutes of slow breathing or a short wind-down routine before bed.")
    elif stress <= 3:
        add("Lower daytime stress is associated with easier sleep maintenance—keep habits that help you stay balanced.")

    if exercise_min < 20:
        add("Even 20–30 minutes of light movement most days can support deeper sleep.")
    elif exercise_min > 90:
        add("If intense exercise is late in the evening, try shifting hard workouts earlier so your body can cool down before bed.")

    if screen_min >= 45:
        add("Try reducing screen time by 30 minutes before bed; dim displays and use night mode if you must use devices.")
    elif screen_min >= 20:
        add("Consider a short screen-free buffer (15–30 minutes) before lights out.")

    caffeine_lower = (caffeine or "").lower()
    if "high" in caffeine_lower or "moderate" in caffeine_lower:
        add("Limit caffeine after mid-afternoon; its effects can still influence sleep onset at night.")

    mood_lower = (mood or "").lower()
    if "anxious" in mood_lower or "sad" in mood_lower:
        add("A brief journal or gratitude note before bed can help quiet racing thoughts.")
    if "happy" in mood_lower or "neutral" in mood_lower:
        add("Stable evening mood helps sleep continuity; keep a consistent pre-sleep routine.")

    if (interruptions or "").lower().startswith("y"):
        add("Night awakenings are common; avoid clock-watching, keep the room cool and dark, and return to bed only when sleepy.")

    if bedtime is not None and wake is not None:
        in_bed = _hours_between_bed_and_wake(bedtime, wake)
        if abs(in_bed - sleep_duration_h) > 2:
            add("Your time in bed and reported sleep duration differ quite a bit—double-check entries or note time awake in bed.")

    if predicted_quality == "Poor":
        add("Focus on consistency: fixed wake time, morning light, and a simple evening routine often help first.")
    elif predicted_quality == "Average":
        add("You are close to a stronger pattern; small tweaks to light, caffeine, and wind-down usually move the needle.")
    elif predicted_quality == "Good":
        add("Keep protecting your sleep schedule and recovery habits—they are working in your favor.")

    if not tips:
        add("Maintain regular sleep and wake times, limit late caffeine, and keep your bedroom cool and dark.")

    return tips


def input_dict_for_history(
    sleep_duration_h: float,
    stress: int,
    exercise_min: int,
    screen_min: int,
    caffeine: str,
    mood: str,
    interruptions: str,
    bedtime: Any,
    wake: Any,
) -> dict[str, Any]:
    return {
        "sleep_duration_h": sleep_duration_h,
        "stress": stress,
        "exercise_min": exercise_min,
        "screen_min": screen_min,
        "caffeine": caffeine,
        "mood": mood,
        "interruptions": interruptions,
        "bedtime": str(bedtime) if bedtime is not None else "",
        "wake": str(wake) if wake is not None else "",
    }
