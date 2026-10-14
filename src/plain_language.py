"""Everyday-language readings for VoiceSense screening results.

The model output stays numerical. This module only translates a finished
result into sentences a non-specialist can follow. It never changes a score.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional


def _family(feature: str) -> Optional[str]:
    """Group acoustic names into ideas a listener can picture.

    Average pitch is separated on purpose. It varies with sex and age and is
    a weak Parkinson's clue on its own (Little et al. dysphonia work; Rusz
    speech-recording guidance).
    """
    name = feature.lower().replace("praat_", "")
    if any(token in name for token in ("f0_std", "f0_range", "f0_cv", "ppe", "spread")):
        return "flatness"
    if name.startswith("f0_") or "pitch" in name or "fo(hz)" in name or "fhi" in name or "flo" in name:
        return "average_pitch"
    if any(token in name for token in ("jitter", "ppq", "ddp")) or name.startswith("rap"):
        return "pitch_wobble"
    if any(token in name for token in ("shimmer", "apq", "dda")):
        return "loudness"
    if any(token in name for token in ("hnr", "nhr", "harmonic")):
        return "breath"
    if "intensity" in name:
        return "quiet"
    if "voiced" in name:
        return "voiced"
    return None


_FAMILY_PD = {
    "pitch_wobble": "The pitch wobbles from moment to moment, which can sound rough or unstable.",
    "loudness": "The loudness is uneven from moment to moment, which can sound shaky.",
    "breath": "The tone is breathier or noisier, with less of a clear hum.",
    "flatness": "The pitch stays unusually flat, with little natural rise and fall.",
    "quiet": "The voice is quieter than most of the voices used to build this check.",
    "voiced": "Only part of the clip is a clear, steady voiced sound.",
    "average_pitch": (
        "Average pitch showed up in the math, but average pitch by itself is a weak clue. "
        "Women, men, and different ages naturally sit at different pitches."
    ),
}

_FAMILY_HC = {
    "pitch_wobble": "The pitch stays fairly steady from moment to moment.",
    "loudness": "The loudness stays fairly even.",
    "breath": "The tone is relatively clear, with less breathy noise.",
    "flatness": "The pitch still has some natural movement.",
    "quiet": "The loudness sits in a range this check treats as ordinary.",
    "voiced": "Most of the clip is a clear voiced sound.",
    "average_pitch": (
        "Average pitch looks ordinary for this check, and it is a weak clue either way."
    ),
}


def _chance_sentence(probability: float) -> str:
    pct = f"{probability:.0%}"
    if probability >= 0.72:
        where = "on the Parkinson's side of this app's scale"
    elif probability <= 0.42:
        where = "on the 'no clear pattern' side of this app's scale"
    else:
        where = "in the middle of this app's scale"
    return (
        f"The main check places this recording at about {pct}, {where}. "
        "A Parkinson's-like flag is only shown at 72% or higher, and a 'no clear pattern' "
        "reading only at 42% or lower, and only when the recording is clean and the second check agrees."
    )


def _heard_lines(explanation: Dict[str, Any]) -> List[str]:
    """One plain sentence per voice idea, skipping ideas that point both ways."""
    sides: Dict[str, str] = {}
    conflicted = set()
    groups = (
        ("pd", explanation.get("why_pd") or []),
        ("hc", explanation.get("why_not_pd") or []),
    )
    for side, items in groups:
        for item in items:
            family = _family(str(item.get("feature") or ""))
            if family is None:
                continue
            previous = sides.get(family)
            if previous is not None and previous != side:
                conflicted.add(family)
                continue
            sides.setdefault(family, side)

    lines: List[str] = []
    # Lead with the clues that actually describe voice quality. Average pitch last.
    order = ["pitch_wobble", "loudness", "breath", "flatness", "quiet", "voiced", "average_pitch"]
    for family in order:
        if family in conflicted or family not in sides:
            continue
        text = _FAMILY_PD[family] if sides[family] == "pd" else _FAMILY_HC[family]
        lines.append(text)
        if len(lines) == 3:
            break
    return lines


def combine_takes(takes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Summarize up to three separate vowels.

    Clinics and the public PD corpora typically keep three sustained /a/ vowels.
    One home-microphone clip is too easy to sway with noise, a short breath, or
    a cold. A Parkinson's-like set reading requires two clean takes on that
    side and no clean take on the other side.
    """
    takes = [item for item in takes if item.get("probability_pd") is not None]
    if not takes:
        return None
    probs = [float(item["probability_pd"]) for item in takes]
    median = float(statistics.median(probs))
    ready = [item for item in takes if item.get("recording_ready")]
    ready_codes = [item.get("decision_code") for item in ready]
    pd_votes = ready_codes.count("PD")
    hc_votes = ready_codes.count("HC")
    n = len(takes)

    if n < 2:
        return {
            "n": n,
            "conclusive": False,
            "decision_code": takes[-1].get("decision_code"),
            "median_probability_pd": median,
            "headline": "You have 1 of 3 recordings.",
            "detail": (
                "Record the same steady vowel twice more, with a short pause between them. "
                "One clip can be thrown off by the room, the microphone, or a single breath. "
                "This app does not treat one recording as the final word."
            ),
        }

    if len(ready) >= 2 and pd_votes >= 2 and hc_votes == 0 and median >= 0.72:
        code = "PD"
        headline = "Across these recordings, a Parkinson's-like pattern showed up more than once."
        conclusive = True
    elif len(ready) >= 2 and hc_votes == len(ready) and median <= 0.42:
        code = "HC"
        headline = "Across these recordings, no clear Parkinson's-like pattern showed up."
        conclusive = True
    else:
        code = "UNCERTAIN"
        headline = "These recordings do not agree enough for a firm reading."
        conclusive = False

    return {
        "n": n,
        "conclusive": conclusive,
        "decision_code": code,
        "median_probability_pd": median,
        "headline": headline,
        "detail": (
            f"The middle score across {n} recordings is about {median:.0%}. "
            "72% is the line for a Parkinson's-like flag and 42% is the line for no clear pattern. "
            "Only clean recordings cast a vote."
        ),
    }


def _trust(primary: Dict[str, Any], takes: List[Dict[str, Any]]) -> Dict[str, str]:
    limits = primary.get("recording_limits") or []
    agreement = primary.get("agreement") or {}
    agree = bool(agreement.get("agree"))
    single = bool(agreement.get("single_model"))
    series = combine_takes(takes)

    if limits:
        return {
            "label": "Weak recording",
            "detail": (
                "Pitch wobble, loudness wobble, and breathiness are unreliable when the clip "
                "is short, noisy, or partly unvoiced. "
                + " ".join(sentence[0].upper() + sentence[1:] + "." for sentence in limits)
            ),
        }
    if not single and not agree:
        return {
            "label": "The two checks disagree",
            "detail": (
                "The same recording was scored by two different models. "
                "They did not land on the same side, so neither one is being treated as final. "
                "Disagreement is common on laptop and phone microphones."
            ),
        }
    if series and series.get("conclusive"):
        return {
            "label": "Fair, for a home recording",
            "detail": (
                "Several clean recordings point the same way. That is more trustworthy than one clip, "
                "and it is still a screening hint from a home microphone, not a medical test."
            ),
        }
    return {
        "label": "Limited",
        "detail": (
            "This recording is usable, but it is still one take. "
            "The same person can get a different score on the next vowel, especially on a built-in microphone. "
            "Three steady vowels are the usual way to see whether the pattern holds."
        ),
    }


def build_plain_reading(
    result: Dict[str, Any],
    takes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Plain reading for one live-voice result, plus the set of recent takes."""
    primary = dict(result.get("primary") or {})
    code = primary.get("decision_code") or "UNCERTAIN"
    probability = float(primary.get("probability_pd") or 0.0)
    model_call = primary.get("model_call") or {}
    raw_code = model_call.get("decision_code") or code
    explanation = primary.get("explanation") or {}
    take_list = list(takes or [])

    if code == "PD":
        headline = "This recording leans toward a Parkinson's-like voice pattern."
        meaning = (
            "That is a screening hint, not a diagnosis. A cold, fatigue, a dry throat, "
            "or a noisy room can look similar. Only a clinician can tell whether someone has Parkinson's."
        )
    elif code == "HC":
        headline = "This recording does not show a clear Parkinson's-like voice pattern."
        meaning = (
            "That does not prove the speaker is free of Parkinson's. "
            "Early Parkinson's often leaves the voice sounding ordinary, and a home microphone misses a lot."
        )
    else:
        headline = "This recording is not clear enough to call either way."
        if raw_code == "PD" and raw_code != code:
            meaning = (
                "One check leaned toward a Parkinson's-like pattern, but that is not being shown as the result. "
                "The recording was not steady enough, or the second check disagreed. "
                "Holding the result back avoids a false alarm."
            )
        elif raw_code == "HC" and raw_code != code:
            meaning = (
                "One check leaned toward 'no clear pattern', but that is not being shown as the result. "
                "The clip was not clean enough, or the second check disagreed, so a reassuring answer would be too strong."
            )
        else:
            meaning = (
                "The score sits between a Parkinson's-like flag and a clearly ordinary voice, "
                "or the recording itself is too messy to trust. Refusing to guess is intentional."
            )

    heard = _heard_lines(explanation)
    if not heard:
        heard = ["No single voice clue stood out strongly enough to describe in plain words."]

    trust = _trust(primary, take_list)
    series = combine_takes(take_list)
    next_steps = [
        "Sit in a quiet room, about a hand's length from the microphone.",
        "Hold a steady “aaa” (or “আ—”) for 5 to 8 seconds. Do that three times, with a pause between them.",
        "If speaking, walking, or handwriting has changed in daily life, talk to a clinician. Do not use this screen as the reason to start or stop treatment.",
    ]
    return {
        "headline": headline,
        "meaning": meaning,
        "chance": _chance_sentence(probability),
        "heard": heard,
        "trust_label": trust["label"],
        "trust_detail": trust["detail"],
        "series": series,
        "next_steps": next_steps,
    }


def build_feature_plain(result: Dict[str, Any]) -> Dict[str, Any]:
    """Plain reading for a feature-table score, which never heard a microphone."""
    code = "PD" if result.get("pred_label") == 1 else "HC"
    probability = float(result.get("probability_pd") or 0.0)
    if code == "PD":
        headline = "These numbers sit closer to the Parkinson's examples in the lab dataset."
        meaning = (
            "This page did not listen to a voice. It compared a table of acoustic numbers with an older research dataset. "
            "A match is not a diagnosis."
        )
    else:
        headline = "These numbers sit closer to the healthy examples in the lab dataset."
        meaning = (
            "This page did not listen to a voice. It compared a table of acoustic numbers with an older research dataset. "
            "A non-match does not rule Parkinson's out."
        )
    heard = _heard_lines(result.get("explanation") or {})
    if not heard:
        heard = ["No single measure stood out strongly enough to describe in plain words."]
    return {
        "headline": headline,
        "meaning": meaning,
        "chance": (
            f"About {probability:.0%} of this model's weight falls on the Parkinson's side of its own scale. "
            "This lab model uses a yes-or-no cutoff near 50%. The live microphone screen is more cautious and leaves a wide middle band."
        ),
        "heard": heard,
        "trust_label": "Lab numbers only",
        "trust_detail": (
            "Useful for checking the model against published feature tables. "
            "It is not a reading of someone's microphone recording."
        ),
        "series": None,
        "next_steps": [
            "Use Live voice studio if you want a reading of an actual recording.",
            "Do not treat a spreadsheet match as a medical result.",
        ],
    }


def plain_quality_rejection(result: Dict[str, Any]) -> str:
    return (
        "This recording was not scored. It was too short, too quiet, or too noisy "
        "for the voice measures to mean anything. Record a steady vowel again for 5 to 8 seconds in a quiet room."
    )
