from __future__ import annotations


def generate_alert(
    original_text: str,
    caption: str | None,
    entities: list[dict[str, str]],
    severity: int,
    sa_label: str,
) -> str:
    """Combine model outputs into a concise human-readable disaster alert."""
    location = next(
        (
            entity["text"]
            for entity in entities
            if entity["label"] in {"LOC", "GPE", "LOCATION"}
        ),
        None,
    )
    disaster = next(
        (
            entity["text"]
            for entity in entities
            if entity["label"] == "DISASTER"
        ),
        None,
    )

    if severity >= 2:
        decision = "Severe damage detected"
    elif severity == 1:
        decision = "Possible moderate damage detected"
    else:
        decision = "Little or no damage detected"

    context: list[str] = []
    if location:
        context.append(f"location: {location}")
    if disaster:
        context.append(f"event: {disaster}")
    context.append(f"damage class: {sa_label}")

    evidence = [f'tweet: "{original_text}"']
    if caption:
        evidence.append(f'image caption: "{caption}"')

    return (
        f"{decision}. "
        f"{'; '.join(context)}. "
        f"Evidence — {'; '.join(evidence)}."
    )
