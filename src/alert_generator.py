def generate_alert(original_text, caption, entities, severity, sa_label):
    location = None
    disaster = None

    for ent in entities:
        if ent["label"] in ["LOC", "GPE", "LOCATION"]:
            location = ent["text"]
        elif ent["label"] == "DISASTER" and disaster is None:
            disaster = ent["text"]

    # Texto del NER
    if entities:
        ner_text = ", ".join([f'{ent["text"]} ({ent["label"]})' for ent in entities])
    else:
        ner_text = "no se han detectado entidades"

    # Texto de la caption
    if caption:
        caption_text = f'La imagen ha generado la caption "{caption}"'
    else:
        caption_text = "No se ha proporcionado imagen, por lo que no se ha generado caption"

    location_text = f" in {location}" if location else ""
    disaster_text = f" caused by {disaster}" if disaster else ""

    # Texto del SA + decisión final
    if severity == 2:
        sa_text = f"el análisis de sentimiento ha detectado una severidad alta ({sa_label})"
        final_decision = (
            f"por lo que la alerta final es: severe damage detected"
            f"{location_text}{disaster_text}"
        )
    elif severity == 1:
        sa_text = f"el análisis de sentimiento ha detectado una severidad media ({sa_label})"
        final_decision = (
            f"por lo que la alerta final es: mild damage reported"
            f"{location_text}{disaster_text}"
        )
    else:
        sa_text = f"el análisis de sentimiento ha detectado una severidad baja ({sa_label})"
        final_decision = (
            f"por lo que la alerta final es: little or no damage detected"
            f"{location_text}{disaster_text}"
        )

    alert = (
        f'A partir del texto "{original_text}", {caption_text}. '
        f'Sobre la información combinada se ha generado un NER que ha identificado {ner_text}, '
        f'y un SA en el que {sa_text}, {final_decision}.'
    )

    return alert
