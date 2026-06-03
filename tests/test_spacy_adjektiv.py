import spacy

nlp = spacy.load("de_dep_news_trf")

sentences = [
    # Fehler — sollte erkannt werden
    "In der Freien Natur führen insbesondere Winde zu Veränderungen.",
    "Die Ergebnisse des Voranstehenden Abschnitts sind relevant.",
    # Korrekt — sollte NICHT erkannt werden
    "Er ging ins Freie und atmete tief durch.",
    "Im Freien ist die Luft besser.",
    "In der freien Natur führen Winde zu Veränderungen.",
]

for sent in sentences:
    doc = nlp(sent)
    findings = []
    for token in doc:
        prev = doc[token.i - 1] if token.i > 0 else None
        if (
                token.text[0].isupper()
                and prev
                and prev.pos_ == "DET"
                and token.dep_ == "nk"
                and token.head.pos_ == "NOUN"
        ):
            findings.append(f"  → '{token.text}' pos={token.pos_} dep={token.dep_} head='{token.head.text}'")

    status = "[GEFUNDEN]" if findings else "[NICHT ERKANNT]"
    print(f"{status} '{sent[:60]}'")
    for f in findings:
        print(f)