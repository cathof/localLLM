import language_tool_python

with language_tool_python.LanguageTool("de-CH") as tool:
    texts = [
        "Die Beschuldigte Person fuhr mit dem Motorkarren von Baustellendepotplatz weg.",
        "Die Aufnahme wurde am 15.02.2024 aufgenohmen.",
        "Es gibt keine sichtbaren Achsenbeschriftungen Legenden oder technischen Masse.",
        "Die Sicht nach vorne war für die beschuldigte Person nicht volständig dokumentiert.",
        "Der Bericht enthällt keine nachvollziebare Begründung.",
        "In der Schweiz ist ss korrekt und Strasse sollte nicht automatisch zu Straße geändert werden.",
    ]

    for i, text in enumerate(texts, start=1):
        print("=" * 80)
        print(f"TEXT {i}:")
        print(text)

        matches = tool.check(text)

        print("\nMATCHES:")
        for m in matches:
            print(m)

        print("\nKORREKTUR:")
        print(tool.correct(text))