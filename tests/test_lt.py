#!/usr/bin/env python3
"""
Quick diagnostic: why does LanguageTool miss 'die Beschuldigte Person'?
Run on Mac: python3 test_lt.py
"""
import language_tool_python

TEST_SENTENCES = [
    "Die Beschuldigte Person fuhr zu schnell.",
    "die Beschuldigte Person war nicht anwesend.",
    "Der Beschuldigte Fahrer hat gebremst.",
    "Das Beschuldigte Fahrzeug war rot.",
    "Die beschuldigte Person fuhr zu schnell.",   # correct version
    "Die Zeugin sah die Beschuldigte Person.",
    "Einvernahmeprotokoll vom Thomas Müller.",
    "- Haben Sie weitere sachdienliche Hinweise? -",
    "Höhe Beginn zweites Parkfeld",
]

print("Loading LanguageTool de-CH...")
tool = language_tool_python.LanguageTool("de-CH")
print(f"Version: {tool.language}\n")

for sentence in TEST_SENTENCES:
    matches = tool.check(sentence)
    print(f"INPUT : {sentence!r}")
    if not matches:
        print("  → No errors detected")
    for m in matches:
        # Try all known attribute names
        rule_id = ""
        for attr in ("ruleId", "rule_id", "matchedByRule", "ruleIssueType"):
            val = getattr(m, attr, None)
            if val:
                rule_id = str(val)
                break
        offset = getattr(m, "offset", getattr(m, "offsetInContext", 0))
        length = getattr(m, "errorLength", getattr(m, "length", 0))
        message = getattr(m, "message", getattr(m, "msg", ""))
        replacements = getattr(m, "replacements", [])
        error_text = sentence[offset:offset+length]
        print(f"  → [{rule_id}] {error_text!r} — {message}")
        print(f"     Suggestions: {replacements[:3]}")
    print()

tool.close()
print("Done.")