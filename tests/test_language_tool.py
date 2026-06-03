import language_tool_python

tool = language_tool_python.LanguageTool("de-CH")
text = " 	Wie werten Sie den Fundort der mit SALEH Ali übereinstixfmmenden DNA-haltigen Spuren, gefunden am Fussgelenk und an den Hosenbeinenden des verstorbenen COVELLO Leonardo sowie den Fundort und die Menge der Tanne nnadeln, gefunden am Rücken des verstorbenen COVELLO Leonardo, bezüglich der beiden vorgenannten Szenarien?"

for m in tool.check(text):
    print("Rule:", m.rule_id)
    print("Message:", m.message)
    print("Suggestions:", m.replacements)
    print("Error:", text[m.offset:m.offset+m.error_length])
    print("---")