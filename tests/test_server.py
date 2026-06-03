import language_tool_python

tool = language_tool_python.LanguageTool("de-CH")
print(type(tool))

text = "Es entstand ein erheblicher Schaden, das Gebäude brannte volständig nieder."

for m in tool.check(text):
    reps   = getattr(m, "replacements", [])
    rule   = getattr(m, "rule_id",     getattr(m, "ruleId",     "?"))
    length = getattr(m, "error_length", getattr(m, "errorLength", 0))
    stelle = text[m.offset:m.offset + length]
    print(f"Rule:        {rule}")
    print(f"Stelle:      '{stelle}'")
    print(f"Vorschläge:  {len(reps)}")
    for i, r in enumerate(reps[:5]):
        print(f"  [{i}] type={type(r).__name__!r}  str={str(r)!r}")
    print()


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import language_tool_python
from rag_answer_reference_facts_OHNEFIX import _get_language_tool

tool = _get_language_tool()
print(type(tool))

text = "Es entstand ein erheblicher Schaden, das Gebäude brannte volständig nieder."

for m in tool.check(text):
    reps   = getattr(m, "replacements", [])
    rule   = getattr(m, "rule_id",     getattr(m,  "ruleId",     "?"))
    length = getattr(m, "error_length", getattr(m, "errorLength", 0))
    stelle = text[m.offset:m.offset + length]
    print(f"Rule:        {rule}")
    print(f"Stelle:      '{stelle}'")
    print(f"Vorschläge:  {len(reps)}")
    for i, r in enumerate(reps[:5]):
        print(f"  [{i}] type={type(r).__name__!r}  str={str(r)!r}")
    print()
