#!/bin/bash
# pipeline/02_inject_validate.sh
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Fehler ins Dokument schreiben + Ground Truth generieren
# Läuft für Case 01–10, überspringt Case 06 (manuelle Ground Truth).
#
# Voraussetzung:
#   - pipeline/01_generate.sh wurde ausgeführt
#   - Proposals wurden manuell geprüft (status="accepted"/"rejected")
#
# Aufruf: bash pipeline/02_inject_validate.sh
#
# Output pro Case:
#   → case_documents/case_XX_modified.docx
#   → ground_truth/ground_truth_case_XX_synthetic.jsonl
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
INJECTOR="$ROOT_DIR/synthetic_error_injector.py"
CASES_DIR="$ROOT_DIR/case_documents"
INJECTION_DIR="$ROOT_DIR/injection"

# Cases 01–10 ohne 06
CASES=(01 02 03 04 05 07 08)

echo "========================================"
echo "  Phase 2: inject + validate"
echo "  Cases: ${CASES[*]}"
echo "  Root:  $ROOT_DIR"
echo "========================================"
echo ""

cd "$ROOT_DIR"

SKIPPED=()
DONE=()

for CASE in "${CASES[@]}"; do
    DOC="$CASES_DIR/case_${CASE}.docx"
    PROPOSALS="$INJECTION_DIR/proposals_case_${CASE}_latest.json"
    MODIFIED="$CASES_DIR/case_${CASE}_modified.docx"

    # Dokument prüfen
    if [ ! -f "$DOC" ]; then
        echo "[SKIP] case_${CASE}: $DOC nicht gefunden"
        SKIPPED+=("$CASE")
        continue
    fi

    # Proposals prüfen
    if [ ! -f "$PROPOSALS" ]; then
        echo "[SKIP] case_${CASE}: $PROPOSALS nicht gefunden – erst 01_generate.sh ausführen"
        SKIPPED+=("$CASE")
        continue
    fi

    # Prüfen ob mindestens ein accepted Proposal vorhanden
    ACCEPTED=$(python3 -c "
import json, sys
data = json.load(open('$PROPOSALS'))
print(sum(1 for p in data if p.get('status') == 'accepted'))
" 2>/dev/null || echo "0")

    if [ "$ACCEPTED" -eq 0 ]; then
        echo "[SKIP] case_${CASE}: keine accepted Proposals in $PROPOSALS"
        echo "       Bitte Proposals prüfen und status='accepted' setzen."
        SKIPPED+=("$CASE")
        continue
    fi

    echo "────────────────────────────────────────"
    echo "  Case ${CASE}: inject  ($ACCEPTED accepted)"
    echo "────────────────────────────────────────"
    python "$INJECTOR" --mode inject \
        --document "$DOC" \
        --proposals "$PROPOSALS"

    echo ""
    echo "────────────────────────────────────────"
    echo "  Case ${CASE}: validate"
    echo "────────────────────────────────────────"
    python "$INJECTOR" --mode validate \
        --proposals "$PROPOSALS" \
        --modified_doc "$MODIFIED"

    echo ""
    echo "[OK] Case ${CASE} abgeschlossen."
    echo ""
    DONE+=("$CASE")
done

echo "========================================"
echo "  Phase 2 abgeschlossen."
echo ""
if [ ${#DONE[@]} -gt 0 ]; then
    echo "  Verarbeitet: ${DONE[*]}"
fi
if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "  Übersprungen: ${SKIPPED[*]}"
fi
echo ""
echo "  Output:"
echo "    case_documents/case_XX_modified.docx"
echo "    ground_truth/ground_truth_case_XX_synthetic.jsonl"
echo "========================================"
