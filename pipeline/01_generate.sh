#!/bin/bash
# pipeline/01_generate.sh
# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Referenzfakten extrahieren + Fehlervorschläge generieren
# Läuft für Case 01–10, überspringt Case 06 (manuelle Ground Truth).
#
# Aufruf: bash pipeline/01_generate.sh
# Voraussetzung: .env konfiguriert, Spark via Ethernet erreichbar
#
# Nach diesem Script:
#   → injection/proposals_case_XX_latest.json pro Case
#   → injection/reference_facts_case_XX_original.json pro Case
#
# NÄCHSTER SCHRITT:
#   Proposals manuell prüfen, status="accepted"/"rejected" setzen,
#   dann pipeline/02_inject_validate.sh ausführen.
# ─────────────────────────────────────────────────────────────────────────────

set -e  # Abbruch bei Fehler

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
INJECTOR="$ROOT_DIR/synthetic_error_injector.py"
CASES_DIR="$ROOT_DIR/case_documents"

# Cases 01–10 ohne 06
CASES=(01 02 03 04 05 07 08)

echo "========================================"
echo "  Phase 1: extract_facts + generate"
echo "  Cases: ${CASES[*]}"
echo "  Root:  $ROOT_DIR"
echo "========================================"
echo ""

cd "$ROOT_DIR"

for CASE in "${CASES[@]}"; do
    DOC="$CASES_DIR/case_${CASE}.docx"

    if [ ! -f "$DOC" ]; then
        echo "[SKIP] case_${CASE}: $DOC nicht gefunden"
        continue
    fi

    echo "────────────────────────────────────────"
    echo "  Case ${CASE}: extract_facts"
    echo "────────────────────────────────────────"
    python "$INJECTOR" --mode extract_facts --document "$DOC"

    echo ""
    echo "────────────────────────────────────────"
    echo "  Case ${CASE}: generate"
    echo "────────────────────────────────────────"
    python "$INJECTOR" --mode generate --document "$DOC"

    echo ""
    echo "[OK] Case ${CASE} abgeschlossen."
    echo ""
done

echo "========================================"
echo "  Phase 1 abgeschlossen."
echo ""
echo "  NÄCHSTER SCHRITT:"
echo "  1. Proposals prüfen in injection/proposals_case_XX_latest.json"
echo "  2. status='accepted' oder 'rejected' setzen"
echo "  3. bash pipeline/02_inject_validate.sh"
echo "========================================"
