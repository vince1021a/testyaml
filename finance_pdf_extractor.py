#!/usr/bin/env python3
"""Extraction robuste de données financières tabulaires depuis un PDF.

Le script:
- détecte des années (ex: 2023, 2024),
- retrouve les libellés (ex: "Production vendue"),
- aligne les valeurs par colonnes via coordonnées (x/y), même sans lignes de tableau,
- écrit un fichier texte structuré (TSV + section diagnostic).

Usage:
    python finance_pdf_extractor.py input.pdf output.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2}|21\d{2})\b")
# Supporte formats FR/EN: 1 234,56 | 1.234,56 | 1,234.56 | (1 234) | -1234
NUMBER_TOKEN_RE = re.compile(
    r"(?<!\w)(?:\(?\-?\d{1,3}(?:[\s.,]\d{3})*(?:[.,]\d+)?\)?|\(?\-?\d+(?:[.,]\d+)?\)?)(?!\w)"
)
LABEL_CLEAN_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class FinancialValue:
    page: int
    label: str
    year: str
    value_raw: str


@dataclass(frozen=True)
class Row:
    y: float
    words: tuple[dict[str, Any], ...]


def normalize_label(text: str) -> str:
    text = text.strip(" \t:;.-")
    text = LABEL_CLEAN_RE.sub(" ", text)
    return text


def canonicalize_number(raw: str) -> str:
    token = raw.strip()
    neg = token.startswith("(") and token.endswith(")")
    token = token.strip("()")
    token = token.replace("\u202f", " ").replace("\xa0", " ").replace(" ", "")

    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "")
            token = token.replace(",", ".")
        else:
            token = token.replace(",", "")
    elif token.count(",") == 1 and token.count(".") == 0:
        token = token.replace(",", ".")
    elif token.count(".") > 1 and token.count(",") == 0:
        token = token.replace(".", "")

    token = token.strip()
    if neg and not token.startswith("-"):
        token = f"-{token}"
    return token


def is_useful_label(text: str) -> bool:
    if not text:
        return False
    if YEAR_RE.fullmatch(text):
        return False
    if NUMBER_TOKEN_RE.fullmatch(text):
        return False
    return any(ch.isalpha() for ch in text)


def find_year_positions(cells: Sequence[str]) -> list[tuple[int, str]]:
    positions: list[tuple[int, str]] = []
    for idx, cell in enumerate(cells):
        for y in YEAR_RE.findall(cell):
            positions.append((idx, y))
    return positions


def parse_table_rows(page_number: int, rows: Sequence[Sequence[str]]) -> list[FinancialValue]:
    """Extraction depuis tableaux détectés par pdfplumber.

    Plus prudente: on ne mappe une ligne que si le libellé est clair et les colonnes années sont présentes.
    """
    extracted: list[FinancialValue] = []

    for row in rows:
        cells = [normalize_label(c or "") for c in row]
        if not any(cells):
            continue

        year_positions = find_year_positions(cells)
        if not year_positions:
            continue

        label = ""
        for c in cells:
            if is_useful_label(c):
                label = c
                break
        if not label:
            continue

        found_at_least_one_value = False
        for year_col, year in year_positions:
            value = ""
            for value_col in range(year_col + 1, len(cells)):
                number_match = NUMBER_TOKEN_RE.search(cells[value_col])
                if number_match:
                    value = canonicalize_number(number_match.group(0))
                    break
            if value:
                found_at_least_one_value = True
                extracted.append(FinancialValue(page=page_number, label=label, year=year, value_raw=value))

        if not found_at_least_one_value:
            continue

    return extracted


def group_words_into_rows(words: Sequence[dict[str, Any]], y_tolerance: float = 3.0) -> list[Row]:
    sorted_words = sorted(words, key=lambda w: (float(w["top"]), float(w["x0"])))
    rows: list[list[dict[str, Any]]] = []

    for w in sorted_words:
        w_top = float(w["top"])
        if not rows:
            rows.append([w])
            continue
        last_row = rows[-1]
        avg_top = sum(float(x["top"]) for x in last_row) / len(last_row)
        if abs(w_top - avg_top) <= y_tolerance:
            last_row.append(w)
        else:
            rows.append([w])

    out: list[Row] = []
    for r in rows:
        r_sorted = sorted(r, key=lambda w: float(w["x0"]))
        y = sum(float(x["top"]) for x in r_sorted) / len(r_sorted)
        out.append(Row(y=y, words=tuple(r_sorted)))
    return out


def extract_year_columns_from_row(row: Row) -> list[tuple[str, float]]:
    cols: list[tuple[str, float]] = []
    for w in row.words:
        txt = normalize_label(str(w.get("text", "")))
        m = YEAR_RE.fullmatch(txt)
        if not m:
            continue
        x_center = (float(w["x0"]) + float(w["x1"])) / 2.0
        cols.append((m.group(1), x_center))
    return cols


def parse_layout_with_alignment(page_number: int, page: Any) -> list[FinancialValue]:
    """Extraction orientée coordonnées pour tableaux sans bordures visibles.

    Stratégie:
    1) regrouper les mots par ligne visuelle (axe Y),
    2) identifier une ligne d'en-tête contenant >=2 années,
    3) pour chaque ligne suivante: libellé à gauche + valeurs proches des X des colonnes années,
       sans "décaler" une valeur vers un autre libellé.
    """
    try:
        words = page.extract_words(
            keep_blank_chars=False,
            use_text_flow=True,
            extra_attrs=["x0", "x1", "top", "bottom"],
        )
    except Exception:
        return []

    if not words:
        return []

    rows = group_words_into_rows(words, y_tolerance=3.0)
    extracted: list[FinancialValue] = []

    i = 0
    while i < len(rows):
        year_cols = extract_year_columns_from_row(rows[i])
        # Seuil 2 années mini pour minimiser les faux positifs.
        if len(year_cols) < 2:
            i += 1
            continue

        year_cols = sorted(year_cols, key=lambda x: x[1])
        first_year_x = year_cols[0][1]
        # Balaye le bloc sous l'en-tête jusqu'au prochain en-tête potentiel / grand saut vertical.
        j = i + 1
        prev_y = rows[i].y
        while j < len(rows):
            row = rows[j]
            if extract_year_columns_from_row(row):
                break
            if row.y - prev_y > 22.0:
                break

            prev_y = row.y
            label_tokens: list[str] = []
            numeric_words: list[tuple[float, str]] = []

            for w in row.words:
                text = normalize_label(str(w.get("text", "")))
                if not text:
                    continue
                x_center = (float(w["x0"]) + float(w["x1"])) / 2.0
                if NUMBER_TOKEN_RE.fullmatch(text):
                    numeric_words.append((x_center, text))
                elif x_center < first_year_x - 10:
                    label_tokens.append(text)

            label = normalize_label(" ".join(label_tokens))
            if not is_useful_label(label):
                j += 1
                continue

            # Associe seulement les nombres proches de la colonne année correspondante.
            mapped_any = False
            for year, year_x in year_cols:
                nearest = None
                nearest_dist = 9999.0
                for nx, ntext in numeric_words:
                    dist = abs(nx - year_x)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest = ntext

                # Seuil horizontal strict pour éviter les mauvais appariements.
                if nearest is not None and nearest_dist <= 45.0:
                    extracted.append(
                        FinancialValue(
                            page=page_number,
                            label=label,
                            year=year,
                            value_raw=canonicalize_number(nearest),
                        )
                    )
                    mapped_any = True

            # Si aucun nombre aligné, on ignore la ligne (libellé sans valeur).
            if not mapped_any:
                j += 1
                continue

            j += 1

        i = j

    return extracted


def parse_text_lines(page_number: int, text: str) -> list[FinancialValue]:
    """Fallback faible confiance.

    Conservé pour certains PDFs mal segmentés, avec garde-fous pour limiter les associations erronées.
    """
    extracted: list[FinancialValue] = []
    lines = [normalize_label(line) for line in text.splitlines() if line.strip()]

    for line in lines:
        years = YEAR_RE.findall(line)
        if len(years) < 2:
            continue

        numbers = NUMBER_TOKEN_RE.findall(line)
        if len(numbers) < len(years):
            continue

        parts = re.split(r"\b(?:19\d{2}|20\d{2}|21\d{2})\b", line, maxsplit=1)
        label = normalize_label(parts[0]) if parts else ""
        if not is_useful_label(label):
            continue

        values = [canonicalize_number(v) for v in numbers]
        mapped = list(zip(years, values[-len(years) :]))
        for year, value in mapped:
            extracted.append(FinancialValue(page=page_number, label=label, year=year, value_raw=value))

    return extracted


def deduplicate(values: Iterable[FinancialValue]) -> list[FinancialValue]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[FinancialValue] = []
    for item in values:
        key = (item.label.lower(), item.year, item.value_raw)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def extract_financial_data(pdf_path: Path) -> list[FinancialValue]:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Erreur: la dépendance 'pdfplumber' est nécessaire. "
            "Installez-la avec: pip install pdfplumber"
        ) from exc

    items: list[FinancialValue] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # 1) Extraction par coordonnées (prioritaire pour tableaux sans bordures)
            items.extend(parse_layout_with_alignment(i, page))

            # 2) Extraction structurée des tableaux détectés
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []

            for tbl in tables:
                if tbl:
                    items.extend(parse_table_rows(i, tbl))

            # 3) Fallback texte
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text:
                items.extend(parse_text_lines(i, text))

    return deduplicate(items)


def write_output(output_path: Path, items: Sequence[FinancialValue]) -> None:
    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    for item in items:
        grouped[item.label][item.year] = item.value_raw

    years_sorted = sorted({item.year for item in items})

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Données financières extraites\n")
        f.write(f"# Entrées: {len(items)}\n\n")

        if not items:
            f.write("Aucune donnée financière détectée.\n")
            return

        header = ["label", *years_sorted]
        f.write("\t".join(header) + "\n")
        for label in sorted(grouped):
            row = [label] + [grouped[label].get(y, "") for y in years_sorted]
            f.write("\t".join(row) + "\n")

        f.write("\n# Détails (page, label, année, valeur)\n")
        for item in sorted(items, key=lambda x: (x.page, x.label.lower(), x.year)):
            f.write(f"p.{item.page}\t{item.label}\t{item.year}\t{item.value_raw}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extrait de manière robuste des données financières tabulaires "
            "depuis un PDF et écrit un fichier texte."
        )
    )
    parser.add_argument("input_pdf", type=Path, help="Chemin du fichier PDF source")
    parser.add_argument("output_txt", type=Path, help="Chemin du fichier texte de sortie")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input_pdf.exists():
        print(f"Erreur: fichier introuvable: {args.input_pdf}", file=sys.stderr)
        return 2

    extracted = extract_financial_data(args.input_pdf)
    write_output(args.output_txt, extracted)

    print(f"Extraction terminée: {len(extracted)} valeurs détectées.")
    print(f"Sortie: {args.output_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
