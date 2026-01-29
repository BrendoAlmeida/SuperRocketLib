"""Junta dois CSVs com os mesmos campos.

Uso:
	python scripts/juntarcsv.py --csv1 data.csv --csv2 data2.csv --out data_merged.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
	with path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		if reader.fieldnames is None:
			raise ValueError(f"Arquivo sem cabeçalho: {path}")
		rows = list(reader)
		return list(reader.fieldnames), rows


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def juntar_csv(csv1: Path, csv2: Path, out: Path) -> None:
	fields1, rows1 = _read_rows(csv1)
	fields2, rows2 = _read_rows(csv2)

	if fields1 != fields2:
		raise ValueError(
			"Os CSVs não possuem os mesmos campos. "
			f"Campos 1: {fields1} | Campos 2: {fields2}"
		)

	_write_rows(out, fields1, rows1 + rows2)


def main() -> int:
	parser = argparse.ArgumentParser(description="Junta dois CSVs com os mesmos campos")
	parser.add_argument("--csv1", required=True, type=Path, help="Caminho do primeiro CSV")
	parser.add_argument("--csv2", required=True, type=Path, help="Caminho do segundo CSV")
	parser.add_argument("--out", required=True, type=Path, help="Caminho do CSV de saída")
	args = parser.parse_args()

	juntar_csv(args.csv1, args.csv2, args.out)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
