#!/usr/bin/env python3
# parse.py - Javított Teljes App Felépítő (Formázás Fix + Newline Handling)
# Kezeli a trim-et és newline-okat jobban, hogy Coq/Rust/Julia ne essen szét.

import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
SOURCE_FILE = SCRIPT_DIR / "complete_source_code 5.txt"

if not SOURCE_FILE.exists():
    print(f"HIBA: '{SOURCE_FILE}' nem található!")
    sys.exit(1)

print("TELJES APP FELEPITO JAVITOTT - Formazás Fix")
print("=" * 60)
print(f"Forras: {SOURCE_FILE} (Meret: {SOURCE_FILE.stat().st_size / (1024*1024):.1f} MB)")
print(f"Sorok: {sum(1 for _ in open(SOURCE_FILE))}")
print("")

# # FÁJL: count
fajl_count = 0
with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if re.match(r"^# FÁJL:\s*(.*)$", line.strip()):
            fajl_count += 1
print(f"# FÁJL: talalatok: {fajl_count}")
if fajl_count == 0:
    print("FIGYELEM: 0 talalat - TXT torott?")
    sys.exit(1)
print("")

# Parse (javítva: append line.rstrip() + '\n', hogy pontos newline legyen)
current_file = ""
current_lines = []
files_detected = 0
files_created = 0
lines_processed = 0

with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        lines_processed += 1
        line_stripped = line.strip()  # Csak strip a match-hez

        match = re.match(r"^# FÁJL:\s*(.*)$", line_stripped)
        if match:
            # Előző mentése (join '\n'-nel, de rstrip minden soron)
            if current_file and current_lines:
                content = '\n'.join(lines.rstrip() for lines in current_lines) + '\n'  # Pontos newline
                full_path = SCRIPT_DIR / current_file
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, 'w', encoding='utf-8') as outf:
                    outf.write(content)
                print(f"LETREHOZVA (fix): {current_file}")
                files_created += 1

            current_file = match.group(1)
            current_lines = []
            files_detected += 1
            if files_detected % 10 == 0 or files_detected <= 5:
                print(f"DETEKALVA ({files_detected}): {current_file}")

        elif current_file:
            current_lines.append(line.rstrip())  # rstrip a trailing space-ekre, de newline később

        if lines_processed % 1000 == 0:
            print(f"Feldolgozva: {lines_processed} sor... (Det: {files_detected}, Letre: {files_created})")

# Utolsó
if current_file and current_lines:
    content = '\n'.join(lines.rstrip() for lines in current_lines) + '\n'
    full_path = SCRIPT_DIR / current_file
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, 'w', encoding='utf-8') as outf:
        outf.write(content)
    print(f"LETREHOZVA (fix): {current_file}")
    files_created += 1

print("")
print("FELEPITES BEFEJEZVE (JAVITOTT)!")
print("=" * 60)
print(f"Sorok: {lines_processed}, Detektalt: {files_detected}, Letrehozott: {files_created}")
if files_created > 0:
    print("Ellenorzendo fajlok:")
    os.system("ls -la | head -20")
    print("\nCoq fajl ellenorzes: cat 02_coq_quantum.v | head -10")  # Teszt a te fájlodra
    os.system("cat 02_coq_quantum.v | head -10")  # Kiírja az első 10 sort
    print("\nFuttasd: coqtop -R . Coq 02_coq_quantum.v  (ha Coq telepítve van)")
else:
    print("HIBA: 0 fajl - TXT ellenorzes: grep '# FÁJL:' 'complete_source_code 5.txt' | head -3")