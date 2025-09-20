# =======================================================================


#!/bin/bash

# JADED - Complete Code Collection Script
# Ez a script összegyűjti az összes létrehozott fájl teljes kódját egy txt fájlba
# Kizárja a dependency, package, root rendszer fájlokat és git-et

OUTPUT_FILE="complete_project_code.txt"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Funkció a fájl típus ellenőrzésére
is_source_file() {
    local file="$1"
    local basename=$(basename "$file")
    local dirname=$(dirname "$file")

    # Kizárt könyvtárak - KIBŐVÍTVE a Replit agent könyvtárral és attached_assets
    if [[ "$dirname" =~ ^\./(\.git|node_modules|__pycache__|\.pytest_cache|build|dist|target|\.cargo|\.stack-work|elm-stuff|\.elixir_ls|_build|deps|\.mix|\.hex|coverage|\.nyc_output|tmp|temp|log|logs|\.log|\.logs|vendor|\.vendor|packages|\.packages|\.npm|\.yarn|\.pnpm|bin|obj|out|Debug|Release|x64|x86|AnyCPU|\.vs|\.vscode|\.idea|\.eclipse|\.gradle|\.m2|\.ivy2|\.sbt|\.mill|\_opem|\.dub|result|\.nix-defexpr|\.direnv|\.cabal-sandbox|cabal\.sandbox\.config|\.stack|\.local|\.cache|\.config\/nix|\.uv|uv|cache|pylibs|site-packages|pip-cache|npm-cache|yarn-cache|pnmp-cache|local/state/replit/agent|attached_assets)$ ]]; then
        return 1
    fi

    # Kizárt fájlnevek (system/dependency files)
    case "$basename" in
        # Package manager fájlok (kizárva)
        package-lock.json|yarn.lock|pnpm-lock.yaml|composer.lock|Pipfile.lock|poetry.lock|Cargo.lock|stack.yaml.lock|uv.lock|requirements.lock)
            return 1 ;;
        # Cache és temp fájlok
        .gitignore|.gitattributes|.gitmodules|.DS_Store|Thumbs.db|desktop.ini|.env.local|.env.production|.env.development|*.tmp|*.temp|*.cache|*.log|*.pid|*.sock|*.swp|*.swo|*~|.pytest_cache|__pycache__|.coverage|coverage.xml|.nyc_output)
            return 1 ;;
        # Binary és compiled fájlok
        *.exe|*.dll|*.so|*.dylib|*.a|*.lib|*.o|*.obj|*.class|*.jar|*.war|*.ear|*.pyc|*.pyo|*.beam|*.hi|*.chi|*.cmi|*.cmo|*.cmx|*.cmxa|*.cma|*.cmxs|*.wasm|*.bc)
            return 1 ;;
        # Media és asset fájlok (nagyok)
        *.jpg|*.jpeg|*.png|*.gif|*.bmp|*.ico|*.pdf|*.mp3|*.mp4|*.avi|*.mov|*.wmv|*.flv|*.wav|*.ogg|*.zip|*.tar|*.gz|*.bz2|*.xz|*.7z|*.rar)
            return 1 ;;
        # Replit agent specifikus fájlok kizárása
        .agent_state_*|.latest.json|repl_state.bin)
            return 1 ;;
        # Root system files (kizárva) - main.jl KIVÉTEL
        .replit|replit.nix|*.lock|uv.toml|Pipfile|pyproject.toml|Cargo.toml|stack.yaml|cabal.project|build.gradle|pom.xml|go.mod|go.sum)
            return 1 ;;
        # main.jl explicit beemelése
        main.jl)
            return 0 ;;
    esac

    # Minden más fájl elfogadva (beleértve a létrehozott kód fájlokat)
    return 0
}

# Túl nagy fájlok kizárása (> 130KB)
is_file_too_large() {
    local file="$1"
    local size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
    [ "$size" -gt 133120 ]  # 130KB = 133120 bytes
}

# Fő script
echo "JADED - Complete Project Code Collection" > "$OUTPUT_FILE"
echo "Generated: $TIMESTAMP" >> "$OUTPUT_FILE"
echo "=========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

total_files=0
processed_files=0
skipped_large=0
skipped_system=0
total_lines=0

echo "Collecting all source code files..."

# Find parancs az összes fájlra a current directory-ban és alkönyvtárakban
while IFS= read -r -d '' file; do
    ((total_files++))

    # Relatív útvonal a kimeneti fájlhoz
    relative_path=$(realpath --relative-to=. "$file" 2>/dev/null || echo "$file")

    # Ellenőrizzük, hogy forráskód fájl-e
    if is_source_file "$file"; then
        # Ellenőrizzük a fájl méretét
        if is_file_too_large "$file"; then
            echo "SKIPPED (too large): $relative_path"
            ((skipped_large++))
            continue
        fi

        # Hozzáadjuk a fájlt a kimenethez
        echo "" >> "$OUTPUT_FILE"
        echo "=================================================================================" >> "$OUTPUT_FILE"
        echo "FILE: $relative_path" >> "$OUTPUT_FILE"
        echo "=================================================================================" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"

        # Fájl tartalmának hozzáadása
        if cat "$file" >> "$OUTPUT_FILE" 2>/dev/null; then
            ((processed_files++))
            file_lines=$(wc -l < "$file" 2>/dev/null || echo 0)
            ((total_lines += file_lines))
            echo "ADDED: $relative_path ($file_lines lines)"
        else
            echo "ERROR reading: $relative_path"
            echo "# ERROR: Could not read file content" >> "$OUTPUT_FILE"
        fi

        # Aláírás minden fájl után
        echo "" >> "$OUTPUT_FILE"
        echo "JADED made by Kollár Sándor on an iPhone 11 with Replit" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    else
        echo "SKIPPED (system/dependency file): $relative_path"
        ((skipped_system++))
    fi
done < <(find . -type f \
  ! -path './.git/*' \
  ! -path './node_modules/*' \
  ! -path './__pycache__/*' \
  ! -path './.*cache*' \
  ! -path './build/*' \
  ! -path './dist/*' \
  ! -path './target/*' \
  ! -path './_build/*' \
  ! -path './deps/*' \
  ! -path './vendor/*' \
  ! -path './packages/*' \
  ! -path './tmp/*' \
  ! -path './temp/*' \
  ! -path './log/*' \
  ! -path './logs/*' \
  ! -path './coverage/*' \
  ! -path './bin/*' \
  ! -path './obj/*' \
  ! -path './out/*' \
  ! -path './.uv/*' \
  ! -path './uv/*' \
  ! -path './cache/*' \
  ! -path './pylibs/*' \
  ! -path './site-packages/*' \
  ! -path '**/pip-cache/*' \
  ! -path '**/npm-cache/*' \
  ! -path '**/yarn-cache/*' \
  ! -path '**/pnpm-cache/*' \
  ! -path './local/state/replit/agent/*' \
  ! -path './attached_assets/*' \
  -print0)

# Összefoglaló a fájl végén
echo "" >> "$OUTPUT_FILE"
echo "=================================================================================" >> "$OUTPUT_FILE"
echo "COLLECTION SUMMARY" >> "$OUTPUT_FILE"
echo "=================================================================================" >> "$OUTPUT_FILE"
echo "Total files found: $total_files" >> "$OUTPUT_FILE"
echo "Source files processed: $processed_files" >> "$OUTPUT_FILE"
echo "System/dependency files skipped: $skipped_system" >> "$OUTPUT_FILE"
echo "Large files skipped: $skipped_large" >> "$OUTPUT_FILE"
echo "Total lines of code: $total_lines" >> "$OUTPUT_FILE"
echo "Generated: $TIMESTAMP" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Konzolra is kiírjuk az összefoglalót
echo ""
echo "==============================================="
echo "COLLECTION COMPLETED"
echo "==============================================="
echo "Output file: $OUTPUT_FILE"
echo "Total files found: $total_files"
echo "Source files processed: $processed_files"
echo "System/dependency files skipped: $skipped_system"
echo "Large files skipped: $skipped_large"
echo "Total lines of code: $total_lines"
echo ""
echo "The complete project code has been collected in: $OUTPUT_FILE"
# =======================================================================


# =======================================================================
