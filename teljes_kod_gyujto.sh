# =======================================================================


#!/bin/bash

# Teljes Kód Gyűjtő Script
# Minden létrehozott fájl teljes kódját összegyűjti egy txt fájlba
# Kizárja: dependencies, cache, temp, nix, agent, memories, packages, root rendszerfájlokat, git

OUTPUT_FILE="osszes_teljes_kod.txt"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Funkció a fájl típus ellenőrzésére - NEM forráskód fájlok kiszűrése
is_excluded_file() {
    local file="$1"
    local basename=$(basename "$file")
    local dirname=$(dirname "$file")

    # Cache és build könyvtárak teljes kizárása bárhol a fájlrendszerben
    if [[ "$file" =~ .*/\.elixir_ls/.* ]] || \
       [[ "$file" =~ .*/_build/.* ]] || \
       [[ "$file" =~ .*/deps/.* ]] || \
       [[ "$file" =~ .*/\.hex/.* ]] || \
       [[ "$file" =~ .*/\.mix/.* ]] || \
       [[ "$file" =~ .*/ebin/.* ]] || \
       [[ "$file" =~ .*/priv/.* ]] || \
       [[ "$file" =~ .*/consolidated/.* ]] || \
       [[ "$file" =~ .*/cache/.* ]] || \
       [[ "$file" =~ .*/\.cache/.* ]] || \
       [[ "$file" =~ .*/node_modules/.* ]] || \
       [[ "$file" =~ .*/__pycache__/.* ]] || \
       [[ "$file" =~ .*/\.pytest_cache/.* ]] || \
       [[ "$file" =~ .*/build/.* ]] || \
       [[ "$file" =~ .*/dist/.* ]] || \
       [[ "$file" =~ .*/target/.* ]] || \
       [[ "$file" =~ .*/\.cargo/.* ]] || \
       [[ "$file" =~ .*/vendor/.* ]] || \
       [[ "$file" =~ .*/packages/.* ]] || \
       [[ "$file" =~ .*/tmp/.* ]] || \
       [[ "$file" =~ .*/temp/.* ]] || \
       [[ "$file" =~ .*/log/.* ]] || \
       [[ "$file" =~ .*/logs/.* ]] || \
       [[ "$file" =~ .*/coverage/.* ]] || \
       [[ "$file" =~ .*/bin/.* ]] || \
       [[ "$file" =~ .*/obj/.* ]] || \
       [[ "$file" =~ .*/out/.* ]] || \
       [[ "$file" =~ .*/\.uv/.* ]] || \
       [[ "$file" =~ .*/uv/.* ]] || \
       [[ "$file" =~ .*/pylibs/.* ]] || \
       [[ "$file" =~ .*/site-packages/.* ]] || \
       [[ "$file" =~ .*/attached_assets/.* ]] || \
       [[ "$file" =~ .*/\.git/.* ]]; then
        return 0  # Kizárva
    fi

    # Könyvtár név alapú kizárások (root szinten)
    if [[ "$dirname" =~ ^\./(\.git|node_modules|__pycache__|\.pytest_cache|build|dist|target|\.cargo|\.stack-work|elm-stuff|\.elixir_ls|_build|deps|\.mix|\.hex|coverage|\.nyc_output|tmp|temp|log|logs|\.log|\.logs|vendor|\.vendor|packages|\.packages|\.npm|\.yarn|\.pnpm|bin|obj|out|Debug|Release|x64|x86|AnyCPU|\.vs|\.vscode|\.idea|\.eclipse|\.gradle|\.m2|\.ivy2|\.sbt|\.mill|\_opem|\.dub|result|\.nix-defexpr|\.direnv|\.cabal-sandbox|cabal\.sandbox\.config|\.stack|\.local|\.cache|\.config|\.uv|uv|cache|pylibs|site-packages|pip-cache|npm-cache|yarn-cache|pnpm-cache|local|attached_assets|consolidated|ebin|priv|spec|test|tests|\.rebar3|\.formatter\.exs|assets|plugins\.mk|include|src|lib)$ ]]; then
        return 0  # Kizárva
    fi

    # Root rendszerfájlok kizárása
    case "$basename" in
        .replit|replit.nix|*.lock|uv.toml|Pipfile|pyproject.toml|Cargo.toml|stack.yaml|cabal.project|build.gradle|pom.xml|go.mod|go.sum|mix.lock|yarn.lock|package-lock.json|composer.lock|Pipfile.lock|poetry.lock|Cargo.lock|stack.yaml.lock|uv.lock|requirements.lock)
            return 0 ;;  # Kizárva
        # Cache és temp fájlok
        .gitignore|.gitattributes|.gitmodules|.DS_Store|Thumbs.db|desktop.ini|.env.local|.env.production|.env.development|*.tmp|*.temp|*.cache|*.log|*.pid|*.sock|*.swp|*.swo|*~|.pytest_cache|__pycache__|.coverage|coverage.xml|.nyc_output)
            return 0 ;;  # Kizárva
        # Binary és compiled fájlok
        *.exe|*.dll|*.so|*.dylib|*.a|*.lib|*.o|*.obj|*.class|*.jar|*.war|*.ear|*.pyc|*.pyo|*.beam|*.hi|*.chi|*.cmi|*.cmo|*.cmx|*.cmxa|*.cma|*.cmxs|*.wasm|*.bc|*.app)
            return 0 ;;  # Kizárva
        # Media és asset fájlok
        *.jpg|*.jpeg|*.png|*.gif|*.bmp|*.ico|*.pdf|*.mp3|*.mp4|*.avi|*.mov|*.wmv|*.flv|*.wav|*.ogg|*.zip|*.tar|*.gz|*.bz2|*.xz|*.7z|*.rar)
            return 0 ;;  # Kizárva
        # Replit agent specifikus fájlok
        .agent_state_*|.latest.json|repl_state.bin)
            return 0 ;;  # Kizárva
        # Package manager config fájlok és Elixir specifikus
        hex_metadata.config|rebar.config|rebar.lock|rebar3.crashdump|mix.rebar.config|erlang.mk|plugins.mk|Makefile|.formatter.exs|.hex|VERSION|CHANGELOG.md|LICENSE|LICENSE.md|README.md|README.asciidoc|NOTICE|package.json|mix.exs|shard.yml)
            return 0 ;;  # Kizárva
        # Test és spec fájlok
        *_test.exs|*_spec.cr|test_helper.exs|spec_helper.cr|*_test.erl|*_SUITE.erl)
            return 0 ;;  # Kizárva
        # Beam és build fájlok
        *.beam|*.app|*.appup|*.o|*.so|*.dylib)
            return 0 ;;  # Kizárva
        # Egyéb cache/generated fájlok
        *.ets|.agent_*|cache.ets|osszes_teljes_kod.txt|complete_project_code.txt|*.tar|*.crashdump|iplt-*|*.map)
            return 0 ;;  # Kizárva
        # Configuration és meta fájlok
        .gitignore|.gitattributes|rebar3.crashdump|.formatter.exs)
            return 0 ;;  # Kizárva
    esac

    return 1  # NEM kizárva - ez forráskód fájl
}

# Túl nagy fájlok kizárása (> 200KB)
is_file_too_large() {
    local file="$1"
    local size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
    [ "$size" -gt 204800 ]  # 200KB = 204800 bytes
}

# Fő script
echo "TELJES KÓD GYŰJTŐ - Minden létrehozott fájl teljes kódja" > "$OUTPUT_FILE"
echo "Generálva: $TIMESTAMP" >> "$OUTPUT_FILE"
echo "=============================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

total_files=0
processed_files=0
skipped_large=0
skipped_excluded=0
total_lines=0

echo "Összes forráskód fájl összegyűjtése..."

# Find parancs az összes fájlra
while IFS= read -r -d '' file; do
    ((total_files++))

    # Relatív útvonal
    relative_path=$(realpath --relative-to=. "$file" 2>/dev/null || echo "$file")

    # Ellenőrizzük, hogy kizárt fájl-e
    if is_excluded_file "$file"; then
        echo "KIHAGYVA (rendszerfájl): $relative_path"
        ((skipped_excluded++))
        continue
    fi

    # Ellenőrizzük a fájl méretét
    if is_file_too_large "$file"; then
        echo "KIHAGYVA (túl nagy): $relative_path"
        ((skipped_large++))
        continue
    fi

    # Hozzáadjuk a fájlt a kimenethez - TELJES TARTALOM
    echo "" >> "$OUTPUT_FILE"
    echo "=============================================================================" >> "$OUTPUT_FILE"
    echo "FÁJL: $relative_path" >> "$OUTPUT_FILE"
    echo "=============================================================================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    # Fájl TELJES tartalmának hozzáadása - SEMMIT NEM HAGYUNK KI
    if cat "$file" >> "$OUTPUT_FILE" 2>/dev/null; then
        ((processed_files++))
        file_lines=$(wc -l < "$file" 2>/dev/null || echo 0)
        ((total_lines += file_lines))
        echo "HOZZÁADVA: $relative_path ($file_lines sor)"
    else
        echo "HIBA olvasáskor: $relative_path"
        echo "# HIBA: A fájl tartalma nem olvasható" >> "$OUTPUT_FILE"
    fi

    # Elválasztó minden fájl után
    echo "" >> "$OUTPUT_FILE"
    echo "--- Fájl vége ---" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

done < <(find . -type f -print0)

# Összefoglaló a fájl végén
echo "" >> "$OUTPUT_FILE"
echo "=============================================================================" >> "$OUTPUT_FILE"
echo "GYŰJTÉS ÖSSZEFOGLALÓJA" >> "$OUTPUT_FILE"
echo "=============================================================================" >> "$OUTPUT_FILE"
echo "Összes talált fájl: $total_files" >> "$OUTPUT_FILE"
echo "Feldolgozott forráskód fájlok: $processed_files" >> "$OUTPUT_FILE"
echo "Kihagyott rendszerfájlok: $skipped_excluded" >> "$OUTPUT_FILE"
echo "Kihagyott nagy fájlok: $skipped_large" >> "$OUTPUT_FILE"
echo "Összes kódsor: $total_lines" >> "$OUTPUT_FILE"
echo "Generálva: $TIMESTAMP" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "=== TELJES FORRÁSKÓD GYŰJTÉS BEFEJEZVE ===" >> "$OUTPUT_FILE"

# Konzolra is kiírjuk az összefoglalót
echo ""
echo "============================================="
echo "GYŰJTÉS BEFEJEZVE"
echo "============================================="
echo "Kimeneti fájl: $OUTPUT_FILE"
echo "Összes talált fájl: $total_files"
echo "Feldolgozott forráskód fájlok: $processed_files"
echo "Kihagyott rendszerfájlok: $skipped_excluded"
echo "Kihagyott nagy fájlok: $skipped_large"
echo "Összes kódsor: $total_lines"
echo ""
echo "Az összes forráskód teljes tartalma itt található: $OUTPUT_FILE"

# =======================================================================

