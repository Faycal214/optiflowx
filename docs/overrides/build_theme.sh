#!/usr/bin/env bash
# Build the SCSS theme into CSS (requires `sass` from Dart Sass or `sassc`)
set -euo pipefail
SRC="docs/overrides/assets/stylesheets/theme.scss"
OUT="docs/overrides/assets/stylesheets/theme.css"

if ! command -v sass >/dev/null 2>&1; then
  echo "Error: `sass` not found. Install Dart Sass (https://sass-lang.com/install) or use your system package manager." >&2
  exit 2
fi

echo "Compiling $SRC -> $OUT"
sass --no-source-map --style=compressed "$SRC" "$OUT"
echo "Done."
