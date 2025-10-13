#!/usr/bin/env bash
# Usage: ./process_zip.sh path/to/archive.zip
# Extracts to a temp dir, runs ipynb_2_py.py on each .ipynb file,
# keeps converted files, and deletes only the original notebooks.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <zip-file>"
  exit 1
fi

ZIPFILE="$1"

if [[ ! -f "$ZIPFILE" ]]; then
  echo "Error: '$ZIPFILE' not found."
  exit 1
fi

command -v unzip >/dev/null 2>&1 || { echo "Error: unzip is required."; exit 1; }
command -v python >/dev/null 2>&1 || { echo "Error: python is required."; exit 1; }

# Create a temporary extraction directory
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# Extract zip contents
unzip -q "$ZIPFILE" -d "$TMPDIR"

# Process each .ipynb file
find "$TMPDIR" -type f -name '*.ipynb' -print0 | while IFS= read -r -d '' FILE; do
  echo "Converting: $FILE"
  python ipynb_2_py.py "$FILE"
  rm -f "$FILE"
done

# Move the converted files (if any) to current directory
# You can adjust this line if you want them in another place
find "$TMPDIR" -type f -not -name '*.ipynb' -exec mv {} . \;

echo "Done. Converted files kept, original notebooks deleted."

