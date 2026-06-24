#!/usr/bin/env bash
# Download + extract the HRSID dataset (JPG version) into ./HRSID_JPG/.
# Source: https://github.com/chaozhong2010/HRSID  (Google Drive mirror)
#
# Note: the Drive file is named .zip but is actually a RAR archive, so we
# extract with bsdtar/unrar/7z (whichever is available).
set -euo pipefail

FILE_ID="1BZTU8Gyg20wqHXtBPFzRazn_lEdvhsbE"
ARCHIVE="HRSID_JPG.rar"

cd "$(dirname "$0")/.."   # repo root

if [ -d HRSID_JPG ]; then
  echo "HRSID_JPG/ already exists — nothing to do."
  exit 0
fi

# 1) download
if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown not found. Install it:  pip install gdown" >&2
  exit 1
fi
echo "Downloading HRSID (~609 MB) ..."
gdown "$FILE_ID" -O "$ARCHIVE"

# 2) extract (it's RAR despite the source name)
echo "Extracting $ARCHIVE ..."
if command -v bsdtar >/dev/null 2>&1; then
  bsdtar -xf "$ARCHIVE"
elif command -v unrar >/dev/null 2>&1; then
  unrar x -o+ "$ARCHIVE"
elif command -v 7z >/dev/null 2>&1; then
  7z x "$ARCHIVE"
else
  echo "No RAR extractor found. Install one: apt-get install libarchive-tools (bsdtar), or unrar/p7zip." >&2
  exit 1
fi

echo "Done -> HRSID_JPG/ (JPEGImages/ + annotations/)"
echo "Next:  python convert_hrsid_to_yolo.py"
