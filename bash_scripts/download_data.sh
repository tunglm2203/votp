#!/bin/bash
# Download and extract datasets from Google Drive.
#
# Usage:
#   bash bash_scripts/download_data.sh                  # required data only (~5G)
#   bash bash_scripts/download_data.sh --with-videos    # also fetch video archives (~10G extra)
#
# Each archive is downloaded to downloads/, extracted into its target folder
# under datasets/, and the downloaded zip is removed after successful
# extraction. Re-runs skip any category whose destination dir already exists.

set -e

# cd to the repo root (parent of bash_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

WITH_VIDEOS=0
for arg in "$@"; do
  case "$arg" in
    --with-videos) WITH_VIDEOS=1 ;;
    -h|--help)
      sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

if ! command -v gdown >/dev/null 2>&1; then
  echo "Error: 'gdown' is required. Install it with:  pip install gdown"
  exit 1
fi

GDRIVE_MW_SEGMENTS_DATA="https://drive.google.com/file/d/1NcA1-YYL3bYsRF1twg6uUmR48DF0sWkf/view?usp=sharing"
GDRIVE_MW_SEGMENTS_FEATURES="https://drive.google.com/file/d/1rzaJ6DZ42GW4ybOlSNY42aVPxk5Yan3c/view?usp=drive_link"
GDRIVE_MW_SEGMENTS_VIDEO="https://drive.google.com/file/d/1BZmqjq4ChISyLBcFQUI6mlTiYI_rIyyi/view?usp=drive_link"
GDRIVE_MW_OFFLINE="https://drive.google.com/file/d/1wBWjhVvAAMIND5wNNH8zxqk0lJWCp5LN/view?usp=drive_link"
GDRIVE_MW_PAIR_INDICES="https://drive.google.com/file/d/1FunYPA0S8XmIm2UVdLqfwZEv5-VZjE8v/view?usp=sharing"

GDRIVE_LOCO_SEGMENTS_DATA="https://drive.google.com/file/d/1SUTxt9eTfncA12Ck3vsPNPwH2BWr-yU7/view?usp=drive_link"
GDRIVE_LOCO_SEGMENTS_FEATURES="https://drive.google.com/file/d/18B8DagC5s4CspSW524fMIL0Kls4DhowM/view?usp=drive_link"
GDRIVE_LOCO_SEGMENTS_VIDEO="https://drive.google.com/file/d/1mcK67J6pDMqjeSmfVBJF5lU4QTYaj82m/view?usp=drive_link"
GDRIVE_LOCO_PAIR_INDICES="https://drive.google.com/file/d/1Mq77kAu5IDBOq0w4sAPbJPiX6YHP9ppu/view?usp=sharing"
# --------------------------------------------------------------------------

mkdir -p downloads

# download_and_extract NAME GDRIVE_URL DEST_DIR
#   - skip if DEST_DIR already exists
#   - download to downloads/NAME.zip
#   - extract into DEST_DIR's parent: each archive wraps its contents in a
#     folder matching DEST_DIR's leaf name (e.g. features/...), so it lands at
#     exactly DEST_DIR
#   - verify DEST_DIR now exists, then delete the zip
download_and_extract() {
  local name="$1"
  local url="$2"
  local dest="$3"

  # Already extracted only if the dir has real content (a bare dir or a lone
  # .gitkeep from the tracked skeleton does NOT count).
  if [ -d "$dest" ] && [ -n "$(find "$dest" -mindepth 1 -not -name '.gitkeep' -print -quit 2>/dev/null)" ]; then
    echo "[skip] $name already extracted ($dest)"
    return
  fi

  local zipfile="downloads/${name}.zip"
  if [ ! -f "$zipfile" ]; then
    echo "[download] $name -> $zipfile"
    gdown --fuzzy -O "$zipfile" "$url"
  else
    echo "[cache] reusing $zipfile"
  fi

  local target_dir
  target_dir="$(dirname "$dest")"
  mkdir -p "$target_dir"

  echo "[extract] $zipfile -> $target_dir/"
  unzip -q -o "$zipfile" -d "$target_dir"

  if [ ! -e "$dest" ]; then
    echo "[error] extraction did not produce '$dest'."
    echo "        The archive's top-level folder must be named '$(basename "$dest")'."
    echo "        Inspect with: unzip -l $zipfile   (zip kept for debugging)"
    exit 1
  fi

  echo "[cleanup] removing $zipfile"
  rm -f "$zipfile"
}

# Required (training + labeling)
download_and_extract "mw_segments_data"       "$GDRIVE_MW_SEGMENTS_DATA"       "datasets/metaworld/segments/data"
download_and_extract "mw_segments_features"   "$GDRIVE_MW_SEGMENTS_FEATURES"   "datasets/metaworld/segments/features"
download_and_extract "mw_offline_dataset"     "$GDRIVE_MW_OFFLINE"             "datasets/metaworld/offline_dataset"
download_and_extract "mw_pair_indices"        "$GDRIVE_MW_PAIR_INDICES"        "datasets/metaworld/pseudo_preferences/pair_indices"
download_and_extract "loco_segments_data"     "$GDRIVE_LOCO_SEGMENTS_DATA"     "datasets/locomotion/segments/data"
download_and_extract "loco_segments_features" "$GDRIVE_LOCO_SEGMENTS_FEATURES" "datasets/locomotion/segments/features"
download_and_extract "loco_pair_indices"      "$GDRIVE_LOCO_PAIR_INDICES"      "datasets/locomotion/pseudo_preferences/pair_indices"

# Optional: source videos (only needed to regenerate features from scratch)
if [ "$WITH_VIDEOS" = "1" ]; then
  download_and_extract "mw_segments_video"   "$GDRIVE_MW_SEGMENTS_VIDEO"   "datasets/metaworld/segments/video"
  download_and_extract "loco_segments_video" "$GDRIVE_LOCO_SEGMENTS_VIDEO" "datasets/locomotion/segments/video"
else
  echo
  echo "[info] Skipped video archives (~10G). Re-run with --with-videos to fetch them."
fi

echo
echo "Done. Datasets extracted under datasets/"
