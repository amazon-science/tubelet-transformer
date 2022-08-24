IN_DATA_DIR="/opt/tiger/data2/ava/videos/val/addi"
OUT_DATA_DIR="/opt/tiger/data2/ava/videos_15min/addi"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  out_name="${OUT_DATA_DIR}/${video##*/}"
  if [ ! -f "${out_name}" ]; then
    ffmpeg -ss 900 -t 901 -i "${video}" "${out_name}"
  fi
done