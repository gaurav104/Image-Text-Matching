BASE_ROOT=../CUHK-PEDES

IMAGE_ROOT=../CUHK-PEDES/imgs
JSON_ROOT=../CUHK-PEDES/reid_raw.json
OUT_ROOT=../CUHK-PEDES/processed_data


echo "Process CUHK-PEDES dataset and save it as pickle form"

python ./data_preprocess/preprocess.py \
        --img_root=${IMAGE_ROOT} \
        --json_root=${JSON_ROOT} \
        --out_root=${OUT_ROOT} \
        --min_word_count 3 \
        --first
