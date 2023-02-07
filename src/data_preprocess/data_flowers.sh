BASE_ROOT=../Flowers

IMAGE_ROOT=../Datasets/Flowers/images
JSON_ROOT=../Datasets/Flowers/reid_raw.json
OUT_ROOT=../Datasets/Flowers/processed_data


echo "Process Oxford-Flowers dataset and save it as pickle form"

python ./data_preprocess/preprocess.py \
        --img_root=${IMAGE_ROOT} \
        --json_root=${JSON_ROOT} \
        --out_root=${OUT_ROOT} \
        --min_word_count 2 \
	--dataset='Flowers' \
        --first
