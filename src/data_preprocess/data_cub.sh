BASE_ROOT=../CUB

IMAGE_ROOT=../Datasets/CUB/images
JSON_ROOT=../Datasets/CUB/reid_raw.json
OUT_ROOT=../Datasets/CUB/processed_data


echo "Process CUB dataset and save it as pickle form"

python ./data_preprocess/preprocess.py \
        --img_root=${IMAGE_ROOT} \
        --json_root=${JSON_ROOT} \
        --out_root=${OUT_ROOT} \
        --min_word_count 2 \
	--dataset='CUB' \
        --first
