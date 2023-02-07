BASE_ROOT=../Flickr30k

IMAGE_ROOT=../Datasets/Flickr30k/images
JSON_ROOT=../Datasets/Flickr30k/reid_raw.json
OUT_ROOT=../Datasets/Flickr30k/processed_data

echo "Process Flickr30k dataset and save it as pickle form"

python ./data_preprocess/preprocess.py \
        --img_root=${IMAGE_ROOT} \
        --json_root=${JSON_ROOT} \
        --out_root=${OUT_ROOT} \
        --min_word_count 2 \
	--dataset='Flickr30k' \
        --first
