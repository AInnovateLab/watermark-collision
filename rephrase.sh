export CUDA_VISIBLE_DEVICES=1
HASHKEY=2023
NEW_DELTA=2.0
FILE=wm_text_delta_1

python rephrase.py \
    --input-file $FILE.jsonl \
    --new-key $HASHKEY \
    --rephraser-file configs/KGW/g0.25_d2_selfhash_t4.yaml \
    --old-detector-file configs/KGW/g0.25_d5_selfhash_t4.yaml \
    --new-detector-file configs/KGW/g0.25_d2_selfhash_t4.yaml \
    --output-file "${FILE}_rephrase_$HASHKEY.jsonl" \
    --output-dir wm_text_rephrased_$NEW_DELTA \
    --use-wm