export CUDA_VISIBLE_DEVICES=1
HASHKEY=2023
NEW_DELTA=2.0
FILE=wm_text_delta_1

python rephrase.py \
    --input_file $FILE.jsonl \
    --hash_key $HASHKEY \
    --output_file "${FILE}_rephrase_$HASHKEY.jsonl" \
    --new_delta $NEW_DELTA \
    --output_dir wm_text_rephrased_$NEW_DELTA \
    --use_wm