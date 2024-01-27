export CUDA_VISIBLE_DEVICES=1
HASHKEY=2023
NEW_DELTA=2.0

python rephrase.py \
    --input_file wm_text_delta_2.jsonl \
    --hash_key $HASHKEY \
    --output_file wm_text_delta_2_rephrase_$HASHKEY.jsonl \
    --output_dir wm_text_rephrased_$NEW_DELTA \
    --use_wm