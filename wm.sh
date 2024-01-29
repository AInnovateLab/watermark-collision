export CUDA_VISIBLE_DEVICES=1
DELTA=1

python create_wm_text.py \
    --generator-file configs/KGW/g0.25_d5_selfhash_t4.yaml \
    --detector-file configs/KGW/g0.25_d5_selfhash_t4.yaml \
    --output-file wm_text_delta_$DELTA.jsonl
