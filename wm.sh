export CUDA_VISIBLE_DEVICES=1
DELTA=1

python main.py \
    --delta $DELTA \
    --output_file wm_text_delta_$DELTA.jsonl