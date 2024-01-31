export CUDA_VISIBLE_DEVICES=1
HASHKEY=2023

python rephrase.py \
    --model-name-or-path TheBloke/Llama-2-7B-GPTQ \
    --input-file results/KGW-KGW/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl \
    --new-key $HASHKEY \
    --rephraser-file configs/KGW/g0.25_d2_selfhash_t4.yaml \
    --old-detector-file configs/KGW/g0.25_d5_selfhash_t4.yaml \
    --new-detector-file configs/KGW/g0.25_d2_selfhash_t4.yaml \
    --output-dir results/KGW-KGW \
    --use-wm
