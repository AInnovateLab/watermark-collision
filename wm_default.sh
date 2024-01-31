export CUDA_VISIBLE_DEVICES=1

python create_wm_text.py \
    --generator-file configs/RDW/wsql256.yaml \
    --detector-file configs/RDW/wsql256.yaml \
    --output-dir results/test
