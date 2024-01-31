export CUDA_VISIBLE_DEVICES=1

python create_wm_text.py \
    --model-name-or-path TheBloke/Llama-2-7B-GPTQ \
    --generator-file configs/SIR/w0_g0.5_d1_z0_context.yaml \
    --detector-file configs/SIR/w0_g0.5_d1_z0_context.yaml \
    --output-dir results/test
