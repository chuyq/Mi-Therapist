CUDA_VISIBLE_DEVICES=1 python infer.py \
    --config_name lama \
    --inputter_name lama \
    --add_nlg_eval \
    --seed 0 \
    --load_checkpoint /mnt/storage/chuyuqi/code/DATA/lama.lama/2024-03-28224807.3e-05.16.1gpu/epoch-1.bin \
    --fp16 false \
    --max_input_length 150 \
    --max_decoder_input_length 50 \
    --max_length 40 \
    --min_length 10 \
    --infer_batch_size 2 \
    --infer_input_file /mnt/storage/chuyuqi/code/data_lama/test_lama.txt \
    --temperature 0.7 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1 \
    --no_repeat_ngram_size 0
