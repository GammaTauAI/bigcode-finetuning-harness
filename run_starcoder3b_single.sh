# need to give deepspeed config file as argument
if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Please give deepspeed config file as argument"
    exit 1
fi
python3 train.py \
        --model_path="bigcode/starcoderbase-3b" \
        --no_custom_tokenizer \
        --dataset_name="nuprl/ts-training" \
        --total_tokens=10063051440 \
        --fim_rate 1 \
        --fim_spm_rate 0.5 \
        --output_dir="./model_starcoder_3b_fit_8k" \
        --seq_length 8192 \
        --epochs 10 \
        --batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 \
        --num_warmup_steps 10 \
        --num_workers=$(expr $(nproc --all) - 4) \
        --no_fp16 \
        --eval_freq 0.1 \
        --save_freq 0.1 \
        --bf16 \
        --perc_valid_set 0.001 \
        --save_total_limit 20
