#CUDA_VISIBLE_DEVICES=... python3 -m torch.distributed.launch \
python3 -m torch.distributed.launch \
        --nproc_per_node 8 train.py \
        --model_path="bigcode/starcoderbase" \
        --no_custom_tokenizer \
        --model_revision="main" \
        --dataset_name="cassanof/starcoderdata-lua-ranked" \
        --dataset_revision="top_10000" \
        --subset="data" \
        --data_column "content" \
        --split="train" \
        --output_dir="./model_starcoder_lora" \
        --seq_length 2048 \
        --max_steps 1400 \
        --batch_size 8 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1e-4 \
        --num_warmup_steps 100 \
        --eval_freq 100 \
        --save_freq 100 \
        --log_freq 1 \
        --num_workers=$(expr $(nproc --all) - 4) \
        --no_fp16 \
        --bf16 \
        --lora \
        --lora_r 32 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --humaneval_eval_loss \
        --eval_reruns 30 
