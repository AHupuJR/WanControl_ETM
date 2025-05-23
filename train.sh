python examples/wanvideo/train_wan_t2v.py \
  --task train \
  --train_architecture full \
  --dataset_path ./datasets/toy_dataset_control \
  --output_path ./ \
  --dit_path "/work/lei_sun/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 500 \
  --max_epochs 1000 \
  --learning_rate 4e-5 \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --dataloader_num_workers 8 \
  --control_layers 15
  