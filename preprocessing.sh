CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/train_wan_t2v.py \
  --task data_process \
  --dataset_path ./datasets/toy_dataset_control \
  --output_path ./models \
  --text_encoder_path "/work/lei_sun/models/Wan2.1-Fun-1.3B-Control/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "/work/lei_sun/models/Wan2.1-Fun-1.3B-Control/Wan2.1_VAE.pth" \
  --tiled \
  --num_frames 10 \
  --height 480 \
  --width 832