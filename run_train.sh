export CUDA_VISIBLE_DEVICES=0,1,2
python3 scripts/train/train_def_grid_full.py \
              --debug false \
              --version train_on_cityscapes_full \
              --encoder_backbone simplenn \
              --resolution 512 1024 \
              --grid_size 20 40 \
              --w_area 0.005

