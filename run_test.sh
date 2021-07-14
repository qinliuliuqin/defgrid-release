export PYTHONPATH=/work/defgrid
export CUDA_VISIBLE_DEVICES=6,7
python3 scripts/inference/test_def_grid_full.py \
              --debug false \
              --version train_on_cityscapes_full \
              --encoder_backbone simplenn \
              --resolution 512 1024 \
              --grid_size 20 40 \
              --w_area 0.005


