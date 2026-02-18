

python test.py \
    --devices 2 \
    --log_dir ./exp/ \
    --exp_name exp_name \
    --input_channel 1 \
    --output_channel 1 \
    --data_dir example_data \
    --data_info_csv example_data.csv \
    --batch_size 1 \
    --norm_mode standard \
    --checkpoint_file exp_name/checkpoint_xxx.pt \
    --sample_steps 500 \
    --noise_step 500 \


    