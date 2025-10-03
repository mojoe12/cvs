# cvs

```

python setup_env.py
source venv/bin/activate
python train_code/test.py --num_labels 3 --input_dir cvs/data/frames --input_json cvs/data/test_frame.json --timm_model swinv2_base_window12_192.ms_in22k --image_size 192 --saved_weights cvs/models/small_model_weights.temporal_model.pth --batch_size 1 --output_json results/test_out.json

# for training:
gdown --fuzzy 'https://drive.google.com/file/d/15xHUKYwzGNDt-b9Ti6I_S0ejflATfymv/view?usp=sharing'
unzip -q ed_hash_17859240328290.zip
python3 train_code/train.py --transformer_model swinv2_base_window12to24_192to384.ms_in22k_ft_in1k --image_size 384 --mlc_batch_size 16 --backbone_weight_decay 0.01 --use_endoscapes --output_file train_code/model_weights --num_epochs 15 --temporal_epochs 15

```
