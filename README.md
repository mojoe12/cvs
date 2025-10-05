# cvs

Before executing the code below, ensure (1) `python` exists on your system (try `which python`), and (2) Git LFS is installed (via `git lfs install`).
Depending on your system's Python installation, you may need to rename `python` to `python3` in the command-line prompts below.

## Running an initial test:
```
python setup_env.py
source venv/bin/activate
git lfs fetch --all
python train_code/test.py --num_labels 3 --input_dir cvs/data/frames --input_json cvs/data/test_frame.json --timm_model swinv2_base_window12_192.ms_in22k --image_size 192 --saved_weights cvs/models/small_model_weights.temporal_model.pth --batch_size 1 --output_json results/test_out.json
```

## Training our model:
```
gdown --fuzzy 'https://drive.google.com/file/d/15xHUKYwzGNDt-b9Ti6I_S0ejflATfymv/view?usp=sharing'
unzip -q ed_hash_17859240328290.zip
python train_code/train.py --timm_model swinv2_base_window12_192.ms_in22k --image_size 192 --num_labels 3 --batch_size 16 --backbone_weight_decay 0.01 --output_file train_code/model_weights --num_epochs 15 --temporal_epochs 15 --training_data cvs/data/train_mlc_data.csv:sages_cvs_challenge_2025/frames/ --validation_data cvs/data/val_mlc_data.csv:sages_cvs_challenge_2025/frames/
```
