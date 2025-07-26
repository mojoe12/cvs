# cvs

```
apt install python3-pip
apt install unzip
pip install gdown torch torchvision pandas scikit-learn scikit-image pycocotools timm
gdown --fuzzy 'https://drive.google.com/file/d/15xHUKYwzGNDt-b9Ti6I_S0ejflATfymv/view?usp=sharing'
unzip ed_hash_17859240328290.zip
wget 'https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip'
unzip endoscapes.zip

```

for reference, the endoscapes and cvs annotation jsons specify the segmentations in RLE format.
We think this is a mistake, because all known coco interpreters expect RLE for iscrowd=1 and polygon otherwise.
As such, we first convert the 4 jsons (train, val, test for endoscapes and the whole thing for coco)
We also remove several ~ <10 images across the 700 + 493 that do not convert successfully.
This appears to be either an issue with image corruption (42c6fdfa-e032-4b85-adb5-cedfd633de68_1350.jpg is all blue)
or with there being no labels whatsoever (this is reasonable, sometimes the camera is not facing the relevant anatomy)

```
TRAIN:
python3 train_code/train.py --transformer_model swinv2_base_window12to24_192to384.ms_in22k_ft_in1k --image_size 384 --mlc_batch_size 16 --backbone_weight_decay 0.01 --use_endoscapes --output_file train_code/model_weights --num_epochs 15 --temporal_epochs 15

LOAD:
python3 train_code/train.py --transformer_model swinv2_base_window12to24_192to384.ms_in22k_ft_in1k --image_size 384 --saved_weights train_code/model_weights.temporal_model.pth

```
