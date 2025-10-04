# cvs

These two models were my submission to the CVS challenge 2025.

To run the following lines, you will need a directory structure:

data/
├── frames/
│   ├── 024e5eee-3a7b-484d-ab29-d2944d781795_0.jpg
│   ├── 024e5eee-3a7b-484d-ab29-d2944d781795_1.jpg
│   ├── ...
│   └── 024e5eee-3a7b-484d-ab29-d2944d781795_2699.jpg
├── videos/
│   ├── 024e5eee-3a7b-484d-ab29-d2944d781795.mp4
│   └── ...
└── metadata/
    ├── test_frames_AB.json       # Frame list used for Subchallenges A and B
    └── test_frames_C.json        # Frame list used for Subchallenge C
```

results/
### Empty

```
python test.py --num_labels 3 --input_dir data/frames/ --input_json data/metadata/test_frames_AB.json --timm_model swinv2_base_window12to24_192to384.ms_in22k_ft_in1k --image_size 384 --saved_weights large_model_weights.temporal_model.pth --batch_size 2 --output_json results/subchallengeA.json

python test.py --num_labels 3 --input_dir data/frames/ --input_json data/metadata/test_frames_AB.json --timm_model swinv2_base_window12_192.ms_in22k --image_size 192 --saved_weights small_model_weights.temporal_model.pth --batch_size 1 --output_json results/subchallengeB.json

```
