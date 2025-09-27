import torch
import torch.nn as nn
import os
import argparse
import json
import time
from losses import AsymmetricLoss, FocalLoss
from loaders import PadToSquareHeight, CropToSquareHeight, MultiLabelImageDataset, getMLCImageLoader, MultiLabelVideoDataset, getMLCVideoLoader
from models import TimmMLCModel, TemporalMLCPredictor, TemporalMLCTCN, TemporalMLCLSTM

def evalModel(model, dataloader, device, output_file):
    model.eval()
    results = []

    start_time = time.time()  # Start timing
    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(device)
            outputs = model(images)  # [batch_size, 3]
            probs = torch.sigmoid(outputs).cpu().numpy()

            for filename_index in range(len(filenames)):
                filename_list = filenames[filename_index]
                for filename, prob in zip(filename_list, probs[:, filename_index]):
                    pred = [int(p >= 0.5) for p in prob]
                    results.append({
                        "file_name": filename,
                        "pred_ds_prob": prob.tolist(),
                        "pred_ds": pred
                    })

    end_time = time.time()  # End timing
    elapsed = (end_time - start_time) * 1.0 / len(results)
    print(f"Inference took {elapsed:.2f} seconds per operation")

    final_json = {"images": results}

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(final_json, f, indent=4)

    print(f"Saved predictions to {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument('--timm_model', type=str, required=True, help='Path to Timm model specification')
    parser.add_argument('--num_labels', type=int, required=True, help='Number of labels to predict')
    parser.add_argument('--saved_weights', type=str, required=True, help='Path to file representing model weights')
    parser.add_argument('--image_size', type=int, required=True, help='Image Size. Depends on model')
    parser.add_argument('--batch_size', type=int, default=32 if torch.cuda.is_available() else 1,
                        help='Batch size for multi-label classification')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input frames')
    parser.add_argument('--input_json', type=str, required=True, help='Path to input json listing useful frames')
    parser.add_argument('--output_json', type=str, required=True, help='Path to output file for writing outputs')
    return parser.parse_args()

def main():
    args = parse_args()
    height, width = args.image_size, args.image_size
    print(f"Batch size: {args.batch_size}, Image size: {height}x{width}")
    cvs_val_mlc_loader, cvs_val_mlc_dataset = getMLCImageLoader(args.input_dir, args.input_json, height, width, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    assert len(args.timm_model) > 0
    print(f"Using timm model {args.timm_model} as backbone")
    model = TimmMLCModel(args.num_labels, args.timm_model).to(device)
    model.set_backbone(False)
    temporal_model = TemporalMLCLSTM(model, 128, args.num_labels, 3).to(device)
    assert len(args.saved_weights) > 0
    print(f"Loading model weights from {args.saved_weights}")
    temporal_model.load_state_dict(torch.load(args.saved_weights, map_location=device))

    val_mlc_video = getMLCVideoLoader(cvs_val_mlc_dataset, args.batch_size, device)
    evalModel(temporal_model, val_mlc_video, device, args.output_json)

if __name__ == "__main__":
    print(os.cpu_count())
    torch.set_num_threads(os.cpu_count())  # Or use os.cpu_count()
    torch.set_num_interop_threads(os.cpu_count())
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
    main()
