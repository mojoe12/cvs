import os
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

def get_jpg_resolutions(directory):
    resolutions = []

    for file in os.listdir(directory):
        if file.lower().endswith('.jpg'):
            path = os.path.join(directory, file)
            try:
                with Image.open(path) as img:
                    resolutions.append(img.size)  # (width, height)
            except Exception as e:
                print(f"Error reading {file}: {e}")

    return resolutions

def plot_resolution_histogram(resolutions):
    counter = Counter(resolutions)
    labels = [f"{w}×{h}" for (w, h) in counter.keys()]
    counts = list(counter.values())

    # Sort by frequency
    sorted_data = sorted(zip(labels, counts), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_counts = zip(*sorted_data)

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_labels, sorted_counts, color='skyblue')
    plt.xlabel('Image Resolution (width × height)')
    plt.ylabel('Count')
    plt.title('Histogram of JPG Image Resolutions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    current_dir = os.getcwd()
    resolutions = get_jpg_resolutions(current_dir)

    if resolutions:
        plot_resolution_histogram(resolutions)
    else:
        print("No JPG files found.")

if __name__ == "__main__":
    main()

