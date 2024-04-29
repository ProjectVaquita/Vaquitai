import jsonlines
from tqdm import tqdm

# Function to convert image paths to JSONL format
def convert_to_jsonl(dataset_names, dataset_counts, image_paths, output_file):
    with jsonlines.open(output_file, mode='w') as writer:
        count_start = 0
        for name, count in zip(dataset_names, dataset_counts):
            count_end = count_start + count
            for path in tqdm(image_paths[count_start:count_end], desc="Converting " + name, unit="image"):
                json_obj = {"images": [path.strip()], "text": "haha", "type": name}
                writer.write(json_obj)
            count_start = count_end

# Dataset names and counts
dataset_names = ["kitti_data", "nuscenes", "waymo", "CC3M", "COCO2014", "FilteredFlickr-30k", 
                 "GQA", "Hateful-Meme", "OCR-VQA", "TextCaps", "VisualGenome", "VizWiz"]
dataset_counts = [47937, 204894, 151805, 527723, 164062, 31783, 
                  148854, 10000, 206671, 25119, 108079, 39703]

# Read image paths from the input file
input_file = "/root/dataset/general_tos.txt"

with open(input_file, 'r') as f_in:
    image_paths = f_in.readlines()

# Output JSONL file
output_file = "datasets.jsonl"

# Convert to JSONL format and save to output file
convert_to_jsonl(dataset_names, dataset_counts, image_paths, output_file)

print("Conversion complete. JSONL file saved as", output_file)

