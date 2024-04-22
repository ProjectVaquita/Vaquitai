
import jsonlines
from multiprocessing import Pool
from tqdm import tqdm
import os


def load_jsonl(jsonl_file):
    with jsonlines.open(jsonl_file) as reader:
        return [entry for entry in reader]

def load_jsonl_file(jsonl_file_path, num_processes=64):
    with Pool(processes=num_processes) as pool:
        # Use tqdm to wrap the pool's imap function to display progress bar
        jsonl_data = list(tqdm(pool.imap(load_jsonl, [jsonl_file_path]), total=1, desc="Loading JSONL", unit="file"))

    # Flatten the list of lists into a single list
    jsonl_data = [entry for sublist in jsonl_data for entry in sublist]

    return jsonl_data

def extract_image_paths(obj):
    paths = []
    for key in obj.keys():
        if key.startswith('dup'):
            if key != "dup_num":
                paths.extend(obj[key]["images"])
    return paths

def process_file(jsonl_file):
    image_paths = []
    with jsonlines.open(jsonl_file) as reader:
        with Pool(processes=64) as pool:
            for result in tqdm(pool.imap_unordered(extract_image_paths, reader)):
                image_paths.extend(result)
    return image_paths


def remove_entries_by_image_paths(original_file_path, corresponding_file_path, image_paths_to_remove, output_root):
    # Load the original and corresponding JSONL files

    original_data = load_jsonl_file(original_file_path)
    # Create a set of image paths to remove for efficient lookup
    image_paths_set = set(image_paths_to_remove)
    # Determine indices of entries to remove from both files
    indices_to_remove = {i for i, entry in enumerate(original_data) if entry.get('images', []) and entry['images'][0] in image_paths_set}
    # Filter out corresponding entries from both files
    filtered_original_data = [entry for i, entry in enumerate(original_data) if i not in indices_to_remove]
    
    with jsonlines.open("%s/demo-processed.jsonl" % output_root, mode='w') as writer:
        writer.write_all(filtered_original_data)
                
    for cor_file_path in corresponding_file_path:
        corresponding_data = load_jsonl_file(cor_file_path)

        filtered_corresponding_data = [entry for i, entry in enumerate(corresponding_data) if i not in indices_to_remove]

        assert len(filtered_original_data) == len(filtered_corresponding_data)
        
        with jsonlines.open("%s/%s_demo-processed_stats.jsonl" % (output_root, cor_file_path.split("/")[-2]), mode='w') as writer:
            writer.write_all(filtered_corresponding_data)


if __name__ == "__main__":
    root_path = ""
    base_op = "%s/demo-1080w-cleanvision" % root_path
    imagedup_file = "%s/demo-1080w-imagedup/trace/duplicate-image_deduplicator.jsonl" % root_path
    docdup_file = "%s/demo-1080w-docdup/trace/duplicate-document_deduplicator.jsonl" % root_path
    
    all_dup_image_paths = process_file(imagedup_file)
    all_dup_doc_paths = process_file(docdup_file)
    
    
    # Print all image paths
    print(len(all_dup_image_paths))
    print(all_dup_image_paths[:10])

    print(len(all_dup_doc_paths))
    print(all_dup_doc_paths[:10])
    
    base_data_path = "%s/demo-processed.jsonl" % base_op
    # base_stats_path_list = ["%s/demo-processed_stats.jsonl" % base_op, "%s/demo-1080w-sim/demo-processed_stats.jsonl" % root_path, "%s/demo-1080w-text/demo-processed_stats.jsonl" % root_path]
    base_stats_path_list = ["%s/demo-1080w-text/demo-processed_stats.jsonl" % root_path]
    
    
    image_paths_to_remove = all_dup_image_paths + all_dup_doc_paths
    output_root = ""
    remove_entries_by_image_paths(base_data_path, base_stats_path_list, image_paths_to_remove, output_root)
    
    