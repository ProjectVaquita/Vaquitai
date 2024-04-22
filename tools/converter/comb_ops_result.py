import jsonlines
from tqdm import tqdm
from multiprocessing import Pool


root_path = "/mnt/share_disk/songyuhao/DCAI/processed_data/demo-1080w-rmdup"
base_file = "%s/demo-1080w-cleanvision_demo-processed_stats.jsonl" % root_path
new_op_file_names = ["%s/demo-1080w-sim_demo-processed_stats.jsonl" % root_path, "%s/demo-1080w-text_demo-processed_stats.jsonl" % root_path]

# Combined file name
combined_file_name = "%s/combined_stats.jsonl" % root_path


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

# Function to read and combine files
def combine_files(base_stats, file_names, combined_file_name):
    for file_name in tqdm(file_names):
        new_stats = load_jsonl_file(file_name)
        for i, new_stat in tqdm(enumerate(new_stats)):
            stat = new_stat["__dj__stats__"]
            base_stats[i]["__dj__stats__"].update(stat)
        
    with jsonlines.open(combined_file_name, mode='w') as writer:
        writer.write_all(base_stats)
# Call the function
base_stats = load_jsonl_file(base_file)
print("loaded base")
combine_files(base_stats, new_op_file_names, combined_file_name)
