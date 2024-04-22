import pandas as pd
import os 
import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool


orientation = "front_middle_camera"
parquet_file_path = '/'
df = pd.read_parquet(parquet_file_path)
usable_df = df[df["sensor_name"] == orientation]
print(usable_df)
bundles = ["/" + _ for _ in set(usable_df["bundle_path"])]
cantread_path = "./demos/data_gn/cantread.txt"
result_path = "./demos/data_gn/gn.jsonl"

def worker(json_path):
    try:
        result = dict()
        with open(os.path.join(json_path), 'r') as file:
            json_info = json.load(file)
            for cam in json_info["camera"]:
                if cam['name'] == orientation:
                    img_path = "/" + cam['oss_path']
                    result['json_path'] = json_path
                    result['image'] = [img_path]
                    return result
    except FileNotFoundError:
        with open(cantread_path, "a") as cantread_f:
            cantread_f.writelines(json_path + "\n")

with ThreadPool(processes = 40) as pool:
    results = list(tqdm(pool.imap(worker, bundles), total=len(bundles), desc='JSON Loading'))
    pool.terminate()
    
with open (result_path, "w") as result_f:
    for res in tqdm(results, desc="Data Saving"):
        result_f.write(json.dumps(res) + '\n')