import pandas as pd
import os 
import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import random
import jsonlines
from datetime import datetime

data_path = ""
result_path = "./demos/data_bdd/bdd.jsonl"

def get_python_files(data_path):
    python_files = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    return python_files

image_paths = get_python_files(data_path=data_path)

def worker(json_path):
    result = dict()
    result['image'] = json_path
    return result

with ThreadPool(processes = 40) as pool:
    results = list(tqdm(pool.imap(worker, image_paths), total=len(image_paths), desc='IMG Loading'))
    pool.terminate()
    
with jsonlines.open(result_path, mode='a') as writer:
    writer.write_all(results)
    