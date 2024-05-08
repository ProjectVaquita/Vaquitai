import pandas as pd
import os 
import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import random
import jsonlines
from datetime import datetime


data_types = ["train", "val"]
result_path = "./demos/data_bdd/bdd_anno.jsonl"

def process(data_type):
    data_path = "" % data_type
    anno_path = "" % data_type 
    def get_python_files(data_path):
        python_files_spec = [file for file in os.listdir(data_path)]
        # python_files_abs = [os.path.join(data_path, file) for file in python_files_spec]
        # return python_files_spec, python_files_abs
        return python_files_spec

    image_paths_spec = get_python_files(data_path=data_path)

    with open(anno_path, "r") as json_obj:
        anno_file = json.load(json_obj)

    anno_json = dict()
    for anno in anno_file:
        anno_json[anno["name"]] = [anno['attributes'], anno['labels']]
        
    # print(anno_json)
        
    def worker(image_path):
        result = dict()
        result['image'] = os.path.join(data_path, image_path)
        annos = anno_json.get(image_path, [None, None])
        result['attributes'] = annos[0]
        result['labels'] = annos[1]
        result['data_source'] = data_type
        return result

    with ThreadPool(processes = 40) as pool:
        results = list(tqdm(pool.imap(worker, image_paths_spec), total=len(image_paths_spec), desc='IMG Loading'))
        pool.terminate()
        
    with jsonlines.open(result_path, mode='a') as writer:
        writer.write_all(results)

for data_type in data_types:
    process(data_type)