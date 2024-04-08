from data_juicer.utils.constant import Fields, DEFAULT_PREFIX
from datasets import Dataset, load_dataset, concatenate_datasets

from ..base_op import OPERATORS, Mycleanlab
from ..op_fusion import LOADED_IMAGES

import numpy as np
import pandas as pd
from cleanvision import Imagelab
from PIL import Image

from tqdm import tqdm
from multiprocessing.pool import ThreadPool


@OPERATORS.register_module('cleanvision_mycleanlab')
@LOADED_IMAGES.register_module('cleanvision_mycleanlab')
class CleanvisionMycleanlab(Mycleanlab):
    """Filter to keep samples within normal blurriness
    """

    def __init__(self,
                 issues: list = ["is_odd_size_issue", "is_odd_aspect_ratio_issue", 
                                "is_low_information_issue", "is_light_issue", 
                                "is_grayscale_issue", "is_dark_issue", "is_blurry_issue"], 
                                # "is_exact_duplicates_issue", "is_near_duplicates_issue"],
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.
        
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.issues = issues
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')
        
    # def save_results(self, sample):
    #     for issue in self.issues:
    #         index = self.hf_dataset[self.image_key + "_path"].index(sample.get(self.image_key))
    #         sample[Fields.stats][DEFAULT_PREFIX + issue] = self.res_df.iloc[[index]].get(issue).to_list()[0]
    #     return sample

    def save_results(self, sample, sample_index):
        for issue in self.issues:
            images_list = sample.get(self.image_key)
            for i in range(len(images_list)):
                path_index = self.index_lookup.get(images_list[i])
                if path_index is not None:
                    for p_ind in path_index:
                        p_index_split = [int(_) for _ in p_ind.split("-")]
                        if p_index_split[0] == sample_index:
                            if sample.get(DEFAULT_PREFIX + issue, None) is None:
                                sample[Fields.stats][DEFAULT_PREFIX + issue] = [False for _ in range(len(images_list))]
                            sample[Fields.stats][DEFAULT_PREFIX + issue][p_index_split[1]] = self.res_df.iloc[[p_index_split[2]]].get(issue).to_list()[0]

        return sample
    
    
    def create_index_list(self, list_of_lists):
        index_list = []
        for i, sublist in enumerate(list_of_lists):
            for j, _ in enumerate(sublist):
                count = len(index_list)
                index_list.append(f"{i}-{j}-{count}")
        return index_list


    def pre_process(self, dataset, num_proc):
        image_paths = dataset[self.image_key]
        hf_dataset_lst, res_df_lst = [], []
        chunk_size = 100000
        for j, image_pathxs in enumerate([image_paths[i : i + chunk_size] for i in range(0, len(image_paths), chunk_size)]):
            def worker(_):
                return [Image.open(x) for x in _] 
            with ThreadPool(processes = num_proc) as pool:
                image_keys = list(tqdm(pool.imap(worker, image_pathxs), total=len(image_pathxs), desc='Images Loading'))
                pool.terminate()
            
            my_dict = {self.image_key: sum(image_keys, []), self.image_key + "_path": sum(dataset[self.image_key][j * chunk_size : (j + 1) * chunk_size], [])}
            tmp_dataset = Dataset.from_dict(my_dict)
            imagelab = Imagelab(hf_dataset=tmp_dataset, image_key=self.image_key)
            imagelab.find_issues()
            hf_dataset_lst.append(tmp_dataset.remove_columns([self.image_key]))
            res_df_lst.append(imagelab.issues)
        self.hf_dataset = concatenate_datasets(hf_dataset_lst)
        self.res_df = pd.concat(res_df_lst)
        img_list = self.hf_dataset[self.image_key + "_path"]
        index_list = self.create_index_list(image_paths)
        self.index_lookup = {}
        for img, index in zip(img_list, index_list):
            if img in self.index_lookup:
                self.index_lookup[img].append(index)
            else:
                self.index_lookup[img] = [index]
        dataset = dataset.map(lambda item, index: self.save_results(item, index), with_indices=True) 
        return dataset
    
    
    def process(self, sample):
        for issue in self.issues:
            tmp = sample[Fields.stats][DEFAULT_PREFIX + issue]
            if True in tmp:
                return False
        return True
            
        # keep_bools = np.array([
        #     self.min_size <= image_size <= self.max_size
        #     for image_size in image_sizes
        # ])
        # if len(keep_bools) <= 0:
        #     return True

        # # different strategies
        # if self.any:
        #     return keep_bools.any()
        # else:
        #     return keep_bools.all()