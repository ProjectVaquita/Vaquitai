import numpy as np
from PIL import ImageOps, Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from umap import UMAP
import concurrent.futures

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys, EmbKeys
from data_juicer.utils.mm_utils import (SpecialTokens, load_image,
                                        remove_special_tokens)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Generator
from ..op_fusion import LOADED_IMAGES

# Assuming the necessary imports and context are already provided

OP_NAME = 'feature_reduce_generator'

@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class FeatureReduceGenerator(Generator):
    """Extracting feature vectors from images and reducing their dimensionality"""

    def __init__(self, *args, **kwargs):
        """
        Initializes the feature reducer with UMAP.
        """
        super().__init__(*args, **kwargs)
        # Adjust 'n_jobs' to match your system's CPU cores
        self.model = UMAP(n_neighbors=15, n_components=2, metric='cosine', n_jobs=-1)

    def process(self, dataset):
        """
        Process the dataset to reduce the dimensionality of image embeddings.
        """
        index = '.'.join([Fields.stats, StatsKeys.image_embedding])
        image_embeddings = np.concatenate(dataset[index])
        image_paths = np.concatenate(dataset[self.image_key])

        # Fit the model and transform the embeddings
        embeddings_2d_flat = self.model.fit_transform(image_embeddings).tolist()

        # Create a mapping from image paths to their reduced embeddings
        img_emb2d_map = {image_paths[i]: embeddings_2d_flat[i] for i in range(len(image_paths))}

        def save_emb2d(sample):
            """
            Save the 2D embeddings back into the dataset.
            """
            emb2d_list = [img_emb2d_map[img] for img in sample[self.image_key]]
            sample[Fields.stats][StatsKeys.image_embedding_2d] = emb2d_list
            return sample

        # Map the function across the dataset, adjust 'num_proc' as per your system's capabilities
        dataset = dataset.map(save_emb2d, desc='feature_reduce_process', num_proc=8)
        return dataset