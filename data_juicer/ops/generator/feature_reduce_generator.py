import numpy as np
from PIL import ImageOps, Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from umap import UMAP

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys, EmbKeys
from data_juicer.utils.mm_utils import (SpecialTokens, load_image,
                                        remove_special_tokens)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Generator
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'feature_reduce_generator'


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class FeatureReduceGenerator(Generator):
    """Extracting feature vectors from images"""

    def __init__(self,
                 method='umap',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_blip: blip model name on huggingface to compute
            the matching score between image and text.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.model = UMAP(n_neighbors=30, n_components=2, metric='cosine', random_state=42)

    def compute_embedding(self, sample):
        # check if it's computed already
        if EmbKeys.image_embedding in sample:
            return sample

        # there is no image in this sample
        sample[EmbKeys.image_embedding] = []
        if self.image_key not in sample or not sample[self.image_key]:
            return sample

        # load images
        image = load_image(sample[self.image_key])
        image = self.transform(image).unsqueeze(0)


        # compute image embeddings
        image_embeds = self.model.vision_model(image)[0] 
        image_feature = F.normalize(self.model.vision_proj(image_embeds[:,0,:]),dim=-1).half()
        sample[EmbKeys.image_embedding] = image_feature.cpu().tolist()[0]

        return sample

    def process(self, dataset):
        """
        For doc-level, dataset --> dataset.

        :param dataset: input dataset
        :param show_num: number of traced samples used when tracer is
            open.
        :return: deduplicated dataset and the sampled duplicate pairs.
        """
        # no need to deduplicate because too few samples
        image_embeddings = dataset[EmbKeys.image_embedding]

        image_embeddings_flat = [emb for sublist in image_embeddings for emb in sublist]
        image_embeddings_flat = np.array(image_embeddings_flat)
        embeddings_2d_flat = self.model.fit_transform(image_embeddings_flat).tolist()
        
        embeddings_2d = []
        cnt = 0
        for idx, emb in enumerate(image_embeddings):
            length = len(emb)
            embeddings_2d.append(embeddings_2d_flat[cnt:cnt+length])
            cnt += length
        dataset = dataset.add_column(name=EmbKeys.image_embedding_2d, column=embeddings_2d)
        return dataset