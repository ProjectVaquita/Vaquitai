import numpy as np
from PIL import ImageOps, Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys, EmbKeys
from data_juicer.utils.mm_utils import (SpecialTokens, load_image,
                                        remove_special_tokens)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Generator
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'image_feature_extract_generator'


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageFeatureExtractGenerator(Generator):
    """Extracting feature vectors from images"""

    def __init__(self,
                 hf_blip='Salesforce/blip-itm-base-coco',
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
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_key = prepare_model(model_type='huggingface', pretrained_model_name_or_path=hf_blip)
        self.model, self.processor = get_model(self.model_key)
        self.model = self.model.to(self.device)
        self.transform = transforms.Compose([
                transforms.Resize([336, 336]),
                transforms.ToTensor()
            ])

    def compute_embedding(self, sample):
        # check if it's computed already
        if EmbKeys.image_embedding in sample:
            return sample

        # there is no image in this sample
        sample[EmbKeys.image_embedding] = []
        if self.image_key not in sample or not sample[self.image_key]:
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        if not isinstance(loaded_image_keys, list):
            loaded_image_keys = [loaded_image_keys]
        
        for image_key in loaded_image_keys:
            image = load_image(image_key)
            image = self.transform(image).unsqueeze(0).to(self.device)

            # compute image embeddings
            image_embeds = self.model.vision_model(image)[0] 
            image_feature = F.normalize(self.model.vision_proj(image_embeds[:,0,:]),dim=-1).half()
            sample[EmbKeys.image_embedding].append(image_feature.cpu().tolist()[0])

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
        dataset = dataset.map(self.compute_embedding,
                              desc= 'image_feature_extract_process')

        return dataset