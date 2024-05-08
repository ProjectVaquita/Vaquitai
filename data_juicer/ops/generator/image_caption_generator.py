import random

import numpy as np
from jsonargparse.typing import PositiveInt

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_image
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper, Generator
from ..deduplicator.document_simhash_deduplicator import \
    DocumentSimhashDeduplicator
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'image_caption_generator'

with AvailabilityChecking(['torch', 'transformers'], OP_NAME):
    import torch
    import transformers  # noqa: F401

    # avoid hanging when calling blip2 in multiprocessing
    torch.set_num_threads(1)


def jaccard_similarity_np(hash_val1, hash_val2):
    equal_hashes = np.sum(hash_val1 == hash_val2)
    total_hashes = len(hash_val1)  # Assume both vectors are the same length
    return equal_hashes / total_hashes


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageCaptionGenerator(Generator):
    """Mapper to generate samples whose captions are generated based on
    another model and the figure."""

    def __init__(self,
                 hf_blip='Salesforce/blip-image-captioning-base',
                 caption_num: PositiveInt = 1,
                 keep_candidate_mode: str = 'random_any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_blip2: blip2 model name on huggingface to generate caption
        :param caption_num: how many candidate captions to generate
        for each image
        :param keep_candidate_mode: retain strategy for the generated
        $caption_num$ candidates.
            'random_any': Retain the random one from generated captions
            'similar_one': Retain the generated one that is most similar to the
                original caption
            'all': Retain all generated captions by concatenation
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if keep_candidate_mode not in [
                'random_any', 'similar_one_simhash', 'all'
        ]:
            raise ValueError(
                f'Keep strategy [{keep_candidate_mode}] is not supported. '
                f'Can only be one of '
                f'["random_any", "similar_one_simhash", "all"].')
        self.model_key = prepare_model(model_type='huggingface',
                                        pretrained_model_name_or_path=hf_blip)
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_in_ctx = None
        self.img_processor_in_ctx = None
        self.caption_num = caption_num
        self.keep_candidate_mode = keep_candidate_mode
        self.extra_args = kwargs
        self.model, self.img_processor = get_model(model_key=self.model_key)
        self.model = self.model.to(self.device)

    def caption(self, sample, context=True):
        # there is no image in this sample
        if Fields.stats not in sample:
            sample[Fields.stats] = {}

        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_sizes] = np.array(
                [], dtype=np.float64)
            return sample

        sample[Fields.stats][StatsKeys.image_caption] = []
        loaded_image_keys = sample[self.image_key]
        if not isinstance(loaded_image_keys, list):
            loaded_image_keys = [loaded_image_keys]

        # 1. load image(s)
        for image_key in loaded_image_keys:
            image = load_image(image_key)
            if image.size[0] < 2 or image.size[1] < 2:
                image = image.resize((224,224))
            inputs = self.img_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_length=60)
            image_caption_text = self.img_processor.decode(outputs[0], skip_special_tokens=True)
            sample[Fields.stats][StatsKeys.image_caption].append(image_caption_text)

        return sample

    def process(self, dataset, num_proc):  
        dataset = dataset.map(self.caption,
                            num_proc=num_proc,
                            desc=OP_NAME + '_process')

        return dataset