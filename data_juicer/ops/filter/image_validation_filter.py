import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys, CleaningKeys
from data_juicer.utils.mm_utils import load_image

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

import cv2
import numpy as np
import os

@OPERATORS.register_module('image_validation_filter')
@LOADED_IMAGES.register_module('image_validation_filter')
class ImageValidationFilter(Filter):
    """Filter to keep samples within normal validation
    """

    def __init__(self,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.
        
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')
        

    def compute_stats(self, sample, context=False):

        if self.image_key == "image":
            # there is no image in this sample
            sample[Fields.stats][CleaningKeys.validation] = True
            if self.image_key not in sample or not sample[self.image_key]:
                return sample

            # load images
            loaded_image_key = sample[self.image_key]
            sample[Fields.stats][CleaningKeys.validation] = False

            try:
                image = load_image(loaded_image_key)
            except:
                sample[Fields.stats][CleaningKeys.validation] = True
        
        elif self.image_key == "images":
            # check if it's computed already
            if CleaningKeys.validation in sample:
                return sample

            sample[Fields.stats][CleaningKeys.validation] = []
            # there is no image in this sample
            if self.image_key not in sample or not sample[self.image_key]:
                sample[Fields.stats][CleaningKeys.validation] = np.array(
                    [], dtype=np.int64)
                return sample

            # load images
            loaded_image_keys = sample[self.image_key]
            images = {}
            for loaded_image_key in loaded_image_keys:
                if context and loaded_image_key in sample[Fields.context]:
                    # load from context
                    images[loaded_image_key] = sample[
                        Fields.context][loaded_image_key]
                else:
                    if loaded_image_key not in images:
                        # avoid load the same images
                        try:
                            image = load_image(loaded_image_key)
                            images[loaded_image_key] = image
                            sample[Fields.stats][CleaningKeys.validation].append(0)
                            if context:
                                # store the image data into context
                                sample[Fields.context][loaded_image_key] = image
                        except:
                            sample[Fields.stats][CleaningKeys.validation].append(1)

        return sample
        

    def process(self, sample):
        if self.image_key == "image":
            return not sample[Fields.stats][CleaningKeys.validation] 
        
        elif self.image_key == "images":
            validation = np.array(sample[Fields.stats][CleaningKeys.validation])
            
            if self.any:
                return not validation.any()
            else:
                return not validation.all()
        
            
