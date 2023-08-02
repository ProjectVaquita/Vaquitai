import numbers

from jsonargparse.typing import ClosedUnitInterval, PositiveInt

from ..base_op import OPERATORS, Selector


@OPERATORS.register_module('frequency_specified_field_selector')
class FrequencySpecifiedFieldSelector(Selector):
    """Selector to select samples based on the sorted frequency of specified
    field."""

    def __init__(self,
                 text_key: str = '',
                 top_ratio: ClosedUnitInterval = None,
                 topk: PositiveInt = None,
                 reverse: bool = True,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param text_key: Selector based on the specified value
            corresponding to the target key. The target key
            corresponding to multi-level field information need to be
            separated by '.'.
        :param top_ratio: Ratio of selected top specified field value,
            samples will be selected if their specified field values are
            within this parameter. When both topk and top_ratio are set,
            the value corresponding to the smaller number of samples
            will be applied.
        :param topk: Number of selected top specified field value,
            samples will be selected if their specified field values are
            within this parameter. When both topk and top_ratio are set,
            the value corresponding to the smaller number of samples
            will be applied.
        :param reverse: Determine the sorting rule, if reverse=True,
            then sort in descending order.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.text_key = text_key
        self.top_ratio = top_ratio
        self.topk = topk
        self.reverse = reverse

    def process(self, dataset):
        if len(dataset) <= 1 or not self.text_key:
            return dataset

        text_keys = self.text_key.split('.')
        assert text_keys[0] in dataset.features.keys(
        ), "'{}' not in {}".format(text_keys[0], dataset.features.keys())

        field_value_dict = {}
        for i, item in enumerate(dataset[text_keys[0]]):
            field_value = item
            for key in text_keys[1:]:
                assert key in field_value.keys(), "'{}' not in {}".format(
                    key, field_value.keys())
                field_value = field_value[key]
            assert field_value is None or isinstance(
                field_value, str) or isinstance(
                    field_value, numbers.Number
                ), 'The {} item is not String, Numbers or NoneType'.format(i)
            if field_value not in field_value_dict.keys():
                field_value_dict[field_value] = [i]
            else:
                field_value_dict[field_value].append(i)

        select_num = 0
        if not self.top_ratio:
            if not self.topk:
                return dataset
            else:
                select_num = self.topk
        else:
            select_num = self.top_ratio * len(field_value_dict)
            if self.topk and self.topk < select_num:
                select_num = self.topk

        select_index = sum(
            sorted(field_value_dict.values(),
                   key=lambda x: len(x),
                   reverse=self.reverse)[:int(select_num)], [])
        return dataset.select(select_index)
