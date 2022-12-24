#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:weishu
# datetime:2022/11/1 4:53 下午
# software: PyCharm
import numpy as np
import torch


class TableGraphLabelConverter(object):

    def __init__(self, alphabet):
        self.alphabet = alphabet  # for `-1` index

        self._index = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self._index[char] = i

    def encode(self, texts):
        """Support batch or single str."""
        result_texts = []
        for batch_text in texts:
            for text in batch_text:
                if len(text) == 0:
                    text = " "
                text_index_list = [self._index.get(char, 0) for char in text]
                text_index_list = torch.from_numpy(np.array(text_index_list)).to(torch.int)
                result_texts.append(text_index_list)

        return result_texts
