# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional, Union

from torchtune.data import OpenAIToMessages
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer


class DebugableSFTDataset(SFTDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def retrieve_sample(self, index: int) -> Dict[str, Any]:
        return self._data[index]


class DebugablePackedDataset(PackedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def retrieve_sample(self, index: int) -> Dict[str, Any]:
        if type(self.ds) == DebugableSFTDataset:
            return self.ds.retrieve_sample(index)

        raise ValueError("Dataset is not a DebugableSFTDataset")


def instruct_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:

    message_transform = OpenAIToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )

    ds = DebugableSFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )

    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds
