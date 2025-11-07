#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch

from data.bioparse.hierarchy import remove_mols, add_dummy_mol
from data.bioparse.utils import recur_index
from utils import register as R

from .peptide import PeptideDataset
from .resample import SizeResampler, SizeSamplerByPocketSpace
from .base import transform_data


@R.register('MoleonlyDataset')
class MoleonlyDataset(PeptideDataset):

    def __init__(self, mmap_dir, specify_data = None, mask_ratio=None, specify_index = None, cluster = None, length_type = 'atom'):
        super().__init__(mmap_dir, specify_data, specify_index, cluster, length_type)
        self.mask_ratio = mask_ratio

    def __getitem__(self, idx):

        data = super().__getitem__(idx)
        # set position ids of small molecules to zero
        data['position_ids'][data['generate_mask']] = 0
        generate_mask = torch.rand(data['generate_mask'].shape) < self.mask_ratio
        if generate_mask.sum()<2:
            generate_mask[-2:] = True
        data['generate_mask'] = generate_mask

        data['center_mask'] = ~data['generate_mask']
        return data

