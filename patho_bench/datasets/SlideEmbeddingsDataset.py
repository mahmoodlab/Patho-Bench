from patho_bench.datasets.BaseDataset import BaseDataset
import torch
import os

"""
SlideEmbeddingsDataset loads a single pre-pooled feature per sample.
A sample may be a slide or a collection of slides (e.g. a patient).
"""

class SlideEmbeddingsDataset(BaseDataset):
    def __init__(self, split, load_from):
        '''
        Loads slide-level embeddings for a sample.
        A sample may be a slide or a collection of slides (e.g. a patient).
        
        Args:
            split (BaseSplit): Split object
            load_from (str or list): One or more directories containing h5 files with pooled sample embeddings
        '''
        super().__init__(split)
        self.load_from = load_from

        if isinstance(self.load_from, str):
            self.load_from = [self.load_from]

        self.paths = {}
        for path in self.load_from:
            for file in os.listdir(path):
                if file.endswith('.h5'):
                    sample_id = os.path.splitext(file)[0]
                    self.paths[sample_id] = os.path.join(path, file)  # Store path to h5 file
        
    def __getitem__(self, idx):
        '''
        Args:
            idx (int or str): Index of sample to return or sample ID
        '''
        sample_id = self.ids[idx] if isinstance(idx, int) else idx
        assert sample_id in self.paths, f'Sample ID {sample_id} not found in {self.load_from}. Please ensure pooling was performed correctly.'
        features, attributes = self.load_h5(self.paths[sample_id], keys = ['features'])
        features = features['features'].squeeze(0)
        assert features.ndim == 1, f'Features must be 1-dimensional (feature_dim), got shape: {features.shape}'

        return {'id': sample_id, 'features': torch.from_numpy(features), 'attributes': attributes}