import torch
import numpy as np
import copy
import h5py
import os
from patho_bench.config.ConfigMixin import ConfigMixin

"""
This is the BaseDataset class, which is inherited by all dataset classes.
It has a universal sampling function, a function for loading features from a path, and a function for getting the dataloader from the dataset
"""

class BaseDataset(torch.utils.data.Dataset, ConfigMixin):
    def __init__(self, split):
        '''
        Initializes the dataset.

        Args:
            dataset_config (BaseDatasetConfig): Config object for dataset
        '''
        super().__init__()
        self.split = split
        self.data = {sample['id']: sample for sample in self.split.data}   # Load all samples as dict
        self.ids = list(self.data.keys())
        self.num_folds = self.split.num_folds
        self.current_iter = None
        
    def get_subset(self, iteration, fold):
        '''
        Returns a subset of the dataset for a specific iteration and fold.

        Args:
            iteration (int): Index of the iteration
            fold (str): 'train', 'val', or 'test'

        Returns:
            subset (BaseDataset): Subset of the dataset
        '''
        subset = self.copy()
        subset.current_iter = iteration
        subset.data = {sample['id']: sample for sample in self.split.data if sample['folds'][iteration] == fold}   # Filter to samples for this iteration and fold
        subset.ids = list(subset.data.keys())

        # If dataset has a child_datasets attribute, update child datasets
        if hasattr(subset, 'child_datasets'):
            for dataset_name, dataset in subset.child_datasets.items():
                assert not hasattr(dataset, 'child_datasets'), f'BaseDataset.get_subset() does not support multiple levels of child datasets. Dataset name: {dataset_name}'
                subset.child_datasets[dataset_name] = dataset.get_subset(iteration, fold)
        
        if len(subset.ids) == 0:
            return None
        
        return subset

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    @staticmethod
    def _sample(bag, sample_n, pad, sample_indices):
        '''
        Randomly sample elements from bag, with optional padding if there are fewer than sample_n in bag.
        Bags can be tensors, arrays, or lists.
        Always samples along first dimension of bag.

        Args:
            bag (ArrayLike): ArrayLike of shape (n, ...)
            sample_n (int): Number of elements to sample from bag. If None, will return all elements in bag in shuffled order.
            pad (bool): If True, pad result with zero tensor of shape (sample_n - n, ...) if sample_n > n and bag is tensor, otherwise pad with Nones. If False, return n elements.
            sample_indices (list): List of predefined indices to sample from bag. If None, will sample randomly.
        Returns:
            baglet (ArrayLike): Zero-padded or none-padded ArrayLike of shape (sample_n, ...) or (n, ...) if n < sample_n and pad is False
            mask (torch.Tensor): Tensor of shape (bag_size,) or (sample_n,) indicating non-padding instances in the baglet
        '''
        if sample_n is None:
            sample_n = len(bag)

        if len(bag) < sample_n:
            # Keep all elements in bag, with optional padding
            if pad:
                mask = torch.cat([torch.ones(len(bag)), torch.zeros(sample_n - len(bag))]) # Shape: (sample_n,)
                if isinstance(bag, torch.Tensor):
                    # Pad with zeros
                    padding = torch.zeros((sample_n - len(bag), ) + bag.shape[1:]) # Shape: (sample_n - n, bag.shape[1], bag.shape[2], ...)
                    baglet = torch.cat([bag, padding], dim=0)
                elif isinstance(bag, list):
                    # Pad with Nones
                    padding = [None] * (sample_n - len(bag))
                    baglet = bag + padding
                elif isinstance(bag, np.ndarray):
                    # Pad with zeros
                    padding = np.zeros((sample_n - len(bag), ) + bag.shape[1:])
                    baglet = np.concatenate([bag, padding], axis=0)
                else:
                    raise NotImplementedError(f'Bag type {type(bag)} not implemented')
            else:
                mask = torch.ones(len(bag))
                baglet = bag
        elif len(bag) >= sample_n:
            # Sample sample_n elements from bag
            mask = torch.ones(sample_n)
            if sample_indices is None:
                sample_indices = torch.randperm(len(bag))[:sample_n].tolist()
            baglet = [bag[i] for i in sample_indices]
            if isinstance(bag, torch.Tensor):
                baglet = torch.stack(baglet, dim=0)
            elif isinstance(bag, list):
                pass
            elif isinstance(bag, np.ndarray):
                baglet = np.stack(baglet, axis=0)

        # Check that baglet is of same type as bag
        assert isinstance(baglet, type(bag)), f'Baglet type {type(baglet)} does not match bag type {type(bag)}'

        # Make mask bool
        mask = mask.bool()

        return baglet, mask, sample_indices
        
    def collate_fn(self, batch):
        '''
        Recursively collates a list of dicts into a single dict of stacked tensors or lists. Adds current_iter to collated dict.
        The output of this function is passed to the model as a batched input.

        Args:
            batch (list[dict]): List of dicts to collate

        Returns:
            collated (dict): Dict of stacked tensors or lists
        '''

        # Start the recursive collation process
        collated = {}
        for key in batch[0].keys():
            collated[key] = self.recursive_collate([d[key] for d in batch])

        # Add current_iter to collated dict because it might be needed for loss functions
        collated['current_iter'] = self.current_iter

        return collated
    
    def recursive_collate(self, items):
        '''
        Recursively collates a list of items (dicts, tensors, or other types).

        Args:
            items (list): List of items to collate

        Returns:
            collated_item: Collated tensors, dicts, or lists
        '''
        try:
            if isinstance(items[0], torch.Tensor):
                return torch.stack(items, dim=0)
            elif isinstance(items[0], np.ndarray):
                return np.stack(items, axis=0)
            elif isinstance(items[0], dict):
                keys = items[0].keys()
                return {key: self.recursive_collate([d[key] for d in items]) for key in keys}
            elif isinstance(items[0], list):
                return [self.recursive_collate([d[i] for d in items]) for i in range(len(items[0]))]
            else:
                return items
        except Exception as e:
            print(f'Recursive collation failed with error: {e}')
            print(f'Attempted to collate items with shapes: {[item.shape if isinstance(item, torch.Tensor) or isinstance(item, np.ndarray) else item for item in items]}')
    
    @staticmethod
    def load_h5(load_path, keys = None):
        '''
        Loads an hdf5 file and returns a dictionary of assets

        Args:
            load_path (str): The path to the hdf5 file
            keys (list, optional): A list of keys to load. Defaults to None.

        Returns:
            assets (dict): A dictionary of assets
            attributes (dict): A dictionary of attributes
        '''
        assert isinstance(keys, list) or keys is None, 'keys must be a list or None'
        assert os.path.exists(load_path), f'File {load_path} does not exist'
        
        try:
            with h5py.File(load_path, 'r') as file:
                if keys is None:
                    keys = list(file.keys())
                assets = {key: file[key][:] for key in keys}
                attributes = {key: dict(file[key].attrs) for key in keys}
        except Exception as e:
            raise Exception(f'\033[91mError loading h5 file at {load_path}\033[0m')
                    
        return assets, attributes

    def copy(self):
        '''
        Returns a deep copy of the dataset.

        Returns:
            dataset (BaseDataset): Deep copy of dataset
        '''
        return copy.deepcopy(self)
    
    def get_datasampler(self, sampler = 'random'):
        '''
        Initialize sampler for a dataloader

        Returns:
            sampler (str or dict): Sampler for a dataloader. Defaults to RandomSampler.
        '''
        # Should probably be moved to child datasets or standalone class.
        if sampler == 'sequential':
            return torch.utils.data.SequentialSampler(self)
        elif sampler == 'random':
            return torch.utils.data.RandomSampler(self)
        elif isinstance(sampler, dict) and sampler['type'] == 'weighted':
            raise NotImplementedError(f'WeightedRandomSampler not implemented yet.')
            all_labels = [self.get_labels_for_case(idx)[sampler['weight_by']] for idx in dataset.ids]
            all_labels = [label.item() for label in all_labels] # Convert from tensor to int
            unique_labels, label_counts = np.unique(all_labels, return_counts=True)
            weights_dict = {label: 1 / count for label, count in zip(unique_labels, label_counts)} # Map every label to its inverse support
            sample_weights = [weights_dict[label] for label in all_labels]
            return torch.utils.data.WeightedRandomSampler(sample_weights, num_samples = len(dataset), replacement = True)
        else:
            raise NotImplementedError(f'Sampler type {sampler} not implemented')
        
    def get_dataloader(self, current_iter, fold, batch_size = None):
        '''
        Returns a dataloader for the dataset.
        
        Args:
            current_iter (int): Index of the current iteration
            fold (str): 'train', 'val', or 'test'
            batch_size (int): Batch size. If None, will pass all samples in a single batch. Defaults to None.
            
        Returns:
            dataloader (torch.utils.data.DataLoader): Dataloader for the dataset
        '''
        
        subset_dataset = self.get_subset(current_iter, fold)
        if subset_dataset is None:
            return None
    
        return torch.utils.data.DataLoader(subset_dataset,
                                            batch_size = len(subset_dataset) if batch_size is None else batch_size,
                                            sampler = subset_dataset.get_datasampler('random'),
                                            num_workers = 4,
                                            collate_fn = subset_dataset.collate_fn)