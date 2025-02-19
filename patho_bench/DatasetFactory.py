import os
import torch
from einops import rearrange
try:
    from einops._torch_specific import allow_ops_in_compiled_graph; allow_ops_in_compiled_graph()  # https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops
finally:
    pass
from patho_bench.datasets.PatchEmbeddingsDataset import PatchEmbeddingsDataset
from patho_bench.datasets.SlideEmbeddingsDataset import SlideEmbeddingsDataset
from patho_bench.datasets.LabelDataset import LabelDataset
from patho_bench.datasets.CombinedDataset import CombinedDataset
from patho_bench.Pooler import Pooler
from patho_bench.helpers.GPUManager import GPUManager

"""
This file contains the DatasetFactory class which is responsible for creating different types of datasets.
"""

class DatasetFactory:
    
    @staticmethod
    def from_patch_embeddings(**kwargs):
        '''
        Creates a dataset that returns patch-level embeddings and labels.
        '''
        return CombinedDataset({
            'slide': DatasetFactory._patch_embeddings_dataset(**kwargs),
            'labels': DatasetFactory._labels_dataset(kwargs['split'], kwargs['task'])
        })
        
    @staticmethod
    def from_slide_embeddings(**kwargs):
        '''
        Creates a dataset that returns slide-level embeddings and labels.
        '''
        return CombinedDataset({
            'slide': DatasetFactory._slide_embeddings_dataset(**kwargs),
            'labels': DatasetFactory._labels_dataset(kwargs['split'], kwargs['task'])
        })

    @staticmethod
    def _slide_embeddings_dataset(split, pooled_embeddings_dir = None, source = None, patch_embeddings_dirs = None, pooled_embeddings_root = None, pooling_level = None, combine_slides_per_patient = None, model_name = None, gpu = -1, **kwargs):
        '''
        Creates a dataset that loads pooled (slide-level or patient-level) features.
        If the pooled features do not exist, they are created from patch-level features and saved to the provided directory.
        
        Args:
            split (Split): Split object
            pooled_embeddings_dir (str): Path to directory containing pooled embeddings. If None, pooled embeddings are created from patch embeddings.
            source (str): Name of the data source
            patch_embeddings_dirs (list): List of directories containing patch embeddings
            pooled_embeddings_root (str): Path to root folder where pooled embeddings are saved. Subdirectories are automatically created for each datasource and pooling type.
            pooling_level (str): Level of pooling ('case_id' or 'slide_id')
            combine_slides_per_patient (bool): Whether to combine patches from multiple slides when pooling at case_id level. If False, will pool each slide independently and take mean (late fusion).
            model_name (str): Name of the model
            gpu (int): GPU to use for pooling. If -1, the best available GPU is used.
        '''
        
        if not pooled_embeddings_dir:
            # If pooled_embeddings_dir is not provided, must prepare pooled features from patch features (this will skip over slides that have already been pooled)
            pooled_embeddings_dir = os.path.join(pooled_embeddings_root, f'by_{pooling_level}', model_name, source)
            print('\033[94m' + f'Saving {pooling_level}-level features to {pooled_embeddings_dir}, using {model_name}...' + '\033[0m')
            pooler = Pooler(patch_embeddings_dataset = DatasetFactory._patch_embeddings_dataset(split, patch_embeddings_dirs, combine_slides_per_patient, bag_size = None),
                                    model_name = model_name,
                                    save_path = pooled_embeddings_dir,
                                    device = GPUManager.get_best_gpu(min_mb=500) if gpu == -1 else gpu)
            pooler.run()
            del pooler
            torch.cuda.empty_cache()
        
        return SlideEmbeddingsDataset(split, load_from = pooled_embeddings_dir)
    
    @staticmethod
    def _patch_embeddings_dataset(split, patch_embeddings_dirs, combine_slides_per_patient, bag_size = None, **kwargs):
        '''
        Creates a dataset that loads patch-level features.
        
        Args:
            split (Split): Split object
            patch_embeddings_dirs (list): List of directories containing patch embeddings
            combine_slides_per_patient (bool): Whether to combine patches from multiple slides when pooling at case_id level. If False, will pool each slide independently and take mean (late fusion).
            bag_size (int): Number of patches to sample. If None, all patches are loaded (caution, this may use a lot of memory).
        '''
        if isinstance(patch_embeddings_dirs, str):
            patch_embeddings_dirs = [patch_embeddings_dirs]
            
        return PatchEmbeddingsDataset(split,
                                      load_from = list(set(patch_embeddings_dirs)),
                                    #   preprocessor= {'features': lambda x: rearrange(x, "1 p f -> p f"),
                                    #                 'coords': lambda x: rearrange(x, "1 p c -> p c")},
                                      bag_size = bag_size,
                                      shuffle = False,
                                      pad = False,
                                      combine_slides_per_patient = combine_slides_per_patient
                                    )
    
    @staticmethod
    def _labels_dataset(split, task):
        '''
        Creates a dataset that loads sample labels.
        
        Args:
            split (Split): Split object
            task (str): Name of the task
        '''
        if task == 'OS': # Overall survival
            return LabelDataset(split, task_names = ["OS"], extra_attrs = ["OS_event", "OS_days"], dtype = 'int')
        elif task == 'PFS': # Progression-free survival
            return LabelDataset(split, task_names = ["PFS"], extra_attrs = ["PFS_event", "PFS_days"], dtype = 'int')
        elif task == 'DSS': # Disease-specific survival
            return LabelDataset(split, task_names = ["DSS"], extra_attrs = ["DSS_event", "DSS_days"], dtype = 'int')
        else:
            return LabelDataset(split, task_names = [task], dtype = 'int')