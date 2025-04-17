import os
from tqdm import tqdm
import h5py
import torch
from trident.slide_encoder_models.load import encoder_factory

"""
Pools patch features given a pretrained slide encoder and saves pooled features to disk.
Used by DatasetFactory to pool features for slide-level or patient-level datasets.
"""

class Pooler:
    def __init__(self, patch_embeddings_dataset, model_name, model_kwargs, save_path, device):
        
        # Load patch features from split
        self.dataset = patch_embeddings_dataset
        
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.device = device
        
        self.model = None
        
    def _load_model(self):
        '''
        Load the frozen slide encoder and save its architecture.
        '''
        self.model = encoder_factory(self.model_name, freeze=True, **self.model_kwargs)
        # Save model architecture
        with open(os.path.join(self.save_path, '_model.txt'), 'w') as f:
            f.write(repr(self.model))
            f.write(f'\nTotal number of parameters: {sum(p.numel() for p in self.model.parameters())}')
            f.write(f'\nNumber of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
    
    @torch.inference_mode()
    def run(self):
        '''
        Loop through the dataset and pool features using the specified model.
        '''
        loop = tqdm(self.dataset.ids, desc="Pre-pooling features")
        for sample_id in loop:
            
            # Skip if already pooled
            if os.path.exists(os.path.join(self.save_path, f"{sample_id}.h5")):
                loop.set_postfix_str(f"Sample {sample_id} already pooled, skipping...")
                continue
            
            # Note that here a sample consists of patch features from a case or slide, depending on the dataset
            sample = self.dataset[sample_id]            
            if sample['id'] is None:
                continue # Skip because could not load the patch features for this sample
            
            # Load model if not already loaded
            if self.model is None:
                self._load_model()
            
            # Try running on GPU, if out of memory, retry on CPU
            try:
                cleaned_sample = self.prepare_slide_encoder_input_batch(self.dataset.collate_fn([sample])) # Collate and clean for forward pass
                loop.set_postfix_str(f"Running on GPU {self.device}...")
                if isinstance(self.model, torch.nn.Module):
                    self.model = self.model.to(f'cuda:{self.device}')
                with torch.amp.autocast('cuda', dtype = self.model.precision, enabled = self.model.precision in [torch.bfloat16, torch.float16]):
                    pooled_feature = self.pool(self.model, cleaned_sample, f'cuda:{self.device}')
                    
                # Save as h5
                with h5py.File(os.path.join(self.save_path, f"{sample['id']}.h5"), 'w') as f:
                    f.create_dataset('features', data=pooled_feature.float().cpu().numpy())
                    
            except Exception as e:
                if "out of memory" in str(e).lower():
                    loop.set_postfix_str("Out of memory on GPU, retrying on CPU...")
                    self.model = self.model.to('cpu')
                    with torch.amp.autocast('cpu', dtype = self.model.precision, enabled = self.model.precision in [torch.bfloat16, torch.float16]):
                        pooled_feature = self.pool(self.model, cleaned_sample, 'cpu')
                        
                    # Save as h5
                    with h5py.File(os.path.join(self.save_path, f"{sample['id']}.h5"), 'w') as f:
                        f.create_dataset('features', data=pooled_feature.float().cpu().numpy())
                        
                else:
                    print(f'\033[31mError processing patch feats for {sample_id}\033[0m')
                    # raise Exception(f"Error processing patch feats for {sample_id}: {e}")
    
    @staticmethod
    def prepare_slide_encoder_input_batch(sample_collated):
        '''
        Clean up the batch for compatibility with Trident slide encoders forward pass.
        All slide encoders loaded from Trident require input batches with a specific format.
        See here for more information: https://github.com/mahmoodlab/trident/blob/main/trident/slide_encoder_models/load.py
        
        Args:
            sample_collated (dict): Sample collated by the dataset collate function
            
        Returns:
            cleaned_batch (dict or list[dict]): Cleaned up slide sample or list of slide samples
        '''
        assert len(sample_collated['id']) == 1, "Batch size must be 1 to be compatible with Trident slide encoders."
        
        if isinstance(sample_collated['features'], torch.Tensor): # combine_slides_per_patient was set to True when initializing the dataset
            cleaned_sample = {
                'features': sample_collated['features'],
                'coords': sample_collated['coords'],
                'attributes': {'patch_size_level0': sample_collated['attributes']['patch_size_level0'][0]} # BaseDataset.collate_fn() collates the attributes as well so we need to get the first (only) element
            }
        elif isinstance(sample_collated['features'], list): # combine_slides_per_patient was set to False when initializing the dataset
            cleaned_sample = []
            for features, coords, attributes in zip(sample_collated['features'], sample_collated['coords'], sample_collated['attributes']):
                cleaned_sample.append({
                    'features': features,
                    'coords': coords,
                    'attributes': {'patch_size_level0': attributes['patch_size_level0'][0]} # BaseDataset.collate_fn() collates the attributes as well so we need to get the first (only) element
                })
        return cleaned_sample
    
    @staticmethod
    def pool(model, cleaned_sample, device):
        '''
        Get pooled features for a sample.
        
        Args:
            model (torch.nn.Module): Slide encoder model (must be compatible with Trident API)
            sample_collated (dict or list[dict]): Sample or list of cleaned up samples
            device (str): Device to run on ('cuda' or 'cpu')
        '''
        if isinstance(cleaned_sample, list):
            pooled_feature = []
            for batch in cleaned_sample:
                pooled_feature.append(model.forward(batch, device))
            return torch.stack(pooled_feature).mean(dim=0)
        else:
            return model.forward(cleaned_sample, device)