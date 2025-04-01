import os
import torch
import json
import numpy as np
from patho_bench.config.ConfigMixin import ConfigMixin

# Monkey-patching shenanigans; DO NOT TOUCH
try:
    import lovely_tensors; lovely_tensors.monkey_patch()
except ImportError:
    print("WARNING: Failed to import lovely_tensors. Please run <pip install lovely-tensors> if you want lovely-tensors (useful for debugging).")


"""
This is a base Experiment class that has common methods for initializing datasets and 
"""

class BaseExperiment(ConfigMixin):
    def __init__(self):
        pass
        
    def bootstrap(self, all_labels_across_folds, all_preds_across_folds, num_bootstraps=100):
        '''
        Perform bootstrapping and calculate 95% CI.

        Args:
            all_labels_across_folds (list): List of labels across folds.
            all_preds_across_folds (list): List of predictions across folds.

        Returns:
            bootstraps (list[tuple]:) List of bootstrapped (labels, preds) tuples.
        '''
        all_labels_across_folds, all_preds_across_folds = np.concatenate(all_labels_across_folds), np.concatenate(all_preds_across_folds)
        bootstraps = []
        for _ in range(num_bootstraps):
            idx = np.random.choice(len(all_labels_across_folds), len(all_labels_across_folds), replace=True)
            labels = np.array([all_labels_across_folds[i] for i in idx])
            preds = np.array([all_preds_across_folds[i] for i in idx])
            bootstraps.append((labels, preds))
        return bootstraps

    def get_95_ci(self, all_scores_across_folds):
        '''
        Calculate 95% CI.

        Args:
            all_scores_across_folds (list[dict]): List of scores across bootstraps, keys are metric names.
            
        Returns:
            mean (float): Mean of bootstrapped scores.
            lower (float): Lower bound of 95% CI.
            upper (float): Upper bound of 95% CI.
        '''
        summary = {}
        for key in all_scores_across_folds[0].keys():
            if any([score[key] is None for score in all_scores_across_folds]): # Sometimes scores are mathematically noncomputable, indicated by None
                summary[key] = {
                    'mean': None,
                    'lower': None,
                    'upper': None,
                    'formatted': 'N/A'
                }
                continue
            mean = np.mean([score[key] for score in all_scores_across_folds])
            lower = np.percentile([score[key] for score in all_scores_across_folds], 2.5)
            upper = np.percentile([score[key] for score in all_scores_across_folds], 97.5)
            summary[key] = {
                'mean': mean,
                'lower': lower,
                'upper': upper,
                'formatted': f'{mean:.3f} ({lower:.3f}-{upper:.3f})'
            }

        return summary
        
    def get_mean_se(self, all_scores_across_folds):
        '''
        Calculate mean ± SE.

        Args:
            all_scores_across_folds (list): List of scores across folds.

        Returns:
            mean (float): Mean of scores.
            se (float): Standard error of scores.
        '''
        summary = {}
        for key in all_scores_across_folds[0].keys():
            mean = np.mean([score[key] for score in all_scores_across_folds if score[key] is not None])
            std = np.std([score[key] for score in all_scores_across_folds if score[key] is not None])
            se = std / np.sqrt(len(all_scores_across_folds))
            summary[key] = {
                'mean': mean,
                # 'std': std,
                'se': se,
                'formatted': f'{mean:.3f} ± {se:.3f}'
            }

        return summary
    
    def set_seed(self, seed = 0, disable_cudnn = False):
        '''
        Sets global seeds for reproducibility.
        Seeds the RNG for all devices (both CPU and CUDA).
        '''
        import random
        from numpy.random import MT19937, RandomState, SeedSequence
        
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        rs = RandomState(MT19937(SeedSequence(seed)))  # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        
        if not disable_cudnn:
            # Causes cuDNN to deterministically select an algorithm,
            torch.backends.cudnn.benchmark = False
            # possibly at the cost of reduced performance
            # (the algorithm itself may be nondeterministic).
            # Causes cuDNN to use a deterministic convolution algorithm,
            torch.backends.cudnn.deterministic = True
            # but may slow down performance.
            # It will not guarantee that your training process is deterministic
            # if you are using other libraries that may use nondeterministic algorithms
        else:
            # Controls whether cuDNN is enabled or not.
            torch.backends.cudnn.enabled = False
            # If you want to enable cuDNN, set it to True.
            
    def report_results(self, metric: str, mode: str = 'test'):
        '''
        Report results of experiment
        
        Args:
            metric (str): Metric to report.
            mode (str): Either 'test'  or 'val'. Defaults to 'test', as sometimes validation set is not defined.
        '''
        with open(os.path.join(self.results_dir, f'{mode}_metrics_summary.json'), 'r') as f:
            results = json.load(f)

        print(f"{metric}: {results[metric]['formatted']}")
        
        return results[metric]['mean']