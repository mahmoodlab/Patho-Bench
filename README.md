# Patho-Bench

[arXiv](https://arxiv.org/pdf/2502.06750) | [HuggingFace](https://huggingface.co/datasets/MahmoodLab/Patho-Bench) | [Blog](https://www.linkedin.com/pulse/announcing-new-open-source-tools-accelerate-ai-pathology-andrew-zhang-loape/?trackingId=pDkifo54SRuJ2QeGiGcXpQ%3D%3D) | [Cite](https://github.com/mahmoodlab/patho-bench?tab=readme-ov-file#how-to-cite)
 | [License](https://github.com/mahmoodlab/patho-bench/blob/main/LICENSE)

**Patho-Bench is a Python library designed to benchmark foundation models for pathology.** 

This project was developed by the [Mahmood Lab](https://faisal.ai/) at Harvard Medical School and Brigham and Women's Hospital. This work was funded by NIH NIGMS R35GM138216.

> [!NOTE]
> Please report any issues on GitHub and contribute by opening a pull request.

## &#9889; Key features:
- **Reproducibility**: Canonical train-test splits for 95 tasks across 33 public datasets.
- **Evaluation frameworks**: Supports linear probing, prototyping (coming soon), retrieval, Cox survival prediction, and supervised fine-tuning.
- **Scalability**: Scales to thousands of experiments with automatic GPU load-balancing.

## 🚨 Updates
- **February 2025**: Patho-Bench is public!

## **Installation**:

- Create a virtual environment, e.g., `conda create -n "pathobench" python=3.10`, and activate it `conda activate pathobench`.
- `git clone https://github.com/mahmoodlab/Patho-Bench.git && cd Patho-Bench`.
- Install dependencies (including trident) `pip install -r requirements.txt`
- Local install with running `pip install -e .`.

Additional packages may be required if you are loading specific pretrained models. Follow error messages for additional instructions.

> [!NOTE]  
> Patho-Bench works with encoders implemented in [Trident](https://github.com/mahmoodlab/trident); use Trident to extract patch embeddings for your WSIs prior to running Patho-Bench.

> [!NOTE]
> Patho-Bench automatically downloads supported tasks from our [HuggingFace repo](https://huggingface.co/datasets/MahmoodLab/patho-bench). Our provided HuggingFace splits only include train and test assignments, not validation. If you want to use a validation set, you can manually reserve a portion of the training set for validation (`val`) after downloading the split. Note that some tasks have a small number of samples, which may make a validation set impractical.
> If you want to use custom splits, format them similarly to our HuggingFace splits.

## 🏃 **Running Patho-Bench**

Patho-Bench supports various evaluation frameworks:
- `linprobe`  ➡️  Linear probing (using pre-pooled features)
- `coxnet`  ➡️  Cox proportional hazards model for survival prediction (using pre-pooled features)
- `protonet`  ➡️  Prototyping (using pre-pooled features) (Coming soon!)
- `retrieval`  ➡️  Retrieval (using pre-pooled features)
- `finetune`  ➡️  Supervised finetuning or training from scratch (using patch features)

Patho-Bench can be used in two ways: 
1. **[Basic](https://github.com/mahmoodlab/patho-bench/?tab=readme-ov-file#-basic-usage-importing-and-using-patho-bench-in-your-custom-workflows):** Importable classes and functions for easy integration into custom codebases
2. **[Advanced](https://github.com/mahmoodlab/Patho-Bench/tree/main/advanced_usage):** Large-scale benchmarking using automated scripts

## 🔨 Basic Usage: Importing and using Patho-Bench in your custom workflows
Running any of the evaluation frameworks is straightforward (see example below). Define general-purpose arguments for setting up the experiment and framework-specific arguments. For a detailed introduction, follow our end-to-end [tutorial](https://github.com/mahmoodlab/Patho-Bench/blob/main/tutorial/tutorial.ipynb).

```python
import os
from patho_bench.SplitFactory import SplitFactory
from patho_bench.ExperimentFactory import ExperimentFactory

model_name = 'titan'
train_source = 'cptac_ccrcc' 
task_name = 'BAP1_mutation'

# For this task, we will automatically download the split and task config from HuggingFace.
path_to_split, path_to_task_config = SplitFactory.from_hf('./_tutorial_splits', train_source, task_name)

# Now we can run the experiment
experiment = ExperimentFactory.linprobe(
                    split = path_to_split,
                    task_config = path_to_task_config,
                    pooled_embeddings_dir = os.path.join('./_tutorial_pooled_features', model_name, train_source, 'by_case_id'), # This task uses case-level pooling
                    saveto = f'./_tutorial_linprobe/{train_source}/{task_name}/{model_name}',
                    combine_slides_per_patient = False,
                    cost = 1,
                    balanced = False,
                    patch_embeddings_dirs = '/media/ssd1/cptac_ccrcc/20x_512px_0px_overlap/features_conch_v15', # Can be omitted if pooled features are already available
                    model_name = model_name, # Can be omitted if pooled features are already available
                )
experiment.train()
experiment.test()
result = experiment.report_results(metric = 'macro-ovr-auc')
```
> [!NOTE]  
> Regarding the `combine_slides_per_patient` argument: If True, will perform early fusion by combining patches from all slides in to a single bag prior to pooling. If False, will pool each slide individually and take the mean of the slide-level features. The ideal value of this parameter depends on what pooling model you are using. For example, Titan requires this to be False because it uses spatial information (patch coordinates) during pooling. If a model doesn't use spatial information, you can usually set this to True, but it's best to consult with model documentation.

> [!NOTE]  
> Provide `patch_embeddings_dirs` so Patho-Bench knows where to find the patch embeddings for pooling. While `Trident` also supports pooling, it doesn't handle patient-level tasks with multiple slides per patient. Patho-Bench uses a generalized pooling function for multi-slide fusion. Patho-Bench requires Trident patch-level features, NOT slide-level features.

Want to do large-scale benchmarking? See instructions for [advanced usage](https://github.com/mahmoodlab/Patho-Bench/blob/main/advanced_usage/README.md).

## Funding
This work was funded by NIH NIGMS [R35GM138216](https://reporter.nih.gov/search/sWDcU5IfAUCabqoThQ26GQ/project-details/10029418).

## How to cite

If you find our work useful in your research or if you use parts of this code, please consider citing the following papers:

```
@article{zhang2025standardizing,
  title={Accelerating Data Processing and Benchmarking of AI Models for Pathology},
  author={Zhang, Andrew and Jaume, Guillaume and Vaidya, Anurag and Ding, Tong and Mahmood, Faisal},
  journal={arXiv preprint arXiv:2502.06750},
  year={2025}
}

@article{vaidya2025molecular,
  title={Molecular-driven Foundation Model for Oncologic Pathology},
  author={Vaidya, Anurag and Zhang, Andrew and Jaume, Guillaume and Song, Andrew H and Ding, Tong and Wagner, Sophia J and Lu, Ming Y and Doucet, Paul and Robertson, Harry and Almagro-Perez, Cristina and others},
  journal={arXiv preprint arXiv:2501.16652},
  year={2025}
}
```

<img src=".github/logo.png">
