class GeneralizabilityExperimentWrapper:
    def __init__(self,
                 base_experiment,
                 external_dataset,
                 test_external_only: bool,
                 saveto: str,
                 ):
        '''
        Wrapper class for generalizability linprobe experiments.
        Performs training on internal dataset and testing on external dataset.
        
        Args:
            base_experiment (Experiment): Experiment object to be used for training.
            external_dataset (Dataset): External dataset to test on.
            test_external_only (bool): If True, train on entirety of internal dataset and test on external dataset.
                                       If False, train on training split of internal dataset and test first on internal dataset, then on external dataset.
                                       Defaults to True.
            saveto (str): Save external generalizability results to this directory.
        '''
        self.exp = base_experiment
        self.external_dataset = external_dataset
        self.test_external_only = test_external_only
        self.saveto = saveto
        
        if test_external_only:
            print(f"\033[96mRunning generalizability experiment with test_external_only = True...\033[0m")
        else:
            print(f"\033[96mRunning generalizability experiment with test_external_only = False...\033[0m")
            import warnings; warnings.warn("test_external_only = False is not yet supported by Agent.collect_results(). You may have to collect the results manually at the end of an Agent sweep.")
        
    def train(self):
        if self.test_external_only:
            print(f"\033[93mAssigning all samples to a single train fold.\033[0m")
            self.exp.dataset.split.remove_all_folds()
            self.exp.dataset.split.assign_folds(num_folds=1, test_frac=0, val_frac=0, method='monte-carlo')  # Assign all samples to a single train fold
            self.exp.dataset.num_folds = 1
            self.exp.results_dir = self.saveto # Save training artifacts directly to external directory
        self.exp.train()

    def test(self):
        if not self.test_external_only:
            self.exp.test() # Test on internal dataset first
            
        self.exp.dataset = self.external_dataset         # Switch to external dataset
        self.exp.results_dir = self.saveto # Save testing artifacts to external directory
        self.exp.test() # Test on external dataset
        
    def report_results(self, metric: str):
        '''
        Report results of experiment. Calls BaseExperiment.report_results().        
        
        Args:
            metric (str): Metric to report. Must be implemented in self.classification_metrics()
        '''
        return self.exp.report_results(metric)