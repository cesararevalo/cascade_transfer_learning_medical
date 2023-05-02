"""
Script used for converting the results from the final training run because...
"""
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis

ea = ExperimentAnalysis(experiment_checkpoint_path="final_results_2023-05-01/experiment_state-2023-04-30_16-58-55.json")

ea.results_df.to_csv('final_results_2023-05-01.csv')
