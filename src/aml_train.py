import argparse
from azureml.core import Workspace, Dataset, Experiment, Run
from azureml.core.model import Model
# local imports
from train import run_train
import config as cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    args = parser.parse_args()
    run = Run.get_context()
    

    best_model_path, best_rmse, best_alpha = run_train(args.input_path, args.output_path)

    run.log('rmse', best_rmse)
    run.log('alpha', best_alpha)

    run.upload_file(name='diabetes_model.pkl', path_or_stream=best_model_path)
    # TODO register model if better than previous models?
    if not 'offline' in run.id.lower():
        run.register_model(cfg.model_name, best_model_path)
    run.complete()

