import argparse
from azureml.core import Workspace, Dataset, Experiment, Run
from azureml.core.model import Model
# local imports
from score import run_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    args = parser.parse_args()

    run = Run.get_context()

    model_name = 'diabetes_model'
    if 'offline' in run.id.lower():
        model_path = 'models\model_alpha_best.pkl'
    else:
        model_path = Model.get_model_path(model_name=model_name)

    results_path = run_score(args.input_path, args.output_path, model_path, run.id)
    run.upload_file(name='score_results.csv', path_or_stream=results_path)

    run.complete()

