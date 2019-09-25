import argparse
from azureml.core import Workspace, Dataset, Experiment, Run
# local imports
from score import run_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    args = parser.parse_args()

    run = Run.get_context()
    print(run_score(args.input_path, args.output_path))

    run.complete()

