import argparse
import json
import os

from estensi.painting_detection.evaluation import get_test_set_dict, create_test_set, eval_test_set, hyperparameters_gridsearch


def arg_parse():
    parser = argparse.ArgumentParser(description='Vision and Cognitive Systems project: Gallerie Estensi')
    parser.add_argument('--param', dest='param', help='path of the param_grid json file', type=str, required=False)
    return parser.parse_args()


def main():
    args = arg_parse()
    path = os.path.abspath(os.path.dirname(__file__))

    # Create test set from selected videos
    dataset_dir_path = os.path.join(path, "dataset")
    test_set_dict = get_test_set_dict(dataset_dir_path=dataset_dir_path)
    test_set_dir_path = os.path.join(path, "dataset", "test_set")
    create_test_set(test_set_dict=test_set_dict, test_set_dir_path=test_set_dir_path)
    ground_truth_set_dir_path = os.path.join(path, "dataset", "ground_truth")

    if args.param is None:
        # Evaluate painting detection on test set
        f1_score, avg_precision, avg_recall = eval_test_set(test_set_dir_path=test_set_dir_path,
                                                            ground_truth_set_dir_path=ground_truth_set_dir_path,
                                                            verbose=True)
        print("f1 = {:.2f}, p = {:.2f}, r = {:.2f}".format(f1_score, avg_precision, avg_recall))
    else:
        # Evaluate painting detection with specific hyperparameters
        file = open(args.param, "r")
        param_grid = json.load(file)
        hyperparameters_gridsearch(test_set_dir_path=test_set_dir_path,
                                   ground_truth_set_dir_path=ground_truth_set_dir_path,
                                   param_grid=param_grid)
        file.close()


if __name__ == '__main__':
    main()
