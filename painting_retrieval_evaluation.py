import argparse
import os

from estensi.painting_detection.evaluation import get_test_set_dict, create_test_set
from estensi.painting_retrieval.evaluation import eval_test_set


def arg_parse():
    parser = argparse.ArgumentParser(description='Vision and Cognitive Systems project: Gallerie Estensi')
    parser.add_argument('--mode', dest='mode', help='select between classification or retrieval evaluation mode',
                        type=str, choices=['classification', 'retrieval'], required=True)
    parser.add_argument('--rank_scope', dest='rank_scope',
                        help='scope of the ranking list where a relevant item can be found. '
                             'It will be ignored in classification mode.',
                        type=int, choices=range(1, 96), required=False)
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
    db_dir_path = os.path.join(path, "dataset", "paintings_db")

    # Evaluate painting retrieval on test set
    mode = args.mode
    rank_scope = args.rank_scope
    if rank_scope is None:
        rank_scope = 5
    mean_avg_precision = eval_test_set(test_set_dir_path=test_set_dir_path,
                                       ground_truth_set_dir_path=ground_truth_set_dir_path,
                                       db_dir_path=db_dir_path,
                                       files_dir_path=dataset_dir_path,
                                       mode=mode,
                                       rank_scope=rank_scope,
                                       verbose=True)
    print("map = {:.2f}".format(mean_avg_precision))


if __name__ == '__main__':
    main()
