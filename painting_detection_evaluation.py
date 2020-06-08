import os

from estensi.painting_detection.evaluation import get_test_set_dict, create_test_set, eval_test_set, hyperparameters_gridsearch


def main():
    path = os.path.abspath(os.path.dirname(__file__))

    # Create test set from selected videos
    dataset_dir_path = os.path.join(path, "dataset")
    test_set_dict = get_test_set_dict(dataset_dir_path=dataset_dir_path)
    test_set_dir_path = os.path.join(path, "dataset", "test_set")
    create_test_set(test_set_dict=test_set_dict, test_set_dir_path=test_set_dir_path)

    # Evaluate painting detection on test set
    ground_truth_set_dir_path = os.path.join(path, "dataset", "ground_truth")
    f1_score, avg_precision, avg_recall = eval_test_set(test_set_dir_path=test_set_dir_path,
                                                        ground_truth_set_dir_path=ground_truth_set_dir_path,
                                                        verbose=True)
    print("f1 = {:.2f}, p = {:.2f}, r = {:.2f}".format(f1_score, avg_precision, avg_recall))

    # Evaluate painting detection with specific hyperparameters
    param_grid = {
        "MIN_ROTATED_BOX_AREA_PERCENT": [0.5, 0.8, 0.9],
        "MIN_ROTATED_ELLIPSE_AREA_PERCENT": [0.4, 0.6],
        "MAX_GRAY_60_PERCENTILE": [170, 200],
        "MIN_VARIANCE": [11, 18],
        "MIN_HULL_AREA_PERCENT_OF_MAX_HULL": [0.08, 0.15],
        "THRESHOLD_BLOCK_SIZE_FACTOR": [50, 80]
    }
    hyperparameters_gridsearch(test_set_dir_path=test_set_dir_path,
                               ground_truth_set_dir_path=ground_truth_set_dir_path,
                               param_grid=param_grid)


if __name__ == '__main__':
    main()
