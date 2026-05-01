import argparse
import os
import shutil
import pandas as pd
from axonpath.data_preparation.data_preparation import split_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare U-Net training data from datasets.")
    parser.add_argument('--input', type=str, default="../datasets", help='Input root folder with datasets')
    parser.add_argument('--output', type=str, default="../prepared_data", help='Output root folder')
    parser.add_argument('--group', type=str, default=None,
                        help='Training group to include (e.g. EM, BF). Reads training_group '
                             'from datasets.csv in <input>/../scripts/. If omitted, all datasets are processed.')
    parser.add_argument('--create_test_split', action='store_true', help='Create test split (default: False)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible train/val/test splits')
    parser.add_argument('--overwrite', action='store_true',
                        help='Delete all existing output and reprocess everything from scratch')
    args = parser.parse_args()

    allowed_datasets = None
    if args.group is not None:
        csv_path = os.path.join(
            os.path.dirname(os.path.normpath(os.path.abspath(args.input))),
            "scripts", "datasets.csv",
        )
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"datasets.csv not found at {csv_path}. "
                "Pass a correct --input path or place datasets.csv at <input>/../scripts/datasets.csv."
            )
        df = pd.read_csv(csv_path)
        allowed_datasets = set(df.loc[df["training_group"] == args.group, "new_name"])
        if not allowed_datasets:
            raise ValueError(f"No datasets found for training_group='{args.group}' in {csv_path}.")
        print(f"Group '{args.group}': {sorted(allowed_datasets)}")

    if args.overwrite and os.path.exists(args.output):
        print(f"--overwrite: deleting {args.output}")
        shutil.rmtree(args.output)

    split_dataset(
        args.input, args.output,
        create_test_split=args.create_test_split,
        seed=args.seed,
        allowed_datasets=allowed_datasets,
    )


if __name__ == "__main__":
    main()
