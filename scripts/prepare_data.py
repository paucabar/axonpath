import argparse
import os
import shutil
from axonpath.data_preparation.data_preparation import split_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare U-Net training data from datasets.")
    parser.add_argument('--input', type=str, default="../datasets", help='Input root folder with datasets')
    parser.add_argument('--output', type=str, default="../prepared_data", help='Output root folder')
    parser.add_argument('--create_test_split', action='store_true', help='Create test split (default: False)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible train/val/test splits')
    parser.add_argument('--overwrite', action='store_true',
                        help='Delete all existing output and reprocess everything from scratch')
    args = parser.parse_args()

    if args.overwrite and os.path.exists(args.output):
        print(f"--overwrite: deleting {args.output}")
        shutil.rmtree(args.output)

    split_dataset(args.input, args.output, create_test_split=args.create_test_split, seed=args.seed)


if __name__ == "__main__":
    main()