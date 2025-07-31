import argparse
from aimsegdl.data_preparation.data_preparation import split_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare U-Net training data from datasets.")
    parser.add_argument('--input', type=str, default="../datasets", help='Input root folder with datasets')
    parser.add_argument('--output', type=str, default="../prepared_data", help='Output root folder')
    parser.add_argument('--create_test_split', action='store_true', help='Create test split (default: False)')
    args = parser.parse_args()

    split_dataset(args.input, args.output, create_test_split=args.create_test_split)


if __name__ == "__main__":
    main()