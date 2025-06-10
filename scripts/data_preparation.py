import argparse
from aimsegdl.data_preparation.data_preparation import create_train_val_test_split_all


def main():
    parser = argparse.ArgumentParser(description="Prepare U-Net training data from datasets.")
    parser.add_argument('--input', type=str, default="../datasets", help='Input root folder with datasets')
    parser.add_argument('--output', type=str, default="../prepared_data", help='Output root folder')
    parser.add_argument('--fix_label_padding', action='store_true', default=True, help='Fix extra padded label rows/cols')
    args = parser.parse_args()

    create_train_val_test_split_all(args.input, args.output, fix_label_padding=args.fix_label_padding)


if __name__ == "__main__":
    main()