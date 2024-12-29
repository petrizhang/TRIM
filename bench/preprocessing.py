import argparse

import utils


def transform(source_path: str, target_path: str):
    train, test, neighbors = utils.read_hdf5_dataset(
        source_path, ["train", "test", "neighbors"]
    )

    if "angular" in source_path:
        train = utils.cos_normalize(train, None, None)
        test = utils.cos_normalize(test, None, None)

        utils.write_hdf5_dataset(target_path,
                                 {"train": train, "test": test, "neighbors": neighbors})


def generate(target_path: str):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--source_path", type=str,
                       help="The path to the source HDF5 dataset.")
    group.add_argument("-r", "--randn", action='store_true',
                       help="Generate a random dataset and write to the target path.")
    parser.add_argument("-t", "--target_path", type=str,
                        required=True, help="The path to target HDF5 dataset.")
    args = parser.parse_args()

    if args.randn:
        generate(args.target_path)
    else:
        transform(args.source_path, args.target_path)
