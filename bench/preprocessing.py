import argparse

import utils


def run(source_path: str, target_path: str):
    train, test, neighbors = utils.read_hdf5_dataset(
        source_path, ["train", "test", "neighbors"]
    )

    if "angular" in source_path:
        train = utils.cos_normalize(train, None, None)
        test = utils.cos_normalize(test, None, None)

        utils.write_hdf5_dataset(target_path,
                                 {"train": train, "test": test, "neighbors": neighbors})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 dataset.")
    parser.add_argument("-s", "--source_path", type=str,
                        required=True, help="The path to the source HDF5 dataset.")
    parser.add_argument("-t", "--target_path", type=str,
                        required=True, help="The path to the target HDF5 dataset.")

    args = parser.parse_args()

    run(args.source_path, args.target_path)
