import numpy as np
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", required=True, help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.load(config_file)
    # Extract config arguments and validate if necessary

    # Load GECKO experiment data, split into ML inputs and outputs and persistence outputs

    # Split into training, validation, testing subsets

    # Rescale training and validation / testing data

    # Train ML models

    # Calculate validation and testing scores

    # Save ML models, scaling values, and verification data to disk
    return


if __name__ == "__main__":
    main()