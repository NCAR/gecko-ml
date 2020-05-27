import numpy as np
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", required=True, help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.load(config_file)
    # We will want to run multiple GECKO experiments in parallel
    return


if __name__ == "__main__":
    main()