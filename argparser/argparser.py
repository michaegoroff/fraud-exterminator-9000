import argparse


class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--stage', '-s', help='Program stage', type=str, choices=['train', 'predict'])
        self.parser.add_argument('--image_folder', '-im', help='Name of the folder with images', type=str)
        self.parser.add_argument('--file', '-f', help='Name of the csv file', type=str)
        self.parser.add_argument('--model', '-m', help='Model being used', type=str, choices=['logreg_custom', 'svm_custom', 'r_forest', 'logreg'])
        self.parser.add_argument('--iterations', '-i', help='Number of iterations for model training', type=int)
        self.parser.add_argument('--learning_rate', '-l', help='Learning rate', type=float)
        self.args = self.parser.parse_args()

    def get_args(self):
        return self.args

