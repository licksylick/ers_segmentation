from utils import prepare_stratified_train_val_test_csv
from omegaconf import OmegaConf
from argparse import ArgumentParser
from utils import preprocess_config

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = preprocess_config(OmegaConf.load(args.config))
    input_path = config['dataset']['folder']

    prepare_stratified_train_val_test_csv(input_path)
    print('Created train.csv, val.csv, test.csv')
    print('Now you can run train_val_test.py')


