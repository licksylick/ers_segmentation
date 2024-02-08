import os
from argparse import ArgumentParser
import importlib
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from utils import preprocess_config, save_config
from augs import Augs
from model import DroneModel
from torch.utils.data import DataLoader
from dataset import SegmentationMulticlassDataset


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()

    config = preprocess_config(OmegaConf.load(args.config))
    save_config(config)
    print(OmegaConf.to_yaml(config))

    seed_everything(config['common']['seed'], workers=True)

    transforms = Augs(config['dataset']['img_size'])
    train_transform, val_test_transform = transforms.train_augs(), transforms.val_test_augs()

    train_dataset = SegmentationMulticlassDataset('train.csv', is_train=True, augs=train_transform,
                                                  h_w=config['dataset']['img_size'],
                                                  num_classes=config['model']['params']['classes'])
    val_dataset = SegmentationMulticlassDataset('val.csv', augs=val_test_transform,
                                                h_w=config['dataset']['img_size'],
                                                num_classes=config['model']['params']['classes'])
    test_dataset = SegmentationMulticlassDataset('test.csv', augs=val_test_transform,
                                                 h_w=config['dataset']['img_size'],
                                                 num_classes=config['model']['params']['classes'])

    train_dataloader = DataLoader(train_dataset, batch_size=config['common']['batch_size'], shuffle=True,
                                  num_workers=config['common']['num_workers'])
    val_dataloader = DataLoader(val_dataset, batch_size=config['common']['batch_size'], shuffle=False,
                                num_workers=config['common']['num_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['common']['batch_size'], shuffle=False,
                                 num_workers=config['common']['num_workers'])


    model = DroneModel(config)

    print(config['trainer']['params'])

    if torch.cuda.device_count() > 0:
        trainer = pl.Trainer(default_root_dir=config['common']['exp_name'], max_epochs=config['common']['max_epochs'],
                             log_every_n_steps=10, devices=config['trainer']['params']['gpus'], accelerator='gpu')
    else:
        trainer = pl.Trainer(default_root_dir=config['common']['exp_name'], max_epochs=config['common']['max_epochs'],
                             log_every_n_steps=10, accelerator='cpu')

    trainer.callbacks += [getattr(importlib.import_module(callback_config.target.rsplit('.', 1)[0]),
                                      callback_config.target.rsplit('.', 1)[1])(**callback_config.params) for
                              callback_config in config.callbacks]

    trainer.fit(model, train_dataloader, val_dataloader)

    trainer.test(model, test_dataloader)
