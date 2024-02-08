import torch
import pytorch_lightning as pl
from utils import instantiate_from_config
import segmentation_models_pytorch as smp


class DroneModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        model_params = config['model']['params']
        self.model = getattr(smp, config['model']['target'])(**model_params)
        self.criterions = {x['name']: instantiate_from_config(x) for x in config['criterions']}
        self.crit_weights = {x['name']: x['weight'] for x in config['criterions']}
        self.config = config

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage):
        image, mask = batch['image'], batch['sg_mask']
        out = self.forward(image.contiguous())
        
        loss = 0
        for c_name in self.criterions.keys():
            c_loss = self.criterions[c_name](out, mask) * self.crit_weights[c_name]
            self.log(f"{c_name}_loss_{stage}", c_loss, on_step=False, on_epoch=True, prog_bar=True)
            loss += c_loss

        tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(out, 1), mask.long(), mode='multiclass',
                                               num_classes=self.config['model']['params']['classes'])
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        self.log(f"{stage}_iou", iou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        return {f"loss": loss, f"iou": iou}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer_name = self.config['optimizers'][0]['target']
        optimizer_params = self.config['optimizers'][0]['params']
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(self.parameters(), **optimizer_params)
        scheduler_name = self.config['scheduler'][0]['target']
        scheduler_params = getattr(torch.optim.lr_scheduler, scheduler_name)
        scheduler = scheduler_params(optimizer, self.config['scheduler'][0]['params']['T_max'],
                                     self.config['scheduler'][0]['params']['eta_min'],
                                     self.config['scheduler'][0]['params']['last_epoch'],
                                     self.config['common']['max_epochs'])
        monitor = self.config['scheduler'][0].get('monitor', '')
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}