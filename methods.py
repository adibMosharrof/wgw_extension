import pytorch_lightning as pl
from torch import nn, optim
import torch
from torchvision import models, transforms
import torchmetrics as metrics
from torch.distributions.poisson import Poisson

CUDA_LAUNCH_BLOCKING=1

class WgwModel(pl.LightningModule):

    def __init__(self, num_classes=91):
        super().__init__()
        self.save_hyperparameters()
        self.aerial_model = models.resnet18(pretrained=True)
        self.aerial_model.fc = nn.Linear(512, num_classes)
        self.train_metrics = metrics.Accuracy()        
        self.test_metrics = metrics.Accuracy()        
        self.val_metrics = metrics.Accuracy()        

    def forward(self, data):
        aerial_feats = self.aerial_model(data['aerial_img'])
        aerial_feats = nn.functional.softplus(aerial_feats)
        return aerial_feats

    def get_loss(self, logits, labels, step='train', metrics=None):
        m = Poisson(logits)
        output = m.sample().int()
        lb = -1 * m.log_prob(labels)
        loss = lb.mean()
        self.log(f'{step}_loss', loss, prog_bar=True)
        self.log(f'{step}_accuracy', metrics(output, labels), prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step='train', metrics=self.train_metrics)

    def validation_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step='val', metrics=self.val_metrics)

    def test_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step='test', metrics=self.test_metrics)

    def _shared_step(self, batch, batch_idx=None, step='train', metrics=None):
        logits = self(batch)
        labels = batch['labels_counts']
        loss = self.get_loss(logits, labels, step=step, metrics=metrics)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-5)

class NlcdBaselineModel(pl.LightningModule):

    def __init__(self, num_classes=8):
        super().__init__()
        self.save_hyperparameters()
        self.aerial_model = models.resnet18(pretrained=True)
        self.aerial_model.fc = nn.Linear(512, num_classes)
        self.train_metrics = metrics.Accuracy()        
        self.test_metrics = metrics.Accuracy()        
        self.val_metrics = metrics.Accuracy()        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data):
        aerial_feats = self.aerial_model(data['aerial_img'])
        aerial_feats = nn.functional.normalize(aerial_feats) # shape = (Batch, Feature)
        data['aerial_feats'] = aerial_feats
        return data

    def get_loss(self, feats, labels, step='train', metric=None):
        loss = self.criterion(feats, labels)
        return loss

    def training_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step='train', metric=self.train_metrics)

    def validation_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step='val', metric=self.val_metrics)

    def test_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step='test', metric=self.test_metrics)

    def _shared_step(self, batch, batch_idx=None, step='train', metric=None):
        logits = self(batch)
        labels = logits['nlcd_coarse_labels']
        feats = logits['aerial_feats']
        loss = self.get_loss(feats, labels, step=step, metric=metric)
        preds = torch.argmax(feats, dim=1)
        accuracy = metric(preds, labels).detach()  
        self.log(f"{step}/acc", accuracy)
        self.log(f"{step}/loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-5)

class NlcdPretrainedModel(NlcdBaselineModel):
    def __init__(self, num_classes=8, checkpoint_path=None):
        super().__init__()
        self.save_hyperparameters()
        checkpoint = torch.load(checkpoint_path)
        wgw_model = WgwModel(num_classes=num_classes)
        wgw_model.load_state_dict(checkpoint['state_dict'])
        self.aerial_model = wgw_model.aerial_model
        self.aerial_model.fc = nn.Linear(512, num_classes)