import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics import Accuracy
from ultralytics import YOLO
import pytorch_lightning as pl
import munch
import yaml

#fix logging
#fix loss calc

class YOLOv8Module(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.yolo_model = YOLO(self.config.model)

    def forward(self, images):
        return self.yolo_model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        results = self.forward(images)

        for r in results:
            print(r.boxes)


        # Calculate detection loss
        loss = self.yolo_model.compute_loss(results, targets)
        acc = Accuracy()(results['pred_labels'], targets['labels'])
        outputs = {'loss': loss, 'acc': acc}
        loss = outputs['loss']
        acc = outputs['acc']
        self.log({'train_loss': loss, 'train_acc': acc}, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = outputs['loss']
        self.log({'val_loss': loss, 'val_acc': acc}, on_step=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = outputs['loss']
        self.log({'test_loss': loss, 'test_acc': acc}, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.lr)
        return optimizer  



if __name__ == "__main__":
    config = munch.munchify(yaml.load(open("configs/default.yaml"), Loader=yaml.FullLoader))

    model = YOLOv8Module(config)
    results = model.forward('NAPLab-LiDAR/images/frame_000000.PNG')
    print(results)


