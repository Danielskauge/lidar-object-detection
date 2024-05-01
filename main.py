from datamodule import LidarDataModule
from models import YOLOv8Module, ResNet50
import pytorch_lightning as pl
from pathlib import Path
import munch
import yaml
import torch
from torch import nn, optim



if __name__ == "__main__":

    #pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')
    config = munch.munchify(yaml.load(open("configs/default.yaml"), Loader=yaml.FullLoader))

    data_module = LidarDataModule(
        image_dir=config.data.images_path, 
        label_dir=config.data.labels_path, 
        batch_size=config.training.batch_size,
        train_split_ratio=config.data.train_split_ratio)

    model = YOLOv8Module(config=config)

    print("Model type:", type(model))
    print("Is instance of nn.Module:", isinstance(model, nn.Module))
    print("Is instance of pl.LightningModule:", isinstance(model, pl.LightningModule))

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs, 
        enable_progress_bar=True,
        precision="bf16-mixed",
        logger=pl.loggers.WandbLogger(project=config.wandb.project, name=config.wandb.experiment, config=config),
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val/acc", patience=config.training.early_stopping_patience, mode="max", verbose=True),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelCheckpoint(dirpath=Path(config.checkpoint_folder, config.wandb.project, config.wandb.experiment), 
                            filename='best_model:epoch={epoch:02d}-val_acc={val/acc:.4f}',
                            auto_insert_metric_name=False,
                            save_weights_only=True,
                            save_top_k=1),
        ])

    trainer.fit(model, train_dataloaders=data_module.train_dataloader(),
                    val_dataloaders=data_module.val_dataloader())

    #trainer.test(model, data_module)

