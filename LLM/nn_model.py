import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
from torch_optimizer import Lamb

from chateval.results import evaluate_metric
import pytorch_lightning as pl

from torchmetrics.functional import log_cosh_error


class LogCoshLoss(nn.Module):
    def __call__(self, preds, target):
        return log_cosh_error(preds, target).sum()


class RegressionModule(pl.LightningModule):
    """
    Train the models with regression loss (MSE by default)
    """

    def __init__(self, args, input_size):
        super().__init__()

        self.args = args

        self.loss = LogCoshLoss()
        layers = [nn.Linear(input_size, args.hid_dim), nn.GELU()]
        layers.extend(
            [nn.Linear(args.hid_dim, args.hid_dim), nn.GELU()] * args.hid_layers
        )
        last_dim = input_size if args.hid_layers == 0 else args.hid_dim
        layers.append(nn.Linear(last_dim, 1))
        self.fnn = nn.Sequential(*layers)
        self.save_hyperparameters()

    @property
    def quality(self):
        return self.args.quality

    def _forward(self, batch):
        x = batch["x"]
        outputs = self.fnn(x)
        return outputs.reshape(x.shape[0])

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return {self.quality: self._forward(batch), "dialogue_id": batch["dialogue_id"]}

    def training_step(self, batch, batch_idx):
        outputs = self._forward(batch)
        loss = self.loss(outputs, batch[f"annotations.{self.quality}"])
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self._forward(batch)
        loss = self.loss(outputs, batch[f"annotations.{self.quality}"])
        return {"loss": loss, self.quality: outputs, **batch}

    def validation_epoch_end(self, outputs):
        pred_quality = torch.cat([o[self.quality] for o in outputs])
        gold_quality = torch.cat([o[f"annotations.{self.quality}"] for o in outputs])
        pcc, spcc = evaluate_metric(
            "nn_op",
            {self.quality: pred_quality.cpu().numpy()},
            {f"annotations.{self.quality}": gold_quality.cpu().numpy()},
            metric_qualities=[self.quality],
        )
        self.log(f"SRCC-{self.quality}", spcc)
        self.log(f"PCC-{self.quality}", pcc)
        self.log("val_loss", torch.stack([o["loss"] for o in outputs]).mean())

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=self.args.weight_decay,
        )
        # return Lamb(
        #     self.parameters(),
        #     lr=self.learning_rate,
        #     weight_decay=self.weight_decay,
        # )
