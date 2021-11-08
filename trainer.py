import os

import torch

from typing import Tuple, List, Dict, Optional, Any

from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader


class Logger:
    def log(self, data: Dict[str, Any]) -> None:
        raise NotImplementedError()


class PrintLogger(Logger):
    def log(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            print(key, value)


# For the sake of not having wandb installed, I am not writing
# WandbLogger class, however note that you can pass wandb module
# itself into learner's logger paramenter and will work just fine.


class Metric:
    def update(self, prediction: Tensor, target: Tensor) -> None:
        raise NotImplementedError()

    def compute(self) -> float:
        raise NotImplementedError()

    def reset(self) -> None:
        raise NotImplementedError()


class Accuracy(Metric):
    def update(self, prediction: Tensor, target: Tensor) -> None:
        prediction = torch.argmax(prediction, dim=1)
        self.correct += torch.sum(prediction == target)
        self.total += torch.numel(target)

    def compute(self) -> float:
        return self.correct / self.total

    def reset(self) -> None:
        self.correct = 0
        self.total = 0


class Trainer:
    def __init__(
        self,
        model: Module,
        criterion: _Loss,
        batch_size: int,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        metrics: Optional[List[Metric]] = None,
        gpu_number: int = 0,
        logger: Optional[Logger] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.metrics = metrics or []
        self.device = torch.device(
            f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu"
        )
        self.logger = logger or PrintLogger()

        self.configure_dataloaders()
        self.model.to(self.device)
        self.epochs = None
        self.learning_rate = None
        self.optimizer = None
        self.scheduler = None

    def fit(self, epochs: int = 10, learning_rate: float = 1e-3) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.configure_optimizers()
        for epoch in range(self.epochs):
            self.training_epoch()

            # Validate if the validation dataset is available.
            if self.val_dataloader is None:
                continue

            with torch.no_grad():
                val_loss, metric_values = self.validation_epoch()

            val_summary = {
                "epoch": epoch,
                "val_loss": val_loss,
            }
            val_summary.update(metric_values)
            self.logger.log(val_summary)

    def training_epoch(self) -> None:
        self.model.train()
        for input, target in self.train_dataloader:
            input, target = input.to(self.device), target.to(self.device)
            loss = self.training_step(input, target)
            self.optimization_step(loss)
            self.logger.log({"train_loss": loss.item()})

    def training_step(self, input: Tensor, target: Tensor) -> Tensor:
        prediction = self.model(input)
        loss = self.criterion(prediction, target)
        return loss

    def optimization_step(self, loss: Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def validation_epoch(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        self.reset_metrics()
        val_loss = 0
        for input, target in self.val_dataloader:
            input, target = input.to(self.device), target.to(self.device)
            loss, prediction = self.validation_step(input, target)
            val_loss += loss.item()
            self.update_metrics(prediction, target)

        return val_loss / len(self.val_dataloader), self.compute_metrics()

    def validation_step(self, input, target) -> Tuple[Tensor, Tensor]:
        prediction = self.model(input)
        loss = self.criterion(prediction, target)
        return loss, prediction

    def configure_dataloaders(self) -> None:
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=True,
        )

        if self.val_dataset is None:
            self.val_dataloader = None
            return

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=True,
        )

    def configure_optimizers(self) -> None:
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            self.learning_rate,
            epochs=self.epochs,
            steps_per_epoch=len(self.train_dataloader),
        )

    def find_learning_rate(
        self,
        start_value: float = 1e-7,
        end_value: float = 10.0,
        steps: int = 100,
        beta: float = 0.98,
        early_stop: bool = True,
    ) -> Tuple[List[float], List[float]]:
        self.save_model()

        gamma = (end_value / start_value) ** (1 / steps)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=start_value)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma)
        self.model.train()

        learning_rate_history, loss_history = [], []
        average_loss = 0.0
        best_loss = None
        train_dataloader = iter(self.train_dataloader)
        step = 0
        while step < steps + 1:
            # In case of len(train_dataloader) < steps.
            try:
                input, target = next(train_dataloader)
            except StopIteration:
                train_dataloader = iter(self.train_dataloader)
                input, target = next(train_dataloader)

            step += 1
            loss = self.training_step(
                input.to(self.device), target.to(self.device)
            )

            # Apply exponential smoothing.
            average_loss = average_loss * beta + loss.item() * (1 - beta)
            smooth_loss = average_loss / (1 - beta ** step)
            if step == 1 or smooth_loss < best_loss:
                best_loss = smooth_loss
            elif early_stop and step > 1 and smooth_loss > 4 * best_loss:
                break

            learning_rate_history.append(self.optimizer.param_groups[0]["lr"])
            loss_history.append(smooth_loss)

            self.optimization_step(loss)

        self.load_model()
        return learning_rate_history, loss_history

    def save_model(self, path="model.pth") -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="model.pth") -> None:
        self.model.load_state_dict(torch.load(path))

    def reset_metrics(self) -> None:
        for metric in self.metrics:
            metric.reset()

    def update_metrics(self, prediction: Tensor, target: Tensor) -> None:
        for metric in self.metrics:
            metric.update(prediction, target)

    def compute_metrics(self) -> Dict[str, float]:
        return {
            f"val_{self.metric_name(metric)}": metric.compute()
            for metric in self.metrics
        }

    def metric_name(self, metric: Metric) -> str:
        return metric.__class__.__name__.lower()
