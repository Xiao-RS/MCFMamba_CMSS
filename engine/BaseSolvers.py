import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch.optim as optim
from collections import OrderedDict
from tensorboardX import SummaryWriter
from torchmetrics import MetricCollection, Accuracy
from .metrics import ConfusionMatrix
from .utils import (
    SchedulerByIter,
    AverageMeter,
    load_cpt,
    save_cpt,
    to_cuda,
    log_msg,
    Convert2Color,
)

Optimizer_Dict = {
    "AdamW": optim.AdamW,
    "SGD": optim.SGD,
}


class Trainer(object):
    def __init__(self, cfg_SOLVER, log_path, train_loader, val_loader, modeler, num_classes, labels_dict):
        self.cfg_SOLVER = cfg_SOLVER
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.modeler = modeler
        self.num_classes = num_classes
        params = self.modeler.module.get_learnable_parameters()
        self.optimizer = Optimizer_Dict[cfg_SOLVER.OPTIMIZER_TYPE](
            params, cfg_SOLVER.LR, **cfg_SOLVER.OPTIMIZER_SPECS[0]
        )
        self.num_iter_perepoch = len(self.train_loader)
        self.scheduler = SchedulerByIter(
            self.optimizer, cfg_SOLVER.LR, self.num_iter_perepoch, **cfg_SOLVER.SCHEDULER_SPECS[0]
        )

        self.epoch = -1
        self.iter = -1
        self.best_score = -1
        self.best_epoch = -1
        self.non_improved_counter = 0
        self.early_stopping_patience = cfg_SOLVER.PATIENCE
        self.ignore_labels = -100

        self.log_path = log_path
        self.tf_writer = SummaryWriter(self.log_path)
        self.cpt_path = os.path.join(self.log_path, "cpt")
        if not os.path.exists(self.cpt_path):
            os.makedirs(self.cpt_path)

    def save_checkpoints(self):
        print(log_msg("Saving checkpoint", "PROCESS"))
        state = {
            "epoch": self.epoch,
            "model": self.modeler.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "non_improved_counter": self.non_improved_counter,
        }

        # update the latest
        save_cpt(state, os.path.join(self.cpt_path, "latest.pth"))

        # update the best
        if self.epoch == self.best_epoch:
            print(log_msg("Saving the best model", "PROCESS"))
            model_state = {
                "model": self.modeler.module.model.state_dict()
            }
            save_cpt(model_state, os.path.join(self.cpt_path, "model_best.pth"))
        elif self.epoch > self.cfg_SOLVER.SAVE_CPT_FREQ:
            print(log_msg(
                "The {} epoch is still the best model with an accuracy of {}".format(
                    self.best_epoch, self.best_score), "INFO"
            ))

    def load_latest_model(self):
        print(log_msg("Loading the latest model", "PROCESS"))
        state = load_cpt(os.path.join(self.cpt_path, "latest.pth"))
        self.epoch = state["epoch"] + 1
        self.modeler.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.best_score = state["best_score"]
        self.best_epoch = state["best_epoch"]
        self.non_improved_counter = state["non_improved_counter"]
        print(log_msg("Best Epoch/Score of previous training: {}/{} ".format(self.best_epoch, self.best_score), "INFO"))

    def train(self, resume=False):
        self.epoch = 1

        if resume:
            self.load_latest_model()

        while self.epoch < self.cfg_SOLVER.EPOCHS + 1:
            self.train_epoch()
            self.epoch += 1

            # early stop
            if self.non_improved_counter == self.early_stopping_patience:
                print(log_msg("Early stopping", "PROCESS"))
                break

        print(log_msg("Best score:{}".format(self.best_score), "EVAL"))

    def train_epoch(self):
        trainval_meters = OrderedDict({"train_loss": AverageMeter(), })
        pbar = tqdm(range(self.num_iter_perepoch))

        # train loops
        self.modeler.train()
        for idx, data in enumerate(self.train_loader):
            self.iter = idx + (self.epoch - 1) * self.num_iter_perepoch
            self.scheduler.step(idx, iter=self.iter)
            self.tf_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.iter)

            self.train_iter(data, trainval_meters)
            pbar.set_description(log_msg(
                "Epoch:{}| Loss:{:.4f}".format(self.epoch, trainval_meters["train_loss"].avg), "TRAIN"
            ))
            pbar.update()
        pbar.close()

        # log
        self.tf_writer.add_scalar('train_loss', trainval_meters["train_loss"].avg, self.epoch)

        # validate
        trainval_meters["val_loss"] = AverageMeter()
        metrics = self.validate(trainval_meters)

        # update the best_acc
        eva_score = metrics["accuracy"]
        if eva_score > self.best_score:
            self.best_score = eva_score
            self.best_epoch = self.epoch
            self.non_improved_counter = 0
        else:
            self.non_improved_counter += 1

        # saving checkpoint
        self.save_checkpoints()

        # log
        self.tf_writer.add_scalar('val_loss', trainval_meters["val_loss"].avg, self.epoch)
        self.tf_writer.add_scalar("val_score", eva_score, self.epoch)

        torch.cuda.empty_cache()

    def train_iter(self, data, trainval_meters):
        image, target = data['image'], data['label']
        image = to_cuda(image)
        target = to_cuda(target)

        # forward
        losses_dict = self.modeler(image=image, target=target, epoch=self.epoch)
        loss = sum([l.mean() for l in losses_dict.values()])
        # backward
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        # collect train info
        batch_size = target.size(0)
        trainval_meters["train_loss"].update(loss.cpu().detach().numpy().mean(), batch_size)

        # tensorboard log
        for k, v in losses_dict.items():
            self.tf_writer.add_scalars('train', {k: v.cpu().detach().numpy()}, self.iter)

        torch.cuda.empty_cache()

    def validate(self, trainval_meters):
        num_iter = len(self.val_loader)
        pbar = tqdm(range(num_iter + 1))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        metric_collection = MetricCollection({
            'accuracy': Accuracy(task="multiclass", num_classes=self.num_classes).to(device=device),
        })

        self.modeler.eval()
        with torch.no_grad():
            for idx, data in enumerate(self.val_loader):
                self.val_batch(data, trainval_meters, metric_collection)
                pbar.set_description(log_msg(
                    "Epoch:{}| Loss:{:.4f}".format(self.epoch, trainval_meters["val_loss"].avg), "EVAL"
                ))
                pbar.update()
            epoch_metrics = metric_collection.compute()
            pbar.set_description(log_msg("Epoch:{}| Loss:{:.4f}| Acc:{:.4f}".format(
                self.epoch, trainval_meters["val_loss"].avg, epoch_metrics['accuracy']
            ), "EVAL"))
            pbar.update()
        pbar.close()
        return epoch_metrics

    def val_batch(self, data, trainval_meters, metric_collection):
        image, target = data['image'], data['label']
        image = to_cuda(image)
        target = to_cuda(target)
        logits, losses_dict = self.modeler(image=image, target=target)
        predicted = torch.argmax(logits, dim=1)

        # update loss
        batch_size = target.shape[0]
        loss = sum([l.mean() for l in losses_dict.values()])
        trainval_meters["val_loss"].update(loss.cpu().detach().numpy().mean(), batch_size)

        mask = (target != self.ignore_labels)
        batch_metrics = metric_collection.forward(predicted[mask], target[mask])

        torch.cuda.empty_cache()


class Tester(object):
    def __init__(self, log_path, test_loader, model, num_classes, labels_dict):
        self.visual_save_path = os.path.join(log_path, 'visual-test')
        self.test_loader = test_loader
        self.model = model
        self.num_classes = num_classes
        self.labels = list(labels_dict.keys())
        self.cm = ConfusionMatrix(num_classes=num_classes if num_classes >= 2 else 2, labels=self.labels)
        self.convert_to_color = Convert2Color(labels_dict)

    def test(self, resultxlsx):
        num_iter = len(self.test_loader)
        pbar = tqdm(range(num_iter))

        self.model.eval()
        with torch.no_grad():
            for idx, data in enumerate(self.test_loader):
                self.test_batch(data)
                # print test info
                msg = "Test Accuracy:{:.3f}".format(self.cm.acc)
                pbar.set_description(log_msg(msg, "TEST"))
                pbar.update()
        pbar.close()
        print(log_msg("Calculating evaluation indicators from the confusion matrix", "PROCESS"))
        self.cm.statistics()
        with pd.ExcelWriter(resultxlsx) as writer:
            self.cm.metric_df.to_excel(writer, sheet_name='dfMetrics')
            pd.DataFrame([self.cm.metric_dict]).to_excel(writer, sheet_name='dictMetrics')
            mat_df = pd.DataFrame(self.cm.matrix)
            mat_df.index = mat_df.columns = self.labels
            mat_df.to_excel(writer, sheet_name='Matrix')
        torch.cuda.empty_cache()

    def test_batch(self, data):
        image, target, idx = data['image'], data['label'], data['id']
        image = to_cuda(image)
        logits = self.model(image)
        predicted = torch.argmax(logits, dim=1)
        target = target.numpy()
        predicted = predicted.cpu().numpy()
        # update confusionmatrix
        self.cm.update(predicted.flatten(), target.flatten())

        self.visualize(image, target, idx, predicted)

    def visualize(self, image, target, idx, predicted):
        if not os.path.exists(self.visual_save_path):
            os.makedirs(self.visual_save_path)
        for i in range(len(idx)):
            pred = Image.fromarray(self.convert_to_color(predicted[i]).transpose(1, 2, 0))
            pred.save(os.path.join(self.visual_save_path, idx[i] + '.png'))
