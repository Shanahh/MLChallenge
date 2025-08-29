import math
import matplotlib.pyplot as plt
import torch

class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.best_loss = None
        self.losses = []
        self.log_lrs = []
        model.to(device)

    def range_test(self, train_loader, start_lr=1e-7, end_lr=10, beta=0.98):
        """
        Runs the Learning Rate Range Test.
        start_lr: initial learning rate
        end_lr: final learning rate
        beta: smoothing factor for the moving average of the loss
        """
        num = len(train_loader) - 1
        mult = (end_lr / start_lr) ** (1/num)
        lr = start_lr
        self.optimizer.param_groups[0]['lr'] = lr

        avg_loss = 0.0
        batch_num = 0
        self.best_loss = float('inf')
        self.losses = []
        self.log_lrs = []

        self.model.train()
        for inputs, masks in train_loader:
            batch_num += 1
            inputs = inputs.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, masks)

            # Smoothed loss to reduce noise
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)

            # Stop if the loss explodes
            if batch_num > 1 and smoothed_loss > 4 * self.best_loss:
                break

            if smoothed_loss < self.best_loss:
                self.best_loss = smoothed_loss

            self.losses.append(smoothed_loss)
            self.log_lrs.append(math.log10(lr))

            loss.backward()
            self.optimizer.step()

            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr

        return self.log_lrs, self.losses

    def plot(self):
        """
        Plots the results of the learning rate test.
        """
        plt.plot(self.log_lrs, self.losses)
        plt.xlabel("log10(Learning Rate)")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True)
        plt.show()
