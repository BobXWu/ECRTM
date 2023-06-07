import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from models.ECRTM import ECRTM


class Runner:
    def __init__(self, args):
        self.args = args
        self.model = ECRTM(args)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def make_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.args.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer,):
        lr_scheduler = StepLR(optimizer, step_size=self.args.lr_step_size, gamma=0.5, verbose=True)
        return lr_scheduler

    def train(self, data_loader):
        optimizer = self.make_optimizer()

        if "lr_scheduler" in self.args:
            print("===>Warning: use lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(data_loader.dataset)

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_data in data_loader:

                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            if 'lr_scheduler' in self.args:
                lr_scheduler.step()

            output_log = f'Epoch: {epoch:03d}'
            for key in loss_rst_dict:
                output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

            print(output_log)

        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def test(self, input_data):
        data_size = input_data.shape[0]
        theta = np.zeros((data_size, self.args.num_topic))
        all_idx = torch.split(torch.arange(data_size), self.args.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = input_data[idx]
                batch_theta = self.model.get_theta(batch_input)
                theta[idx] = batch_theta.cpu()

        return theta
