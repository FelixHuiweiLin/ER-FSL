import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_aug
from utils.utils import maybe_cuda
import numpy as np

class ER_FSL(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ER_FSL, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.seen_so_far = torch.LongTensor(size=(0,)).to('cuda')
        self.selective_index = None
        self.subspace = params.subspace
        self.scale = params.scale

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=None)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()
        fea_length = self.model.fer_linear.L.weight.data.shape[1]
        base_length = self.subspace
        aaa = self.i
        if base_length * (aaa+1) > fea_length:
            linear_weight = self.model.fer_linear.L.weight.data[self.old_labels,:]
            score = torch.var(linear_weight, dim=0)
            new_index = torch.argsort(score, descending=False)[0: base_length]
            old_index = np.arange(0, fea_length, 1)
        else:
            new_index = np.arange(aaa * base_length, (aaa + 1) * base_length, 1)
            old_index = np.arange(0, (aaa + 1) * base_length, 1)

        self.selective_index = old_index

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_y_save = batch_y
                batch_x_save = batch_x
                batch_x_aug = torch.stack([transforms_aug[self.data](batch_x[idx].cpu())
                                           for idx in range(batch_x.size(0))])
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_x_aug = maybe_cuda(batch_x_aug, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                batch_y_save = maybe_cuda(batch_y_save, self.cuda)
                batch_x_save = maybe_cuda(batch_x_save, self.cuda)
                batch_x_combine = torch.cat((batch_x, batch_x_aug))
                batch_y_combine = torch.cat((batch_y, batch_y))

                present = batch_y.unique()
                self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()
                novel_loss = 0
                scale = self.scale
                for j in range(self.mem_iters):
                    self.opt.zero_grad()
                    mem_x, mem_y, mem_old_logits, _, _, _ = self.buffer.retrieve(x=batch_x, y=batch_y)
                    if mem_x.size(0) > 0:
                        mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
                                                 for idx in range(mem_x.size(0))])
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_x_combine = torch.cat([mem_x, mem_x_aug])
                        mem_y_combine = torch.cat([mem_y, mem_y])
                        mem_logits= self.model.erfsl_forward(mem_x_combine, selective_index=old_index)
                        mem_loss=scale*self.criterion(mem_logits, mem_y_combine)
                        novel_loss += mem_loss

                    logits= self.model.erfsl_forward(batch_x_combine, selective_index=new_index)
                    current_loss = (1-scale) * self.criterion(logits, batch_y_combine)
                    novel_loss += current_loss

                    # # backward
                    novel_loss.backward()

                    self.opt.step()
                # update mem
                self.buffer.update(batch_x_save, batch_y_save)

        self.after_train()