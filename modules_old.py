import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
from scipy import stats
import math
import random
from ushuffle import shuffle

from sklearn import metrics

import matplotlib.pyplot as plt

class TranscriptsDataset(Dataset):
    def __init__(self, data, max_seq_len):
        self.seq = data.seq.to_list()
        self.is_exon = data.is_exon.to_list()
        # self.label = data.is_exprs.to_list()
        self.n = data.total_reads.to_list()
        self.k_s = data.xlr.to_list()
        self.tr_id = data.index.to_list()
        self.gene_id = data.gene_id.to_list()
        # self.gene_len = data.len.to_list()
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.tr_id)
    
    def _pad_seq(self, seq, target_len):
        seq = seq + ('N' * (target_len - len(seq)))
        return seq
    
    def _get_seq_1h(self, seq):
    
        seq_code = {
            'A': [1,0,0,0],
            'C': [0,1,0,0],
            'G': [0,0,1,0],
            'T': [0,0,0,1],
            'N': [0,0,0,0]
        }

        seq_1h = []
        for s in seq:
            seq_1h.append(seq_code[s])    
        seq_1h = np.array(seq_1h).T.astype(np.float32)

        return seq_1h
    
    def _pad_is_exon(self, is_exon, target_len):
        is_exon = is_exon + ('0' * (target_len - len(is_exon)))
        return is_exon
    
    def _get_is_exon_1h(self, is_exon):
        is_exon_1h = np.array(list(is_exon)).astype(np.float32)[np.newaxis]
        return is_exon_1h

    def __getitem__(self, idx):
        
        seq = self.seq[idx]
        is_exon = self.is_exon[idx]
        
        rand_seq = shuffle(bytes(seq, 'utf-8'), 1).decode('utf-8')
        rand_is_exon = shuffle(bytes(is_exon, 'utf-8'), 1).decode('utf-8')
        
        ## True Seq.
        seq = self._pad_seq(seq, self.max_seq_len)
        seq = self._get_seq_1h(seq)
        is_exon = self._pad_is_exon(is_exon, self.max_seq_len)
        is_exon = self._get_is_exon_1h(is_exon)
        seq = np.row_stack((seq, is_exon))
        seq = torch.from_numpy(seq)    
        
        ## Random Seq.
        rand_seq = self._pad_seq(rand_seq, self.max_seq_len)
        rand_seq = self._get_seq_1h(rand_seq)
        rand_is_exon = self._pad_is_exon(rand_is_exon, self.max_seq_len)
        rand_is_exon = self._get_is_exon_1h(rand_is_exon)
        rand_seq = np.row_stack((rand_seq, rand_is_exon))
        rand_seq = torch.from_numpy(rand_seq)
        
        # label = self.label[idx]
        # torch.tensor(label, dtype = torch.float32)
        
        n = self.n[idx]
        n = torch.tensor([n], dtype = torch.float32)
        
        k_s = self.k_s[idx]
        k_s = torch.tensor([k_s], dtype = torch.float32)

        tr_id = self.tr_id[idx]
        gene_id = self.gene_id[idx]

        return seq, rand_seq, n, k_s, tr_id, gene_id

def PrepareDatasets(data_path,
                    train_frac, val_frac,
                    min_seq_len, max_seq_len):
                    # random_seed
    
    data = pd.read_pickle(data_path)
    data.set_index('tr_id', inplace=True, verify_integrity=True)
    data = data[data.len.between(min_seq_len, max_seq_len)]
    
    # test_frac = 1 - (train_frac + val_frac)
    
    num_data = data.shape[0]
    num_train = int(num_data * train_frac)
    num_val = int(num_data * val_frac)
    num_test = int(num_data - (num_train + num_val))
        
    genes = data.gene_id.unique()
    genes = list(genes)
    # random.Random(random_seed).shuffle(genes)
    random.shuffle(genes)

    train_genes = []
    train_trs = []
    
    num_trs = 0
    while num_trs < num_train:
        train_genes.append(genes.pop())
        train_trs = data[data.gene_id.isin(train_genes)].index
        num_trs = len(train_trs)

    val_genes = []
    val_trs = []
    
    num_trs = 0
    while num_trs < num_val:
        val_genes.append(genes.pop())
        val_trs = data[data.gene_id.isin(val_genes)].index
        num_trs = len(val_trs)

    non_test_trs = set(train_trs).union(set(val_trs))
    test_trs = set(data.index) - non_test_trs
    test_trs = list(test_trs)

    train_data = data.loc[train_trs]
    val_data = data.loc[val_trs]
    test_data = data.loc[test_trs]

    train_data.sort_values('gene_id', inplace=True)
    val_data.sort_values('gene_id', inplace=True)
    test_data.sort_values('gene_id', inplace=True)
    
    train_dataset = TranscriptsDataset(train_data, max_seq_len)
    val_dataset = TranscriptsDataset(val_data, max_seq_len)
    test_dataset = TranscriptsDataset(test_data, max_seq_len)
    
    return (train_dataset, val_dataset, test_dataset)

class BinomialNLLLoss(nn.Module):
    def __init__(self, eps=1e-08, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    ## inputs: p, targets: n, k_s
    def forward(self, p, n, k_s):

        # p = p.view(-1)
        k_f = n - k_s
        nll = -1 * (k_s * torch.log(p + self.eps) + k_f * torch.log(1 - p + self.eps))
        
        if self.reduction == 'mean':
            loss = nll.mean()
        elif self.reduction == 'sum':
            loss = nll.sum()
        else:
            loss = nll

        return loss

class Trainer(nn.Module):
    '''
    The Trainer class. Handles data loading, model training and evaluation, and visualization. 
    Attributes: 
        param_vals (dict) - a dictionary with the parameters
        model (pytoch model) - a predefined model 
        batch_size (int) - the batch size, pulled from the parameter dictionary
        num_targets (int) - the number of targets that the model is trained on, pulled from the parameter dictionary
        train_losses, valid_losses, train_eval_metric_1, valid_eval_metric_1, train_eval_metric_2, valid_eval_metric_2 (arrs) - arrays that keep track of the loss, Pearson R and R2 
        train_losses_ind (arr) - array that keeps track of individual losses for each target 
    '''
    def __init__(self, param_vals, model, data_path): # input_files_dir, target_files_dir
        super(Trainer, self).__init__()
    
        self.param_vals = param_vals
        self.model = model 
        self.mode = self.param_vals.get('mode', 'regression')
        self.train_losses, self.valid_losses, self.train_eval_metric_1, self.valid_eval_metric_1, self.train_eval_metric_2, self.valid_eval_metric_2 = [], [], [], [], [], []
        
        self.tr_epoch_df, self.val_epoch_df = pd.DataFrame(), pd.DataFrame()
        self.tr_cor, self.tr_cor_thres, self.tr_cor_gene, self.tr_kl_gene = [], [], [], []
        self.val_cor, self.val_cor_thres, self.val_cor_gene, self.val_kl_gene = [], [], [], []
        self.tr_rand_cor, self.tr_rand_diff, self.val_rand_cor, self.val_rand_diff = [], [], [], []
        
        self.device = self.model.device
        self.writer = SummaryWriter()
        
        num_targets = self.param_vals.get('num_targets', 1)
        if isinstance(num_targets, list): 
            self.num_targets_lst = num_targets
            self.num_targets = np.sum(num_targets)
        else: 
            self.num_targets_lst = [num_targets]
            self.num_targets = num_targets
        
        self.train_losses_ind = [[] for i in range(self.num_targets)]
        self.optim_step = 0
        
        self.batch_size = self.param_vals.get('batch_size', 8)
#         self.num_targets = self.param_vals.get('num_targets', 1)
        self.make_optimizer()
        self.init_loss()
        self.make_dsets(data_path) # self.make_dsets(input_files_dir, target_files_dir, self.num_targets_lst, mode=self.mode)
        print('init dsets')

    def make_optimizer(self): 
        '''
        Initializes the optimizer
        '''
        if self.param_vals["optimizer"]=="Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.param_vals["init_lr"])
        if self.param_vals["optimizer"]=="AdamW":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.param_vals["init_lr"])
        if self.param_vals["optimizer"]=="SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.param_vals["init_lr"]) #, momentum = self.param_vals["optimizer_momentum"])
        if self.param_vals["optimizer"]=="Adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.param_vals["init_lr"], weight_decay = self.param_vals["weight_decay"])
    
    def make_dsets(self, data_path):
        # cut = self.param_vals.get('cut', .8)
        self.training_dset, self.val_dset, self.test_dset = PrepareDatasets(
            data_path,
            train_frac=self.param_vals.get('train_frac', .8),
            val_frac=self.param_vals.get('val_frac', .1),
            min_seq_len=self.param_vals.get('min_seq_len', 2048),
            max_seq_len=self.param_vals.get('max_seq_len', 4096))
            # random_seed=random_seed)
    
    def make_loaders(self):
        '''
        Initializes three dataloaders: training, validation. 
        '''
        # the batch size for the dataloaders is defined as (seq_len * batch_size) / target_window
        batch_size =self.param_vals.get('batch_size', 64)
        num_workers = self.param_vals.get('num_workers', 4)
        is_shuffle = self.param_vals.get('shuffle_data', False)
        
        train_loader = DataLoader(self.training_dset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers)
        val_loader = DataLoader(self.val_dset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers)
        test_loader = DataLoader(self.test_dset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader
    
    
    def decayed_learning_rate(self, step, initial_learning_rate, decay_rate=0.96, decay_steps=100000):
        '''
        Define the decayed learning rate.
        '''
        return initial_learning_rate * math.pow(decay_rate, (step / decay_steps))
    
    def upd_optimizer(self, optim_step):
        '''
        Update the optimizer given the decayed learning rate calculated above. 
        '''
        decayed_lr = self.decayed_learning_rate(optim_step, initial_learning_rate=self.param_vals["init_lr"])
        for g in self.optimizer.param_groups:
            g['lr'] = decayed_lr 

        
    def init_loss(self):
        '''
        Initializes the losses. 
        '''
        if self.param_vals["loss"]=="mse":
            self.loss_fn = F.mse_loss
        if self.param_vals["loss"]=="poisson":
            self.loss_fn = torch.nn.PoissonNLLLoss(log_input=False, reduction=self.param_vals.get("loss_reduction", "sum"))
        if self.param_vals["loss"]=="bce":
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3000/4000]).to(self.device))
        if self.param_vals["loss"]=="binomial":
            # self.loss_fn = BinomialNLLLoss(reduction=self.param_vals.get("loss_reduction", "sum"))
            self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction=self.param_vals.get("loss_reduction", "mean"))
    

    def plot_results(self, y, out, num_targets):
        '''
        Plots the predictions vs the true values. 
        '''
        if num_targets >= 6: 
            num_targets_plot = 6
        else: 
            num_targets_plot = num_targets
        for i in range(num_targets_plot):
            # ys = y[:, :, i].flatten().cpu().numpy()
            ys = y[:, i].flatten().cpu().numpy()
            if self.mode == 'classification': 
                preds = torch.sigmoid(out).cpu().detach().numpy()
            else: 
                # preds = out[:, :, i].flatten().detach().cpu().numpy()
                preds = out[:, i].flatten().detach().cpu().numpy()
            plt.plot(np.arange(len(ys.flatten())), ys.flatten(), label='True')
            plt.plot(np.arange(len(preds.flatten())), preds.flatten(), label='Predicted', alpha=0.5)
            plt.legend()
            plt.show()        

    def train(self, debug):
        '''
        Main training loop
        '''
        print('began training')
        for epoch in range(self.param_vals.get('num_epochs', 10)):
            self.model.train()
            self.tr_epoch_df = pd.DataFrame()
            train_loader, val_loader, _ = self.make_loaders()
            print(len(train_loader), len(val_loader))
            for batch_idx, batch in enumerate(train_loader):
                print_res, plot_res = False, False

                x, rand_x, n, k_s, tr_id, gene_id = batch
                x, rand_x, n, k_s = x.to(self.device), rand_x.to(self.device), n.to(self.device), k_s.to(self.device)
                if (debug): 
                    print (x.shape, n.shape, k_s.shape)
                if x.shape[0] != 1: 
                    self.optimizer.zero_grad()
                    if batch_idx%10==0:
                        print_res = True
                        if batch_idx%300==0:
                            plot_res = True
                    self.train_step(x, rand_x, n, k_s, tr_id, gene_id, print_res, plot_res, epoch, batch_idx, train_loader)
                    print_res, plot_res = False, False
#             print(self.train_R2)
                             
            if val_loader:
                self.val_epoch_df = pd.DataFrame()
                print_res, plot_res = False, False
                self.model.eval()
                for batch_idx, batch in enumerate(val_loader):
                    print_res, plot_res = False, False 
                    x, rand_x, n, k_s, tr_id, gene_id = batch
                    x, rand_x, n, k_s = x.to(self.device), rand_x.to(self.device), n.to(self.device), k_s.to(self.device)
                    if x.shape[0] != 1: 
                        if batch_idx%10==0:
                            print_res = True
                            if batch_idx%300==0:
                                plot_res = True
                        self.eval_step(x, rand_x, n, k_s, tr_id, gene_id, print_res, plot_res, epoch, batch_idx, val_loader) 
                        print_res, plot_res = False, False 

            train_arrs = np.array([self.train_losses, self.train_eval_metric_1, self.train_eval_metric_2])
            val_arrs = np.array([self.valid_losses, self.valid_eval_metric_1, self.valid_eval_metric_2])
            self.plot_metrics(epoch+1, train_arrs, val_arrs)
            if self.num_targets > 1: 
                self.plot_ind_loss(epoch+1, self.train_losses_ind)
                
            epoch_df_colnames = {0: 'gene_id', 1: 'tr_id', 2: 'n', 3: 'k_s', 4: 'p_true', 5: 'p_pred', 6: 'p_rand'}
            self.tr_epoch_df.rename(epoch_df_colnames, axis=1, inplace=True)
            self.val_epoch_df.rename(epoch_df_colnames, axis=1, inplace=True)
            
            n_thres = self.param_vals.get('n_thres', 5)
            
            self.tr_cor.append(stats.spearmanr(self.tr_epoch_df[['p_true', 'p_pred']])[0])
            self.tr_cor_thres.append(stats.spearmanr(self.tr_epoch_df[self.tr_epoch_df.n >= n_thres][['p_true', 'p_pred']])[0])
            # self.tr_cor_gene.append(self.tr_epoch_df[self.tr_epoch_df.n >= n_thres].groupby('gene_id').apply(lambda x: stats.spearmanr(x[['p_true', 'p_pred']])[0]).mean())
            # self.tr_kl_gene.append(self.tr_epoch_df[self.tr_epoch_df.n >= n_thres].groupby('gene_id').apply(lambda x: stats.entropy(x[['p_true', 'p_pred']])).mean())
            tr_cor_gene_ = self.tr_epoch_df[self.tr_epoch_df.n >= n_thres].groupby(['gene_id', 'n']).apply(lambda x: stats.spearmanr(x[['p_true', 'p_pred']])[0]).reset_index().rename({0: 'cor'}, axis=1).dropna()
            self.tr_cor_gene.append(np.average(tr_cor_gene_.cor, weights=tr_cor_gene_.n))
            tr_kl_gene_ = self.tr_epoch_df[self.tr_epoch_df.n >= n_thres].groupby(['gene_id', 'n']).apply(lambda x: stats.entropy(x[['p_true', 'p_pred']])).reset_index().rename({0: 'kl'}, axis=1).dropna()
            self.tr_kl_gene.append(np.average(tr_kl_gene_.kl, weights=tr_kl_gene_.n))
                        
            self.val_cor.append(stats.spearmanr(self.val_epoch_df[['p_true', 'p_pred']])[0])
            self.val_cor_thres.append(stats.spearmanr(self.val_epoch_df[self.val_epoch_df.n >= n_thres][['p_true', 'p_pred']])[0])
            # self.val_cor_gene.append(self.val_epoch_df[self.val_epoch_df.n >= n_thres].groupby('gene_id').apply(lambda x: stats.spearmanr(x[['p_true', 'p_pred']])[0]).mean())
            # self.val_kl_gene.append(self.val_epoch_df[self.val_epoch_df.n >= n_thres].groupby('gene_id').apply(lambda x: stats.entropy(x[['p_true', 'p_pred']])).mean())                       
            val_cor_gene_ = self.val_epoch_df[self.val_epoch_df.n >= n_thres].groupby(['gene_id', 'n']).apply(lambda x: stats.spearmanr(x[['p_true', 'p_pred']])[0]).reset_index().rename({0: 'cor'}, axis=1).dropna()
            self.val_cor_gene.append(np.average(val_cor_gene_.cor, weights=val_cor_gene_.n))
            val_kl_gene_ = self.val_epoch_df[self.val_epoch_df.n >= n_thres].groupby(['gene_id', 'n']).apply(lambda x: stats.entropy(x[['p_true', 'p_pred']])).reset_index().rename({0: 'kl'}, axis=1).dropna()
            self.val_kl_gene.append(np.average(val_kl_gene_.kl, weights=val_kl_gene_.n))

            self.tr_rand_cor.append(stats.spearmanr(self.tr_epoch_df[['p_rand', 'p_pred']])[0])
            self.val_rand_cor.append(stats.spearmanr(self.val_epoch_df[['p_rand', 'p_pred']])[0])
            self.tr_rand_diff.append((self.tr_epoch_df.p_pred - self.tr_epoch_df.p_rand).abs().mean())
            self.val_rand_diff.append((self.val_epoch_df.p_pred - self.val_epoch_df.p_rand).abs().mean())
            
            self.plot_metrics_within_gene(epoch+1)
            
            if(self.param_vals.get('save_results', False)):
                if epoch%10==0:
                    print(f'Saving model on epoch {epoch}')
                    self.save_model(self.model, f'data/model/Basenji_{self.model.max_seq_len}_ep{epoch}.pt')
         

    def train_step(self, x, rand_x, n, k_s, tr_id, gene_id, print_res, plot_res, epoch, batch_idx, train_loader):
        '''
        Define each training step
        '''

        with torch.cuda.amp.autocast():
            y = k_s / n  
            out = self.model(x).view(y.shape)
            p = torch.sigmoid(out.detach())
            
            out_rand = self.model(rand_x).view(y.shape)
            p_rand = torch.sigmoid(out_rand.detach())
            
            loss = 0
            # calculate loss for each target
            for i in range(y.shape[-1]):
                # loss_ = self.loss_fn(p[:, i], n[:, i], k_s[:, i])
                # loss_ = self.loss_fn(out[:, i], y[:, i], weight=n[:, i])
                loss_ = F.binary_cross_entropy_with_logits(out[:, i], y[:, i], weight=n[:, i])
                self.train_losses_ind[i].append(loss_.data.item())
                loss += loss_
            # if the regularization is required, update the loss
            if self.param_vals.get('lambda_param', None): 
                loss = self.regularize_loss(self.param_vals["lambda_param"], self.param_vals["ltype"], self.model, loss)
                
        if self.mode == 'classification': 
            # calculate the precision and f1 score for classification
            pres, f1 = self.calc_pres_f1(y, out)

        else: 
            # calculate the Pearson R, Spearman R and R2 for regression
            Rp, Rs, R2, mae = self.calc_R_R2(y, p)
        
        # backpropagate the loss
        loss.backward()
        # clip the gradient if required
        if self.param_vals.get('clip', None): 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.param_vals["clip"])
        
        # update the optimizer
        self.optimizer.step()
        self.optim_step += 1
        self.upd_optimizer(self.optim_step)
        
        # record the values for loss, Pearson R, and R2
        self.train_losses.append(loss.data.item())
        if self.mode == 'classification': 
            self.train_eval_metric_1.append(pres)
            self.train_eval_metric_2.append(f1)
        else: 
            self.train_eval_metric_1.append(Rp)
            self.train_eval_metric_2.append(R2)
            
        n_ = n.detach().clone().flatten().cpu().tolist()
        k_s_ = k_s.detach().clone().flatten().cpu().tolist()
        p_true = y.detach().clone().flatten().cpu().tolist()
        p_pred = p.detach().clone().flatten().cpu().tolist()
        p_rand = p_rand.detach().clone().flatten().cpu().tolist()
        
        batch_df = pd.DataFrame(zip(gene_id, tr_id, n_, k_s_, p_true, p_pred, p_rand))
        self.tr_epoch_df = pd.concat([self.tr_epoch_df, batch_df], ignore_index=True)
            
        if print_res: 
            if self.mode == 'classification':
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPres: {:.6f}\tF1 Score: {:.6f}'.format(
                              epoch, batch_idx, len(train_loader), int(100. * batch_idx / len(train_loader)),
                              loss.item(), pres, f1))                
            else: 
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRp: {:.6f}\tRs: {:.6f}\tR2: {:.6f}\tMAE: {:.6f}'.format(
                              epoch, batch_idx, len(train_loader), int(100. * batch_idx / len(train_loader)),
                              loss.item(), Rp, Rs, R2, mae))
        if plot_res: 
            print (torch.sum(y).item(), torch.sum(p).item())
            self.plot_results(y, p, self.num_targets)
            
            
        self.writer.add_scalar('Loss/Train', loss, f'{epoch}_{batch_idx}')

    def eval_step(self, x, rand_x, n, k_s, tr_id, gene_id, print_res, plot_res, epoch, batch_idx, val_loader):
        '''
        Define each evaluation step
        '''
        y = k_s / n
        out = self.model(x).view(y.shape)
        p = torch.sigmoid(out.detach())
        # p = self.model(x).view(y.shape)
        
        out_rand = self.model(rand_x).view(y.shape)
        p_rand = torch.sigmoid(out_rand.detach())

        loss = 0
        for i in range(y.shape[-1]):
            # loss_ = self.loss_fn(p[:, i], n[:, i], k_s[:, i])
            # loss_ = self.loss_fn(out[:, i], y[:, i], weight=n[:, i])
            loss_ = F.binary_cross_entropy_with_logits(out[:, i], y[:, i], weight=n[:, i])

            loss += loss_

#         loss = self.loss_fn(out,y)
        if self.mode == 'classification': 
            # calculate the precision and f1 score for classification
            pres, f1 = self.calc_pres_f1(y, out)

        else: 
            # calculate the Pearson R and R2 for regression
            Rp, Rs, R2, mae = self.calc_R_R2(y, p)
        
        self.valid_losses.append(loss.data.item())
        if self.mode == 'classification': 
            self.valid_eval_metric_1.append(pres)
            self.valid_eval_metric_2.append(f1)                
        else: 
            self.valid_eval_metric_1.append(Rp)
            self.valid_eval_metric_2.append(R2)                
        
        n_ = n.detach().clone().flatten().cpu().tolist()
        k_s_ = k_s.detach().clone().flatten().cpu().tolist()
        p_true = y.detach().clone().flatten().cpu().tolist()
        p_pred = p.detach().clone().flatten().cpu().tolist()
        p_rand = p_rand.detach().clone().flatten().cpu().tolist()
        
        batch_df = pd.DataFrame(zip(gene_id, tr_id, n_, k_s_, p_true, p_pred, p_rand))
        self.val_epoch_df = pd.concat([self.val_epoch_df, batch_df], ignore_index=True)

        if print_res: 
            if self.mode == 'classification':
                print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPres: {:.6f}\tF1 Score: {:.6f}'.format(
                              epoch, batch_idx, len(val_loader), int(100. * batch_idx / len(val_loader)),
                              loss.item(), pres, f1))
            
            else:
                print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRp: {:.6f}\tRs: {:.6f}\tR2: {:.6f}\tMAE: {:.6f}'.format(
                              epoch, batch_idx, len(val_loader), int(100. * batch_idx / len(val_loader)),
                              loss.item(), Rp, Rs, R2, mae))
        if plot_res: 
            self.plot_results(y, p, self.num_targets)
        
        self.writer.add_scalar('Loss/Validation', loss, f'{epoch}_{batch_idx}')


    def mean_arr(self, num_epochs, arr):
        num_iter = int(len(arr) / num_epochs)
        mean_train_arr = [np.mean(arr[i*num_iter:(i+1)*num_iter]) for i in range(num_epochs)]
        return mean_train_arr
            
    def plot_metrics(self, num_epochs, train_arrs, val_arrs): 
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
        titles = ['CELoss', 'PCor', 'RSquared']
        for i in range(3):
            mean_train_arr = self.mean_arr(num_epochs, train_arrs[i])
            mean_val_arr = self.mean_arr(num_epochs, val_arrs[i])
            axs[i].plot(np.arange(num_epochs-1), mean_train_arr[1:], label='Train')
            axs[i].plot(np.arange(num_epochs-1), mean_val_arr[1:], label='Val')
            axs[i].set_title(titles[i])
        fig.tight_layout()
        if(self.param_vals.get('save_results', False)):
            if num_epochs%10 == 0:
                plt.savefig(f'data/plots/Basenji_{self.param_vals.get("max_seq_len")}_ep{num_epochs}.png')
        plt.show()
        
    def plot_metrics_within_gene(self, num_epochs):
        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 6))
        
        axs[0, 0].plot(np.arange(num_epochs), self.tr_cor, label='Train')
        axs[0, 0].plot(np.arange(num_epochs), self.val_cor, label='Val')
        axs[0, 0].set_title('SpCor')
        
        axs[0, 1].plot(np.arange(num_epochs), self.tr_cor_thres, label='Train')
        axs[0, 1].plot(np.arange(num_epochs), self.val_cor_thres, label='Val')
        axs[0, 1].set_title('SpCorNThres')
        
        axs[0, 2].plot(np.arange(num_epochs), self.tr_cor_gene, label='Train')
        axs[0, 2].plot(np.arange(num_epochs), self.val_cor_gene, label='Val')
        axs[0, 2].set_title('SpCorWithinGene')
        
        axs[0, 3].plot(np.arange(num_epochs), self.tr_kl_gene, label='Train')
        axs[0, 3].plot(np.arange(num_epochs), self.val_kl_gene, label='Val')
        axs[0, 3].set_title('KLWithinGene')
        
        axs[1, 0].plot(np.arange(num_epochs), self.tr_rand_cor, label='Train')
        axs[1, 0].plot(np.arange(num_epochs), self.val_rand_cor, label='Val')
        axs[1, 0].set_title('SpCorPredVSRand')

        axs[1, 1].plot(np.arange(num_epochs), self.tr_rand_diff, label='Train')
        axs[1, 1].plot(np.arange(num_epochs), self.val_rand_diff, label='Val')
        axs[1, 1].set_title('DiffPredVSRand')

        fig.tight_layout()
        if self.param_vals.get('save_results', False):
            if num_epochs%10 == 0:
                plt.savefig(f'data/plots/Basenji_{self.param_vals.get("max_seq_len")}_ep{num_epochs}_within_gene.png')
        plt.show()    
        
    def plot_ind_loss(self, num_epochs, train_arrs_ind):
        '''
        Plots individual losses for 4 targets side by side
        '''
#         num_targets = self.param_vals.get('num_targets', 1)
        if self.num_targets >= 4: 
            num_targets = 4
        else: 
            num_targets = self.num_targets

        fig, axs = plt.subplots(nrows=1, ncols=num_targets+1, figsize=(15, 3))
        for i in range(num_targets):
            mean_train_arr = self.mean_arr(num_epochs, train_arrs_ind[i])
            axs[num_targets].plot(np.arange(num_epochs-1), mean_train_arr[1:], label='Train')
            axs[i].plot(np.arange(num_epochs-1), mean_train_arr[1:], label='Train')
        fig.tight_layout()
        plt.show()    


    def calc_pres_f1(self, y_true, y_pred): 
        '''
        Handles the precision and f1-score calculation
        '''

        y_true = y_true.cpu().detach().numpy().astype(int).flatten()
#         y_pred = torch.round(y_pred).cpu().detach().numpy().flatten().astype(int)
        y_pred_tag = torch.round(torch.sigmoid(y_pred)).cpu().detach().numpy().astype(int).flatten()

        f1 = f1_score(y_true, y_pred_tag, average='binary', zero_division=0)
        pres = precision_score(y_true, y_pred_tag, average='binary', zero_division=0)
        return pres, f1

    def calc_R_R2(self, y_true, y_pred):
        '''
        Handles the Pearson R and R2 calculation
        '''        
        y_true = y_true.detach().clone().cpu().numpy().astype(np.float32).flatten()
        y_pred = y_pred.detach().clone().cpu().numpy().astype(np.float32).flatten()
        
        pearson_cor = round(stats.pearsonr(y_true, y_pred)[0], 4)
        spearman_cor = round(stats.spearmanr(y_true, y_pred)[0], 4)
        r2 = round(metrics.r2_score(y_true, y_pred), 4)
        mae = round(metrics.mean_absolute_error(y_true, y_pred), 4)
        
        return pearson_cor, spearman_cor, r2, mae


        
    def regularize_loss(self, lambda1, ltype, net, loss):
        '''
        Handles regularization for each conv block.
            ltype values: 
                1, 2 - L1 and L2 regularizations 
                3 - gradient clipping 
               
        '''
        if ltype == 3:
                torch.nn.utils.clip_grad_norm_(
                    net.conv_block_1.parameters(), lambda1)
                torch.nn.utils.clip_grad_norm_(
                    net.conv_block_2.parameters(), lambda1)
                torch.nn.utils.clip_grad_norm_(
                    net.conv_block_3.parameters(), lambda1)
                torch.nn.utils.clip_grad_norm_(
                    net.conv_block_4.parameters(), lambda1)
                torch.nn.utils.clip_grad_norm_(
                        net.conv_block_5.parameters(), lambda1)
                for i in range(len(net.dilations)):
                    torch.nn.utils.clip_grad_norm_(
                        net.dilations[i].parameters(), lambda1)

        else:      
            l0_params = torch.cat(
                [x.view(-1) for x in net.conv_block_1[1].parameters()])
            l1_params = torch.cat(
                [x.view(-1) for x in net.conv_block_2[1].parameters()])
            l2_params = torch.cat(
                [x.view(-1) for x in net.conv_block_3[1].parameters()])
            l3_params = torch.cat(
                [x.view(-1) for x in net.conv_block_4[1].parameters()])
            l4_params = torch.cat(
                    [x.view(-1) for x in net.conv_block_5[1].parameters()])
            dil_params = []
            for i in range(len(net.dilations)):
                dil_params.append(torch.cat(
                    [x.view(-1) for x in net.dilations[i][1].parameters()]))

        if ltype in [1, 2]:
            l1_l0 = lambda1 * torch.norm(l0_params, ltype)
            l1_l1 = lambda1 * torch.norm(l1_params, ltype)
            l1_l2 = lambda1 * torch.norm(l2_params, ltype)
            l1_l3 = lambda1 * torch.norm(l3_params, ltype)
            l1_l4 = lambda1 * torch.norm(l4_params, 1)
            l1_l4 = lambda1 * torch.norm(l4_params, 2)
            dil_norm = []
            for d in dil_params:
                dil_norm.append(lambda1 * torch.norm(d, ltype))  
            loss = loss + l1_l0 + l1_l1 + l1_l2 + l1_l3 + l1_l4 + torch.stack(dil_norm).sum()

        elif ltype == 4:
            l1_l0 = lambda1 * torch.norm(l0_params, 1)
            l1_l1 = lambda1 * torch.norm(l1_params, 1)
            l1_l2 = lambda1 * torch.norm(l2_params, 1)
            l1_l3 = lambda1 * torch.norm(l3_params, 1)
            l2_l0 = lambda1 * torch.norm(l0_params, 2)
            l2_l1 = lambda1 * torch.norm(l1_params, 2)
            l2_l2 = lambda1 * torch.norm(l2_params, 2)
            l2_l3 = lambda1 * torch.norm(l3_params, 2)
            l1_l4 = lambda1 * torch.norm(l4_params, 1)
            l2_l4 = lambda1 * torch.norm(l4_params, 2)
            dil_norm1, dil_norm2 = [], []
            for d in dil_params:
                dil_norm1.append(lambda1 * torch.norm(d, 1))  
                dil_norm2.append(lambda1 * torch.norm(d, 2))  

            loss = loss + l1_l0 + l1_l1 + l1_l2 +                    l1_l3 + l1_l4 + l2_l0 + l2_l1 +                    l2_l2 + l2_l3 + l2_l4 +                 torch.stack(dil_norm1).sum() + torch.stack(dil_norm2).sum()
        return loss

    def save_model(self, model, filename):
        '''
        Handles model saving
        '''
        torch.save(model.state_dict(), filename)

    def load_model(self, model, filename):
        '''
        Handles model loading
        '''
        model.load_state_dict(torch.load(filename))
