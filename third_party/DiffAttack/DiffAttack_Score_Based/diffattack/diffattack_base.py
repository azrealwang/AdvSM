import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

from .other_utils import L0_norm, L1_norm, L2_norm
from .checks import check_zero_gradients


def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    #u = torch.min(u, epsinf - torch.clone(y).abs())
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()
    
    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)
    
    inu = 2*(indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)
    
    s1 = -u.sum(dim=1)
    
    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)
    
    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)
    
    if c2.nelement != 0:
    
      lb = torch.zeros_like(c2).float()
      ub = torch.ones_like(lb) *(bs.shape[1] - 1)
      
      #print(c2.shape, lb.shape)
      
      nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
      counter2 = torch.zeros_like(lb).long()
      counter = 0
          
      while counter < nitermax:
        counter4 = torch.floor((lb + ub) / 2.)
        counter2 = counter4.type(torch.LongTensor)
        
        c8 = s[c2, counter2] + c[c2] < 0
        ind3 = c8.nonzero().squeeze(1)
        ind32 = (~c8).nonzero().squeeze(1)
        #print(ind3.shape)
        if ind3.nelement != 0:
            lb[ind3] = counter4[ind3]
        if ind32.nelement != 0:
            ub[ind32] = counter4[ind32]
        
        #print(lb, ub)
        counter += 1
        
      lb2 = lb.long()
      alpha = (-s[c2, lb2] -c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
      d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])
    
    return (sigma * d).view(x2.shape)





class APGDDiffAttack():
    """
    AutoPGD
    https://arxiv.org/abs/2003.01690

    :param predict:       forward pass function
    :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           bound on the norm of perturbations
    :param seed:          random seed for the starting point
    :param loss:          loss to optimize ('ce', 'dlr' supported)
    :param eot_iter:      iterations for Expectation over Trasformation
    :param rho:           parameter for decreasing the step size
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            seed=0,
            loss='ce',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=True,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None,
            args=None):
        """
        AutoPGD implementation in PyTorch
        """
        
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        #self.init_point = None
        self.use_largereps = use_largereps
        #self.larger_epss = None
        #self.iters = None
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0.
        self.is_tf_model = is_tf_model
        self.y_target = None
        self.logger = logger
        self.args = args

        assert self.norm in ['Linf', 'L2', 'L1']
        assert not self.eps is None

        ### set parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

        self.x_test = None
        self.y_test = None

        # self.mean_x = torch.load('file_cluster_centers')
        #
        # self.mean_centers = torch.zeros([10,10,5,1024],dtype=torch.float)
        # for obs_i in range(0,10,1):
        #     for k in range(10):
        #         self.mean_centers[obs_i][k] = self.mean_x[obs_i*10][k].clone()
        # self.mean_centers = self.mean_centers.cuda()


        self.dd = 'imagenet'

    def init_hyperparam(self, x):

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()

        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)

        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
            1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def mse_loss(self, mid_x, ori_x, idx):
        mse = torch.nn.MSELoss(reduction='none')
        mid_x = torch.stack(mid_x)
        ori_x = torch.stack(ori_x)

        loss = mse(mid_x[idx], ori_x[idx].view(mid_x[idx].shape))

        # print(loss.shape) 5, 55, 3072
        # print(f'mid_x: {mid_x}')
        # print(f'ori_x: {ori_x}')

        return loss


    def score_matching_loss(self, model, x, t, std, z):

        score = model.runner.rev_vpsde.rvpsde_fn(t, x, return_type='score')

        z = z.view(x.shape[0],-1)

        loss = torch.square(score * std + z)
        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1)
        # loss = torch.mean(losses)

        return loss

    def mid_score(self, model):
        # assert len(model.runner.rev_vpsde.score_store) == 100, f'wrong number of mid scores: {len(model.runner.rev_vpsde.score_store)}'
        return model.runner.rev_vpsde.score_store

    def find_nearest_feature(self,fea, y, mean):
        min_dis = 1e10
        tar_fea = None
        for l in range(10):
            if l == y:
                continue
            for kk in range(mean[l].shape[0]):
                # x = mean[l][kk].clone()
                # x = x.cuda()
                dis = torch.norm(fea - mean[l][kk], p=2)
                if dis < min_dis:
                    tar_fea = mean[l][kk]
                    min_dis = dis
        return tar_fea

    def loss_boundary_aware(self,mean_feature, x_fea, y):
        target_fea = torch.zeros_like(x_fea).to(x_fea.device)
        for i_x in range(x_fea.shape[0]):
            target_fea[i_x] = self.find_nearest_feature(x_fea[i_x], y[i_x], mean_feature)

        loss = torch.nn.MSELoss(reduction='none')(x_fea, target_fea)
        loss = torch.mean(loss, dim=-1)
        return loss



    
    def attack_single_run(self, x, y, x_init=None):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L1':
            t = torch.randn(x.shape).to(self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta
        
        
        if not x_init is None:
            x_adv = x_init.clone()
            if self.norm == 'L1' and self.verbose:
                print('[custom init] L1 perturbation {:.5f}'.format(
                    (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()))
            
        
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]
            ).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
            ).to(self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        if not self.is_tf_model:
            if self.loss == 'ce':
                criterion_indiv = nn.CrossEntropyLoss(reduction='none')
            elif self.loss == 'ce-targeted-cfts':
                criterion_indiv = lambda x, y: -1. * F.cross_entropy(x, y,
                    reduction='none')
            elif self.loss == 'dlr':
                criterion_indiv = self.dlr_loss
            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.dlr_loss_targeted
            elif self.loss == 'ce-targeted':
                criterion_indiv = self.ce_loss_targeted
            else:
                raise ValueError('unknowkn loss')
        else:
            if self.loss == 'ce':
                criterion_indiv = self.model.get_logits_loss_grad_xent
            elif self.loss == 'dlr':
                criterion_indiv = self.model.get_logits_loss_grad_dlr
            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.model.get_logits_loss_grad_target
            else:
                raise ValueError('unknowkn loss')
        
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)


        target_ind = list(range(0,self.args.t,self.args.t_interval))

        for _ in range(self.eot_iter):
            if not self.is_tf_model:
                with torch.enable_grad():
                    logits, mid_x, ori_x = self.model(x_adv, return_mid=True)
                    loss_indiv = self.mse_loss(mid_x, ori_x, target_ind)
                    if len(loss_indiv.shape)>3:
                        loss_indiv = loss_indiv.view(loss_indiv.shape[0],loss_indiv.shape[1],-1)

                    loss_indiv = torch.mean(loss_indiv, dim=-1)
                    loss_indiv = torch.mean(loss_indiv, dim=0).view(x.shape[0])
                    loss_ce = criterion_indiv(logits, y)
                    loss_indiv += loss_ce

                    loss = torch.mean(loss_indiv)



                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            else:
                if self.y_target is None:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y)
                else:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y,
                        self.y_target)
                grad += grad_curr
        
        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        if self.loss in ['dlr', 'dlr-targeted']:
            # check if there are zero gradients
            check_zero_gradients(grad, logger=self.logger)
        
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0

        loss_best = loss_ce.detach()
        # print(f'loss_best.shape: {loss_best.shape}')
        # loss_best = loss_indiv.detach().clone()

        alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2
        step_size = alpha * self.eps * torch.ones([x.shape[0], *(
            [1] * self.ndims)]).to(self.device).detach()
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        if self.norm == 'L1':
            k = max(int(.04 * self.n_iter), 1)
            n_fts = math.prod(self.orig_dim)
            if x_init is None:
                topk = .2 * torch.ones([x.shape[0]], device=self.device)
                sp_old =  n_fts * torch.ones_like(topk)
            else:
                topk = L0_norm(x_adv - x) / n_fts / 1.5
                sp_old = L0_norm(x_adv - x)
            #print(topk[0], sp_old[0])
            adasp_redstep = 1.5
            adasp_minstep = 10.
            #print(step_size[0].item())
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        n_reduced = 0

        n_fts = x.shape[-3] * x.shape[-2] * x.shape[-1]        
        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter):


            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()


                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                        x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(
                        x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                        x - self.eps), x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)

                elif self.norm == 'L1':
                    grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                    topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                    grad_topk = grad_topk[u, topk_curr].view(-1, *[1]*(len(x.shape) - 1))
                    sparsegrad = grad * (grad.abs() >= grad_topk).float()
                    x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                        L1_norm(sparsegrad.sign(), keepdim=True) + 1e-10)
                    
                    delta_u = x_adv_1 - x
                    delta_p = L1_projection(x, delta_u, self.eps)
                    x_adv_1 = x + delta_u + delta_p
                    
                    
                x_adv = x_adv_1 + 0.

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                if not self.is_tf_model:
                    with torch.enable_grad():


                        logits, mid_x, ori_x = self.model(x_adv, return_mid=True)
                        loss_indiv = self.mse_loss(mid_x, ori_x, target_ind)
                        if len(loss_indiv.shape) > 3:
                            loss_indiv = loss_indiv.view(loss_indiv.shape[0], loss_indiv.shape[1], -1)

                        loss_indiv = torch.mean(loss_indiv, dim=-1)
                        loss_indiv = torch.mean(loss_indiv, dim=0).view(x.shape[0])
                        loss_ce = criterion_indiv(logits, y)
                        loss_indiv += loss_ce

                        loss = torch.mean(loss_indiv)

    
                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()

                else:
                    assert 1==0, 'must not be tf_model'
            
            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y

            acc = pred


            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
            if self.verbose:
                logits, mid_x, ori_x = self.model(x_adv, return_mid=True)
                pred = logits.detach().max(1)[1] == y
                acc = pred
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                    step_size.mean(), topk.mean() * n_fts) if self.norm in ['L1'] else ''
                print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
                    i, loss_best.mean(), acc.float().mean(), str_stats))

            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1 + 0

              # can be further improved here
              # to use CW as validation loss
              ind = (y1 > loss_best).nonzero().squeeze()

              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0

              counter3 += 1

              if counter3 == k:
                  if self.norm in ['Linf', 'L2']:
                      fl_oscillation = self.check_oscillation(loss_steps, i, k,
                          loss_best, k3=self.thr_decr)
                      fl_reduce_no_impr = (1. - reduced_last_check) * (
                          loss_best_last_check >= loss_best).float()
                      fl_oscillation = torch.max(fl_oscillation,
                          fl_reduce_no_impr)
                      reduced_last_check = fl_oscillation.clone()
                      loss_best_last_check = loss_best.clone()
    
                      if fl_oscillation.sum() > 0:
                          ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                          step_size[ind_fl_osc] /= 2.0
                          n_reduced = fl_oscillation.sum()
    
                          x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                          grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                      k = max(k - self.size_decr, self.n_iter_min)
                  
                  elif self.norm == 'L1':
                      sp_curr = L0_norm(x_best - x)
                      fl_redtopk = (sp_curr / sp_old) < .95
                      topk = sp_curr / n_fts / 1.5
                      step_size[fl_redtopk] = alpha * self.eps
                      step_size[~fl_redtopk] /= adasp_redstep
                      step_size.clamp_(alpha * self.eps / adasp_minstep, alpha * self.eps)
                      sp_old = sp_curr.clone()
                  
                      x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                      grad[fl_redtopk] = grad_best[fl_redtopk].clone()
                  
                  counter3 = 0
                  #k = max(k - self.size_decr, self.n_iter_min)

        #
        
        return (x_best, acc, loss_best, x_best_adv)

    def perturb(self, x, y=None, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True the points attaining highest loss
                            are returned, otherwise adversarial examples
        """

        assert self.loss in ['ce', 'dlr'] #'ce-targeted-cfts'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if not self.is_tf_model:
            y_pred = self.model(x).max(1)[1]
        else:
            y_pred = self.model.predict(x).max(1)[1]
        if y is None:
            #y_pred = self.predict(x).max(1)[1]
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        if self.loss != 'ce-targeted':
            acc = y_pred == y
        else:
            acc = y_pred != y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        
        
        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1]) # make sure to use the given iterations
            if self.verbose:
                print('using schedule [{}x{}]'.format('+'.join([str(c
                    ) for c in epss]), '+'.join([str(c) for c in iters])))
        
        startt = time.time()
        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for counter in range(self.n_restarts):

                ind_to_fool = torch.ones_like(acc).nonzero().squeeze()
                # ind_to_fool = torch.nonzero(acc).squeeze()


                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()

                    # print(f'size of x_to_fool: {x_to_fool.shape}')
                    
                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()


                    acc[ind_to_fool[ind_curr]] = 0

                    ind_curr = list(range(adv_curr.shape[0]))


                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                            time.time() - startt))

            return adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(
                self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(
                        counter, loss_best.sum()))

            return adv_best

    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
        assert len(epss) == len(iters)
        assert self.norm in ['L1']
        self.use_rs = False
        if not use_rs:
            x_init = None
        else:
            x_init = x + torch.randn_like(x)
            x_init += L1_projection(x, x_init - x, 1. * float(epss[0]))
        eps_target = float(epss[-1])
        if self.verbose:
            print('total iter: {}'.format(sum(iters)))
        for eps, niter in zip(epss, iters):
            if self.verbose:
                print('using eps: {:.2f}'.format(eps))
            self.n_iter = niter + 0
            self.eps = eps + 0.
            #
            if not x_init is None:
                x_init += L1_projection(x, x_init - x, 1. * eps)
            x_init, acc, loss, x_adv = self.attack_single_run(x, y, x_init=x_init)

        return (x_init, acc, loss, x_adv)


class APGDDiffAttackPGDStep():
    """
    DiffAttack's proposal (diffusion-aware loss: CE + MSE on mid-layer) with
    a fixed-step PGD update instead of APGD (no step-size schedule, no momentum).
    When model does not support return_mid or args is None, falls back to CE-only.
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            seed=0,
            loss='ce',
            eot_iter=1,
            verbose=False,
            device=None,
            is_tf_model=False,
            logger=None,
            args=None):
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.verbose = verbose
        self.device = device
        self.is_tf_model = is_tf_model
        self.logger = logger
        self.args = args
        assert self.norm in ['Linf', 'L2', 'L1']
        assert self.eps is not None

    def init_hyperparam(self, x):
        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()
        # print(self.seed)

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except Exception:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)
        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def mse_loss(self, mid_x, ori_x, idx):
        mse = torch.nn.MSELoss(reduction='none')
        # mid_x / ori_x come from diffusion purification; their lengths depend on
        # the purifier settings. Be robust to mismatched/short lists.
        if not isinstance(mid_x, (list, tuple)) or not isinstance(ori_x, (list, tuple)):
            return torch.tensor(0.0)
        n = min(len(mid_x), len(ori_x))
        if n <= 0:
            return torch.tensor(0.0)
        idx = [int(i) for i in idx if int(i) >= 0 and int(i) < n]
        if len(idx) == 0:
            return torch.zeros((n,), device=mid_x[0].device, dtype=mid_x[0].dtype)[:0]
        mid_x = torch.stack(list(mid_x)[:n])
        ori_x = torch.stack(list(ori_x)[:n])
        loss = mse(mid_x[idx], ori_x[idx].view(mid_x[idx].shape))
        return loss

    def _compute_grad_and_loss(self, x_adv, x, y, criterion_indiv):
        """Same gradient as APGDDiffAttack (return_mid + mse + CE when available), else CE-only."""
        grad = torch.zeros_like(x_adv)
        loss_ce = None
        use_proposal = (
            self.args is not None
            and getattr(self.args, 't', None) is not None
            and getattr(self.args, 't_interval', None) is not None
        )
        for _ in range(self.eot_iter):
            if self.is_tf_model:
                raise NotImplementedError('APGDDiffAttackPGDStep is PyTorch only')
            with torch.enable_grad():
                if use_proposal:
                    out = self.model(x_adv, return_mid=True)
                    if not (isinstance(out, (tuple, list)) and len(out) >= 3):
                        raise RuntimeError(
                            "DiffAttack proposal loss requires model(x, return_mid=True) to return "
                            "(logits, mid_x, ori_x). The defense does not support return_mid or returned "
                            "an unexpected value; the core DiffAttack algorithm cannot run."
                        )
                    logits, mid_x, ori_x = out[0], out[1], out[2]
                    target_ind = list(range(0, self.args.t, self.args.t_interval))
                    loss_indiv = self.mse_loss(mid_x, ori_x, target_ind)
                    if len(loss_indiv.shape) > 3:
                        loss_indiv = loss_indiv.view(loss_indiv.shape[0], loss_indiv.shape[1], -1)
                    loss_indiv = torch.mean(loss_indiv, dim=-1)
                    loss_indiv = torch.mean(loss_indiv, dim=0).view(x.shape[0])
                    loss_ce = criterion_indiv(logits, y)
                    loss_indiv = loss_indiv + loss_ce
                    loss = torch.mean(loss_indiv)
                    # print(loss)
                    grad = grad + torch.autograd.grad(loss, x_adv, retain_graph=False)[0].detach()
                else:
                    logits = self.model(x_adv)
                    loss_ce = criterion_indiv(logits, y)
                    loss = torch.mean(loss_ce)
                    grad = grad + torch.autograd.grad(loss, x_adv, retain_graph=False)[0].detach()
        grad = grad / float(self.eot_iter)
        if loss_ce is None:
            with torch.no_grad():
                logits = self.model(x_adv)
                loss_ce = criterion_indiv(logits, y)
        return grad, loss_ce

    def attack_single_run(self, x, y, x_init=None):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.eps * torch.ones_like(x).detach() * self.normalize(t)
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x).detach() * self.normalize(t)
        else:
            t = torch.randn(x.shape).to(self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta
        if x_init is not None:
            x_adv = x_init.clone()
        x_adv = x_adv.clamp(0., 1.)
        x_best_adv = x_adv.clone()
        criterion_indiv = nn.CrossEntropyLoss(reduction='none')
        alpha = (self.eps / 4.0) if self.norm == 'Linf' else (self.eps / 3.0)
        best_loss = None
        for i in range(self.n_iter):
            x_adv = x_adv.detach().requires_grad_(True)
            grad, loss_ce = self._compute_grad_and_loss(x_adv, x, y, criterion_indiv)
            with torch.no_grad():
                if self.norm == 'Linf':
                    x_adv = x_adv + alpha * torch.sign(grad)
                    x_adv = torch.clamp(torch.clamp(x_adv, x - self.eps, x + self.eps), 0.0, 1.0)
                elif self.norm == 'L2':
                    g_flat = grad.view(x.shape[0], -1)
                    g_norm = torch.norm(g_flat, p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    x_adv = x_adv + alpha * (grad / g_norm.view(-1, *([1] * self.ndims)))
                    delta = x_adv - x
                    delta_flat = delta.view(x.shape[0], -1)
                    n = torch.norm(delta_flat, p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    factor = torch.clamp(self.eps / n, max=1.0)
                    x_adv = x + (delta_flat * factor).view_as(delta)
                    x_adv = x_adv.clamp(0.0, 1.0)
                else:
                    sparsegrad = grad.sign()
                    x_adv = x_adv + alpha * sparsegrad / (L1_norm(sparsegrad, keepdim=True) + 1e-10)
                    delta = x_adv - x
                    x_adv = x + delta + L1_projection(x, delta, self.eps)
                    x_adv = x_adv.clamp(0.0, 1.0)
                if best_loss is None:
                    best_loss = loss_ce.detach().clone()
                loss_curr = criterion_indiv(self.model(x_adv), y).detach()
                improved = (loss_curr > best_loss)
                best_loss = torch.where(improved, loss_curr, best_loss)
                x_best_adv = torch.where(improved.view(-1, 1, 1, 1), x_adv, x_best_adv)
        with torch.no_grad():
            acc = (self.model(x_best_adv).argmax(1) == y)
        return (x_adv, acc, best_loss, x_best_adv)

    def perturb(self, x, y=None):
        if y is not None and len(y.shape) == 0:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        self.init_hyperparam(x)
        x = x.detach().clone().float().to(self.device)
        y_pred = self.model(x).max(1)[1]
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)
        adv = x.clone()
        acc = (y_pred == y)
        for counter in range(self.n_restarts):
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()
                _, acc_curr, _, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                adv[ind_to_fool] = adv_curr.clone()
                acc[ind_to_fool] = acc_curr
        return adv


class PGDDiffAttack():
    """
    Standard PGD (fixed step size, no schedule) for consistency when using
    DiffAttack with a PGD backbone instead of APGD. Uses only model(x) -> logits
    and CE loss (no diffusion mid-layer loss).
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            seed=0,
            loss='ce',
            eot_iter=1,
            verbose=False,
            device=None,
    ):
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.verbose = verbose
        self.device = device
        assert self.norm in ['Linf', 'L2', 'L1']
        assert self.eps is not None

    def init_hyperparam(self, x):
        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
        elif self.norm == 'L1':
            t = x.abs().view(x.shape[0], -1).sum(dim=-1)
        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def attack_single_run(self, x, y):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        if self.norm == 'Linf':
            x_adv = x + self.eps * (2 * torch.rand(x.shape).to(self.device).detach() - 1)
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * self.normalize(t)
        else:
            t = torch.randn(x.shape).to(self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta
        x_adv = x_adv.clamp(0., 1.)
        x_best_adv = x_adv.clone()
        criterion = nn.CrossEntropyLoss(reduction='none')
        alpha = (self.eps / 4.0) if self.norm == 'Linf' else (self.eps / 3.0)
        best_loss = criterion(self.model(x_best_adv), y).detach()
        for i in range(self.n_iter):
            x_adv = x_adv.detach().requires_grad_(True)
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                logits = self.model(x_adv)
                loss_indiv = criterion(logits, y)
                loss = loss_indiv.sum()
                grad = grad + torch.autograd.grad(loss, x_adv, retain_graph=False)[0].detach()
            grad = grad / float(self.eot_iter)
            with torch.no_grad():
                if self.norm == 'Linf':
                    x_adv = x_adv + alpha * grad.sign()
                else:
                    g_flat = grad.view(x.shape[0], -1)
                    g_norm = torch.norm(g_flat, p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    x_adv = x_adv + alpha * (grad / g_norm.view(-1, *([1] * self.ndims)))
                delta = x_adv - x
                if self.norm == 'Linf':
                    x_adv = x + delta.clamp(-self.eps, self.eps)
                elif self.norm == 'L2':
                    delta_flat = delta.view(x.shape[0], -1)
                    n = torch.norm(delta_flat, p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    factor = torch.clamp(self.eps / n, max=1.0)
                    x_adv = x + (delta_flat * factor).view_as(delta)
                else:
                    x_adv = x + L1_projection(x, delta, self.eps)
                x_adv = x_adv.clamp(0., 1.)
                loss_curr = criterion(self.model(x_adv), y).detach()
                improved = (loss_curr > best_loss)
                best_loss = torch.where(improved, loss_curr, best_loss)
                x_best_adv = torch.where(improved.view(-1, 1, 1, 1), x_adv, x_best_adv)
        with torch.no_grad():
            acc = (self.model(x_best_adv).argmax(1) == y)
        return (x_best_adv, acc, best_loss, x_best_adv)

    def perturb(self, x, y=None):
        if y is not None and len(y.shape) == 0:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        self.init_hyperparam(x)
        x = x.detach().clone().float().to(self.device)
        y_pred = self.model(x).max(1)[1]
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)
        adv = x.clone()
        acc = (y_pred == y)
        for counter in range(self.n_restarts):
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()
                _, acc_curr, _, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                adv[ind_to_fool] = adv_curr.clone()
                acc[ind_to_fool] = acc_curr
        return adv


class APGDAttack_targeted(APGDDiffAttack):
    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            seed=0,
            eot_iter=1,
            rho=.75,
            topk=None,
            n_target_classes=9,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None):
        """
        AutoPGD on the targeted DLR loss
        """
        super(APGDAttack_targeted, self).__init__(predict, n_iter=n_iter, norm=norm,
            n_restarts=n_restarts, eps=eps, seed=seed, loss='dlr-targeted',
            eot_iter=eot_iter, rho=rho, topk=topk, verbose=verbose, device=device,
            use_largereps=use_largereps, is_tf_model=is_tf_model, logger=logger)

        self.y_target = None
        self.n_target_classes = n_target_classes

    def dlr_loss_targeted(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x[u, self.y_target]) / (x_sorted[:, -1] - .5 * (
            x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)

    def ce_loss_targeted(self, x, y):
        return -1. * F.cross_entropy(x, self.y_target, reduction='none')
    
    
    def perturb(self, x, y=None, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        """

        assert self.loss in ['dlr-targeted'] #'ce-targeted'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if not self.is_tf_model:
            y_pred = self.model(x).max(1)[1]
        else:
            y_pred = self.model.predict(x).max(1)[1]
        if y is None:
            #y_pred = self._get_predicted_label(x)
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        acc = y_pred == y
        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        #
        
        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1])
            if self.verbose:
                print('using schedule [{}x{}]'.format('+'.join([str(c
                    ) for c in epss]), '+'.join([str(c) for c in iters])))
        
        for target_class in range(2, self.n_target_classes + 2):
            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    
                    if not self.is_tf_model:
                        output = self.model(x_to_fool)
                    else:
                        output = self.model.predict(x_to_fool)
                    self.y_target = output.sort(dim=1)[1][:, -target_class]

                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('target class {}'.format(target_class),
                            '- restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                            time.time() - startt))

        return adv

