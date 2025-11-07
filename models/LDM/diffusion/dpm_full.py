import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm

from data.bioparse import VOCAB, const
from torch.autograd import grad
from torch_scatter import scatter_mean, scatter_sum, scatter_std

from utils.gnn_utils import variadic_meshgrid, length_to_batch_id
from utils.nn_utils import SinusoidalTimeEmbeddings

from ...modules.GET.tools import fully_connect_edges, knn_edges
from .transition import construct_transition
from ...modules.create_net import create_net
from ...modules.nn import MLP
import torch.autograd as autograd
import math

def low_trianguler_inv(L):
    # L: [bs, 3, 3]
    L_inv = torch.linalg.solve_triangular(L, torch.eye(3).unsqueeze(0).expand_as(L).to(L.device), upper=False)
    return L_inv


class EpsilonNet(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            encoder_type='EPT',
            opt={ 'n_layers': 3 }
        ):
        super().__init__()
        
        edge_embed_size = hidden_size // 4
        self.input_mlp = MLP(
            input_size + hidden_size * 2, # latent variable, cond embedding, time embedding
            hidden_size, hidden_size, 3
        )
        self.encoder = create_net(encoder_type, hidden_size, edge_embed_size, opt)
        self.hidden2input = nn.Linear(hidden_size, input_size)
        self.edge_embedding = nn.Embedding(2, edge_embed_size)
        self.time_embedding = SinusoidalTimeEmbeddings(hidden_size)

    def forward(
            self,
            H_noisy,
            X_noisy,
            cond_embedding,
            edges,
            edge_types,
            generate_mask,
            batch_ids,
            beta,
        ):
        """
        Args:
            H_noisy: (N, hidden_size)
            X_noisy: (N, 3)
            generate_mask: (N)
            batch_ids: (N)
            beta: (N)
        Returns:
            eps_H: (N, hidden_size)
            eps_X: (N, 3)
        """
        t_embed = self.time_embedding(beta)
        in_feat = torch.cat([H_noisy, cond_embedding, t_embed], dim=-1)
        in_feat = self.input_mlp(in_feat)
        edge_embed = self.edge_embedding(edge_types)
        block_ids = torch.arange(in_feat.shape[0], device=in_feat.device)
        
        next_H, next_X = self.encoder(in_feat, X_noisy, block_ids, batch_ids, edges, edge_embed)

        # equivariant vector features changes
        eps_X = next_X - X_noisy
        eps_X = torch.where(generate_mask[:, None].expand_as(eps_X), eps_X, torch.zeros_like(eps_X)) 

        # invariant scalar features changes
        next_H = self.hidden2input(next_H)
        eps_H = next_H - H_noisy
        eps_H = torch.where(generate_mask[:, None].expand_as(eps_H), eps_H, torch.zeros_like(eps_H))

        return eps_H, eps_X


class FullDPM(nn.Module):

    def __init__(
        self, 
        latent_size,
        hidden_size,
        num_steps, 
        trans_pos_type='Diffusion',
        trans_seq_type='Diffusion',
        encoder_type='EPT',
        lpl_loss=0.0,
        trans_pos_opt={}, 
        trans_seq_opt={},
        encoder_opt={},
    ):
        super().__init__()
        self.eps_net = EpsilonNet(latent_size, hidden_size, encoder_type, encoder_opt)
        self.num_steps = num_steps
        self.trans_x = construct_transition(trans_pos_type, num_steps, trans_pos_opt)
        self.trans_h = construct_transition(trans_seq_type, num_steps, trans_seq_opt)
        self.lpl_loss = lpl_loss
        

    @torch.no_grad()
    def _get_edges(self, chain_ids, batch_ids, lengths):
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        ) # (row, col)
        
        is_ctx = chain_ids[row] == chain_ids[col]
        is_inter = ~is_ctx
        ctx_edges = torch.stack([row[is_ctx], col[is_ctx]], dim=0) # [2, Ec]
        inter_edges = torch.stack([row[is_inter], col[is_inter]], dim=0) # [2, Ei]
        edges = torch.cat([ctx_edges, inter_edges], dim=-1)
        edge_types = torch.cat([torch.zeros_like(ctx_edges[0]), torch.ones_like(inter_edges[0])], dim=0)
        return edges, edge_types
    
    def forward(
            self,
            H_0,                # [Nblock, latent size]
            X_0,                # [Nblock, 3]
            cond_embedding,     # [Nblock, hidden size], conditional embedding
            chain_ids,          # [Nblock]
            generate_mask,      # [Nblock]
            lengths,            # [batch size]
            t=None,
            vae_model=None
        ):
        # if L is not None:
        #     L = L / self.std
        batch_ids = length_to_batch_id(lengths)
        batch_size = batch_ids.max() + 1
        if t == None: # sample time step
            t = torch.randint(0, self.num_steps + 1, (batch_size,), dtype=torch.long, device=H_0.device)
        X_noisy, eps_X = self.trans_x.add_noise(X_0, generate_mask, batch_ids, t)
        H_noisy, eps_H = self.trans_h.add_noise(H_0, generate_mask, batch_ids, t)

        edges, edge_types = self._get_edges(chain_ids, batch_ids, lengths)

        beta = self.trans_x.get_timestamp(t)[batch_ids]  # [N]
        eps_H_pred, eps_X_pred = self.eps_net(
            H_noisy, X_noisy, cond_embedding, edges, edge_types, generate_mask, batch_ids, beta
        )

        def normalize_per_channel(feat, batch_ids, mask):
            # feat: [N, C] or [N, 3, C] (we handle [N,C] here)
            feat_ = feat.view(feat.shape[0], -1)
            mean = scatter_mean(feat_[mask], batch_ids[mask], dim=0, dim_size=batch_ids.max() + 1)
            std = scatter_std(feat_[mask], batch_ids[mask], dim=0, dim_size=batch_ids.max() + 1)

            return mean[batch_ids].detach(), std[batch_ids].detach()

        def normalize_with_stats(feat, stats):
            mean, std = stats
            if feat.dim() == 2:
                return (feat - mean) / (std + 1e-4)
            else:
                return (feat - mean.view(-1, feat.shape[1], feat.shape[2])) / (std.view(-1, feat.shape[1], feat.shape[2]) + 1e-4)


        loss_dict = {}
        if self.lpl_loss:
            vae_model.eval()
            #with torch.enable_grad():
            H_0_map, X_0_map = vae_model.extract_decoder_feature(H_0, X_0, chain_ids, lengths)
            pred_h0, pred_x0 = self.trans_h._predict_p0_from_eps(H_noisy, eps_H_pred, generate_mask, batch_ids, t), self.trans_x._predict_p0_from_eps(X_noisy, eps_X_pred, generate_mask, batch_ids, t)
            H_pred_map, X_pred_map = vae_model.extract_decoder_feature(pred_h0, pred_x0, chain_ids, lengths)
            

            alpha_bar = self.trans_x.var_sched.alpha_bars[t][batch_ids][lpl_mask].unsqueeze(-1)
            snr = alpha_bar / (1 - alpha_bar + 1e-8)

            lpl_mask = generate_mask & (snr > 6)

            loss_H_f = []
            for l, (f_h, f_g) in enumerate(zip(H_0_map, H_pred_map)):    
                
                stats = normalize_per_channel(f_g, batch_ids, lpl_mask)
                f_h_norm = normalize_with_stats(f_h, stats)
                f_g_norm = normalize_with_stats(f_g, stats)
                loss_h_f_log = (torch.log1p(F.mse_loss(f_h_norm[lpl_mask], f_g_norm[lpl_mask], reduction='none') * alpha_bar).mean(dim=1))
                
                loss_H_f.append(loss_h_f_log.mean())
            
            loss_X_f = []
            for l, (f_h, f_g) in enumerate(zip(X_0_map, X_pred_map)):

                stats = normalize_per_channel(f_g, batch_ids, lpl_mask)
                f_h_norm = normalize_with_stats(f_h, stats)
                f_g_norm = normalize_with_stats(f_g, stats)
                loss_x_f_log = (torch.log1p(F.mse_loss(f_h_norm[lpl_mask], f_g_norm[lpl_mask], reduction='none') * alpha_bar.unsqueeze(-1)).mean(dim=[1,2]))
                
                loss_X_f.append(loss_x_f_log.mean())

            loss_dict['X_feat_lpl'] = torch.stack(loss_X_f).mean()
            loss_dict['H_feat_lpl'] = torch.stack(loss_H_f).mean()

        # equivariant vector feature loss
        loss_X = F.mse_loss(eps_X_pred[generate_mask], eps_X[generate_mask], reduction='none').sum(dim=-1)  # (Ntgt * n_latent_channel)
        loss_X = loss_X.sum() / (generate_mask.sum().float() + 1e-8)
        loss_dict['X'] = loss_X

        # invariant scalar feature loss
        loss_H = F.mse_loss(eps_H_pred[generate_mask], eps_H[generate_mask], reduction='none').sum(dim=-1)  # [N]
        loss_H = loss_H.sum() / (generate_mask.sum().float() + 1e-8)
        loss_dict['H'] = loss_H

        return loss_dict

    #@torch.no_grad()
    def sample(
            self,
            H,
            X,
            S,
            cond_embedding,
            chain_ids,
            batch_ids,
            is_aa,
            generate_mask,
            lengths,
            vae_model,
            guidance_scale,
            max_grad_norm,
            temperature,
            pbar=False,    
        ):
        """
        Args:
            H: contextual hidden states, (N, latent_size)
            X: contextual atomic coordinates, (N, 14, 3)
            L: cholesky decomposition of the covariance matrix \Sigma=LL^T, (bs, 3, 3)
        """
        batch_ids = length_to_batch_id(lengths)
        batch_size = lengths.shape[0]

        # Set the orientation and position of residues to be predicted to random values
        X_rand = torch.randn_like(X) # [N, 3]
        X_init = torch.where(generate_mask[:, None].expand_as(X), X_rand, X)

        H_rand = torch.randn_like(H)
        H_init = torch.where(generate_mask[:, None].expand_as(H), H_rand, H)

        traj = {self.num_steps: (X_init, H_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x

        edges, edge_types = self._get_edges(chain_ids, batch_ids, lengths)

        vae_model.eval()
        
        eps_freq, prob_thresh = 1e-6, 1e-3            # 防止 log(0)
        bb_mat = torch.tensor(VOCAB.freq_mat, dtype=torch.float).to(batch_ids.device)

        for t in pbar(range(self.num_steps, 0, -1)):
            with torch.enable_grad():
                X_t, H_t = traj[t]
                X_t, H_t = torch.round(X_t, decimals=4), torch.round(H_t, decimals=4) # reduce numerical error
                
                beta = self.trans_x.get_timestamp(t).view(1).repeat(X_t.shape[0])
                t_tensor = torch.full([X_t.shape[0], ], fill_value=t, dtype=torch.long, device=X_t.device)

                eps_H, eps_X = self.eps_net(
                    H_t, X_t, cond_embedding, edges, edge_types, generate_mask, batch_ids, beta
                )

                # TODO
                eps_H.requires_grad_(True)
                eps_X.requires_grad_(True)

                pred_H0 = self.trans_h._predict_p0_from_eps(H_t, eps_H, generate_mask, batch_ids, t_tensor)
                pred_X0 = self.trans_x._predict_p0_from_eps(X_t, eps_X, generate_mask, batch_ids, t_tensor)
                pred_block_prob, pred_block_pos = vae_model.generate_block_type(pred_H0, pred_X0, chain_ids, lengths, is_aa)
                pred_block_prob = pred_block_prob.clone()
                pred_block_prob[~generate_mask] = F.one_hot(S, num_classes=pred_block_prob.shape[1])[~generate_mask].float()
                
                row, col, _ = self._get_inter_block_nbh(pred_block_pos, batch_ids, generate_mask)
                
                # pred_block_prob: [Nblock, T]  (soft probs over block types)
                # pred_block_pos:  [Nblock, 3]  (block centers in latent/atom coords used for distance)
                # [row, col]:     [2, E]
                # dfactor:         [E] (weights in (0.5,2])
                # bb_mat:     [T, T] (频率矩阵，值在 [0,1])

                p_row = pred_block_prob[row]   # [E, T]
                p_col = pred_block_prob[col]   # [E, T]
                p_row = (p_row * (p_row > prob_thresh)).float()
                p_col = (p_col * (p_col > prob_thresh)).float()

                freq = (torch.matmul(p_row, bb_mat) * p_col).sum(dim=-1).clamp_min(eps_freq) 

                # smoothing / temperature
                if temperature != 1.0:
                    freq = freq ** (1.0 / temperature)

                edge_energy = -torch.log(freq)
                E_total = scatter_mean(edge_energy, batch_ids[row]).sum()
            
                grad_eps_H, grad_eps_X = autograd.grad(E_total, (eps_H, eps_X))
                
            with torch.no_grad():

                def clip_by_norm(g, max_norm):
                    g_view = g.view(g.shape[0], -1)
                    norms = g_view.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    factor = (max_norm / norms).clamp(max=1.0)
                    return g * factor.view(-1, *([1] * (g.dim()-1)))
                
                grad_eps_H = clip_by_norm(grad_eps_H, max_grad_norm)
                grad_eps_X = clip_by_norm(grad_eps_X, max_grad_norm)
                
                H_next_guided = self.trans_h.denoise(H_t, eps_H, generate_mask, batch_ids, t_tensor, guidance=grad_eps_H, guidance_weight=guidance_scale)
                X_next_guided = self.trans_x.denoise(X_t, eps_X, generate_mask, batch_ids, t_tensor, guidance=grad_eps_X, guidance_weight=guidance_scale)

                H_next = H_next_guided
                X_next = X_next_guided
                
                traj[t-1] = (X_next, H_next)
                traj[t] = (traj[t][0].cpu(), traj[t][1].cpu()) # Move previous states to cpu memory.
        #print('\nover\n')
        #exit()

        return traj

    @torch.no_grad()
    def _get_inter_block_nbh(self, X_t, batch_ids, generate_mask, dist_th = 8):
        # local neighborhood for negative bonds
        row, col = fully_connect_edges(batch_ids)

        # inter-block and at least one end is in topo generation part, and row < col to avoid repeated bonds
        select_mask = (generate_mask[row] | generate_mask[col]) * (row < col)
        row, col = row[select_mask], col[select_mask]

        #distance = torch.norm(X_t[row] - X_t[col], dim=-1)
        #dfactor = torch.clamp(1 / distance, max=2.0)
        #select_mask = distance < dist_th
        #row, col, dfactor = row[select_mask], col[select_mask], dfactor[select_mask]

        return row, col, None

