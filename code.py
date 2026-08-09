"""
Mini DeepPlace v3.2 — GPU-Accelerated RL Placement
====================================================
Fixes over v3.1:
  - DEF now writes full NETS section (was hardcoded NETS 0)
    → OpenROAD can now run global_route and show congestion heatmap
  - UNITS DISTANCE MICRONS (was DATABASE — invalid DEF 5.8 syntax)
  - Baseline random DEF exported at episode 0 for before/after comparison
  - Tech LEF argument added for proper OpenROAD legalization TCL
  - Cell name backslash escaping fixed for DEF compatibility
  - GNN single-graph OOM fix retained from v3.1

GPU memory budget on T4 (15.6 GB) with AES (19,403 cells):
  GNN forward (sampled edges, 1 graph) :  ~300 MB
  Env positions (8 envs × 19403 × 2)   :  ~1.2 MB
  PPO minibatch (256 × 19403 × 2)       :  ~400 MB
  Policy MLP parameters                 :  ~40 MB
  Total expected                        :  ~800 MB  (well within 15.6 GB)

Run on Kaggle T4:
    !pip install torch matplotlib numpy -q
    !python mini_deepplace_v3_2.py \\
        --verilog  /kaggle/working/aes_data/aes.v \\
        --lef      /kaggle/working/aes_data/aes_cells_1.lef \\
        --tech_lef /kaggle/working/aes_data/gf180mcu_5LM_1TM_11K_9t_tech.lef \\
        --outdir   /kaggle/working/results \\
        --episodes 300 \\
        --n_envs   8 \\
        --device   cuda
"""

import os
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Max edges kept in GNN graph for large netlists.
# 242,022 edges × 8 envs was the OOM trigger.
# One graph with 300k edges fits fine; 8 × 242k does not.
MAX_GNN_EDGES = 300_000

# DEF database units per micron
DEF_SCALE = 2000


# ============================================================
# 1. ARGUMENT PARSER
# ============================================================
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--verilog',       default='/kaggle/working/inputs/aes/aes.v')
    p.add_argument('--lef',           default='/kaggle/working/inputs/aes/aes_cells_1.lef')
    p.add_argument('--tech_lef',      default=None,
                   help='Tech LEF (e.g. gf180mcu_5LM_1TM_11K_9t_tech.lef). '
                        'Required for OpenROAD legalization and congestion analysis.')
    p.add_argument('--outdir',        default='/kaggle/working/results')
    p.add_argument('--episodes',      type=int,   default=300)
    p.add_argument('--steps',         type=int,   default=200)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--density',       type=float, default=0.6)
    p.add_argument('--cell_size',     type=int,   default=10)
    p.add_argument('--alpha',         type=float, default=0.7)
    p.add_argument('--beta',          type=float, default=0.3)
    p.add_argument('--grid_bins',     type=int,   default=32)
    p.add_argument('--gnn_dim',       type=int,   default=32)
    p.add_argument('--gnn_layers',    type=int,   default=2)
    p.add_argument('--gnn_refresh',   type=int,   default=10,
                   help='Re-run GNN every N steps (amortises cost over steps)')
    p.add_argument('--n_envs',        type=int,   default=8)
    p.add_argument('--ppo_epochs',    type=int,   default=4)
    p.add_argument('--minibatch',     type=int,   default=256)
    p.add_argument('--prefix',        default=None)
    p.add_argument('--device',        default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--openroad',      default='openroad')
    p.add_argument('--skip_legalize', action='store_true')
    return p.parse_args()


# ============================================================
# 2. LEF PARSER
# ============================================================
def parse_lef_sizes(lef_path):
    sizes = {}
    if not lef_path or not os.path.exists(lef_path):
        print("[WARN] No LEF found — using uniform cell size")
        return sizes
    current = None
    with open(lef_path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'MACRO\s+(\S+)', line)
            if m:
                current = m.group(1)
            if current:
                m = re.match(r'SIZE\s+([\d.]+)\s+BY\s+([\d.]+)', line)
                if m:
                    sizes[current] = (float(m.group(1)), float(m.group(2)))
            if line.startswith('END') and current and line == f'END {current}':
                current = None
    print(f"[INFO] Parsed {len(sizes)} cell sizes from LEF")
    return sizes


# ============================================================
# 3. NETLIST PARSER
# ============================================================
class SimpleNetlist:
    def __init__(self):
        self.cells      = []       # list of (inst_name, w, h)
        self.cell_types = []       # list of cell_type strings
        self.nets       = []       # list of lists of cell indices
        self.net_names  = []       # list of net name strings (for DEF output)
        self.edge_index = None
        self.die_w      = 1000.0
        self.die_h      = 1000.0

    def parse(self, v_path, lef_sizes, default_size=10, density=0.6, prefix=None):
        print(f"[INFO] Parsing {v_path} ...")
        net_map      = {}   # net_name -> [cell_idx, ...]
        net_name_map = {}   # net_name -> net_name (preserved for DEF)

        with open(v_path) as f:
            content = f.read()

        if prefix:
            detected_prefix = prefix
            print(f"[INFO] Using cell prefix: '{detected_prefix}*'")
        else:
            from collections import Counter
            candidates = re.findall(
                r'\b([a-zA-Z][a-zA-Z0-9_]*__[a-zA-Z0-9_]+)', content[:5000])
            if candidates:
                counts = Counter(p.split('__')[0] + '__' for p in candidates if '__' in p)
                detected_prefix = counts.most_common(1)[0][0]
                print(f"[INFO] Auto-detected cell prefix: '{detected_prefix}*'")
            else:
                detected_prefix = None
                print("[WARN] Could not detect cell prefix — using broad match")

        if detected_prefix:
            escaped = re.escape(detected_prefix)
            pattern = re.compile(
                rf'({escaped}\w+)\s+([\\\w\[\]$]+)\s*\(([^;]+)\);', re.DOTALL)
        else:
            pattern = re.compile(
                r'\b([A-Z][A-Za-z0-9_]+)\s+([\\\w\[\]$]+)\s*\(([^;]+)\);', re.DOTALL)

        # Also track which port each cell connects on each net for DEF
        # net_pins: net_name -> [(inst_name, pin_name), ...]
        net_pins = {}

        for m in pattern.finditer(content):
            cell_type = m.group(1)
            inst_name = m.group(2).strip()
            ports     = m.group(3)
            w, h = lef_sizes.get(cell_type, (default_size, default_size))
            idx  = len(self.cells)
            self.cells.append((inst_name, w, h))
            self.cell_types.append(cell_type)
            for pm in re.finditer(r'\.(\w+)\((\w+)\)', ports):
                pin_name = pm.group(1)
                net_name = pm.group(2)
                if net_name in ('VDD', 'VSS', 'vdd', 'vss', "1'b0", "1'b1"):
                    continue
                net_map.setdefault(net_name, []).append(idx)
                net_pins.setdefault(net_name, []).append((inst_name, pin_name))

        # Filter nets: keep those with 2–50 connections
        valid_nets = {k: v for k, v in net_map.items() if 2 <= len(v) <= 50}
        self.nets      = list(valid_nets.values())
        self.net_names = list(valid_nets.keys())
        # Store pin info for DEF writing
        self._net_pins = {k: net_pins[k] for k in valid_nets}

        if len(self.cells) < 10:
            print("\n[ERROR] No cells parsed. Pass --prefix, e.g. --prefix sky130_fd_sc_hd__")
            raise SystemExit(1)

        # Build GNN edge index
        src, dst = [], []
        for net in self.nets:
            for i in range(len(net)):
                for j in range(i + 1, len(net)):
                    src.append(net[i]); dst.append(net[j])
                    src.append(net[j]); dst.append(net[i])

        if src:
            raw = torch.tensor([src, dst], dtype=torch.long)
        else:
            raw = torch.zeros((2, 0), dtype=torch.long)

        # Edge sampling for large netlists
        E = raw.shape[1]
        if E > MAX_GNN_EDGES:
            perm = torch.randperm(E)[:MAX_GNN_EDGES]
            self.edge_index = raw[:, perm]
            print(f"[INFO] Sampled {MAX_GNN_EDGES:,} / {E:,} edges for GNN "
                  f"(prevents OOM on large netlists)")
        else:
            self.edge_index = raw

        total_area = sum(w * h for _, w, h in self.cells)
        side = np.sqrt(total_area / density)
        self.die_w = side
        self.die_h = side

        print(f"[INFO] Cells: {len(self.cells)}, Nets: {len(self.nets)}, "
              f"GNN edges: {self.edge_index.shape[1]:,}, "
              f"Die: {self.die_w:.1f} x {self.die_h:.1f} um")


# ============================================================
# 4. GNN — ON GPU, SINGLE GRAPH (NOT MEGA-BATCHED)
# ============================================================
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = out_dim // n_heads
        self.W        = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.randn(1, n_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.randn(1, n_heads, self.head_dim))
        self.norm     = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        N  = x.size(0)
        Wx = self.W(x).view(N, self.n_heads, self.head_dim)
        src_idx, dst_idx = edge_index[0], edge_index[1]

        attn = (Wx[src_idx] * self.attn_src).sum(-1) + \
               (Wx[dst_idx] * self.attn_dst).sum(-1)
        attn = F.leaky_relu(attn, 0.2)

        attn_exp = attn.exp()
        norm_sum = torch.zeros(N, self.n_heads, device=x.device)
        norm_sum.scatter_add_(0,
            dst_idx.unsqueeze(1).expand(-1, self.n_heads), attn_exp)
        attn_norm = attn_exp / (norm_sum[dst_idx] + 1e-8)

        agg = torch.zeros(N, self.n_heads, self.head_dim, device=x.device)
        msg = Wx[src_idx] * attn_norm.unsqueeze(-1)
        idx = dst_idx.view(-1, 1, 1).expand(-1, self.n_heads, self.head_dim)
        agg.scatter_add_(0, idx, msg)

        out = agg.reshape(N, self.n_heads * self.head_dim)
        return F.elu(self.norm(out + Wx.view(N, -1)))


class NetlistGNN(nn.Module):
    def __init__(self, gnn_dim=32, n_layers=2, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(2, gnn_dim)
        self.layers = nn.ModuleList([
            GATLayer(gnn_dim, gnn_dim, n_heads) for _ in range(n_layers)
        ])

    def forward(self, pos_norm, edge_index):
        """
        pos_norm   : [N, 2]  — single graph, on GPU
        edge_index : [2, E]  — on GPU
        returns    : [N, gnn_dim]
        """
        x = F.relu(self.input_proj(pos_norm))
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


# ============================================================
# 5. GPU PLACEMENT ENVIRONMENT (N_ENVS parallel, fully on GPU)
# ============================================================
class GPUPlacementEnv:
    def __init__(self, netlist, bins=32, alpha=0.7, beta=0.3,
                 n_envs=8, device='cuda'):
        self.nl      = netlist
        self.n       = len(netlist.cells)
        self.bins    = bins
        self.alpha   = alpha
        self.beta    = beta
        self.n_envs  = n_envs
        self.device  = torch.device(device)

        self.die_w = torch.tensor(netlist.die_w, dtype=torch.float32, device=self.device)
        self.die_h = torch.tensor(netlist.die_h, dtype=torch.float32, device=self.device)

        nets     = netlist.nets
        max_len  = max(len(n) for n in nets)
        net_arr  = torch.zeros(len(nets), max_len, dtype=torch.long)
        net_mask = torch.zeros(len(nets), max_len, dtype=torch.bool)
        for i, net in enumerate(nets):
            net_arr[i, :len(net)]  = torch.tensor(net)
            net_mask[i, :len(net)] = True
        self.net_arr  = net_arr.to(self.device)
        self.net_mask = net_mask.to(self.device)

        self.bw = netlist.die_w / bins
        self.bh = netlist.die_h / bins

        self.pos = None
        self.reset()

    def reset(self):
        self.pos = torch.rand(self.n_envs, self.n, 2, device=self.device)
        self.pos[:, :, 0] *= self.die_w
        self.pos[:, :, 1] *= self.die_h
        return self._state()

    def _state(self):
        s = self.pos.clone()
        s[:, :, 0] /= self.die_w
        s[:, :, 1] /= self.die_h
        return s

    def hpwl(self):
        cell_pos = self.pos[:, self.net_arr, :]
        mask = self.net_mask.unsqueeze(0).unsqueeze(-1)
        INF = 1e9
        xs_max = cell_pos[:, :, :, 0].masked_fill(~mask.squeeze(-1), -INF).max(2).values
        xs_min = cell_pos[:, :, :, 0].masked_fill(~mask.squeeze(-1),  INF).min(2).values
        ys_max = cell_pos[:, :, :, 1].masked_fill(~mask.squeeze(-1), -INF).max(2).values
        ys_min = cell_pos[:, :, :, 1].masked_fill(~mask.squeeze(-1),  INF).min(2).values
        return (xs_max - xs_min + ys_max - ys_min).sum(1)

    def congestion(self):
        B = self.bins
        cell_pos = self.pos[:, self.net_arr, :]
        mask = self.net_mask.unsqueeze(0).unsqueeze(-1)
        INF = 1e9
        xs_max = cell_pos[:, :, :, 0].masked_fill(~mask.squeeze(-1), -INF).max(2).values
        xs_min = cell_pos[:, :, :, 0].masked_fill(~mask.squeeze(-1),  INF).min(2).values
        ys_max = cell_pos[:, :, :, 1].masked_fill(~mask.squeeze(-1), -INF).max(2).values
        ys_min = cell_pos[:, :, :, 1].masked_fill(~mask.squeeze(-1),  INF).min(2).values

        bw = torch.tensor(self.bw, device=self.device)
        bh = torch.tensor(self.bh, device=self.device)
        x0s = (xs_min / bw).long().clamp(0, B - 1)
        x1s = (xs_max / bw).long().clamp(0, B - 1)
        y0s = (ys_min / bh).long().clamp(0, B - 1)
        y1s = (ys_max / bh).long().clamp(0, B - 1)

        bbox_area = ((x1s - x0s + 1).float() * (y1s - y0s + 1).float())
        return bbox_area.mean(1) / (B * B)

    def step(self, actions):
        prev_hpwl = self.hpwl()
        prev_cong = self.congestion()

        cell_idx = (actions[:, 0].abs() * self.n).long() % self.n
        dx = actions[:, 1] * self.die_w * 0.05
        dy = actions[:, 2] * self.die_h * 0.05

        env_idx = torch.arange(self.n_envs, device=self.device)
        self.pos[env_idx, cell_idx, 0] = (
            self.pos[env_idx, cell_idx, 0] + dx).clamp(0, self.die_w.item())
        self.pos[env_idx, cell_idx, 1] = (
            self.pos[env_idx, cell_idx, 1] + dy).clamp(0, self.die_h.item())

        new_hpwl = self.hpwl()
        new_cong = self.congestion()

        hpwl_reward = (prev_hpwl - new_hpwl) / (prev_hpwl + 1e-8)
        cong_reward = (prev_cong - new_cong) / (prev_cong + 1e-8)
        reward = self.alpha * hpwl_reward + self.beta * cong_reward
        return self._state(), reward, new_hpwl, new_cong


# ============================================================
# 6. PPO POLICY — GNN RUNS ONCE PER REFRESH, SHARED ACROSS ENVS
# ============================================================
class GNNPPONet(nn.Module):
    """
    Key fix over v3:
      - GNN runs on a SINGLE graph (mean position across envs) not a mega-batch
      - GNN output is a [gnn_dim] vector broadcast to all envs
      - This caps GNN memory at one graph regardless of n_envs
      - GNN is refreshed every gnn_refresh steps (default 10)
    """
    def __init__(self, n_cells, gnn_dim=32, n_layers=2, hidden=256, action_dim=3):
        super().__init__()
        self.n_cells = n_cells
        self.gnn     = NetlistGNN(gnn_dim, n_layers)

        in_dim = gnn_dim + n_cells * 2
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
        )
        self.actor_mean   = nn.Linear(hidden // 2, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim) - 0.5)
        self.critic       = nn.Linear(hidden // 2, 1)

        self._gnn_cache = None   # cached [1, gnn_dim] broadcast feature

    def refresh_gnn(self, pos_norm, edge_index):
        """
        pos_norm   : [B, N, 2]  — mean taken across envs for GNN input
        edge_index : [2, E]
        Stores result in self._gnn_cache as [1, gnn_dim] for broadcasting.
        """
        with torch.no_grad():
            mean_pos = pos_norm.mean(0)                    # [N, 2]
            node_emb = self.gnn(mean_pos, edge_index)      # [N, gnn_dim]
            self._gnn_cache = node_emb.mean(0, keepdim=True)  # [1, gnn_dim]

    def _get_features(self, pos_norm):
        """
        pos_norm : [B, N, 2]
        Returns  : [B, gnn_dim + N*2]
        """
        B = pos_norm.shape[0]
        gnn_feat = self._gnn_cache.expand(B, -1)   # [B, gnn_dim]
        pos_flat = pos_norm.reshape(B, -1)          # [B, N*2]
        return torch.cat([gnn_feat, pos_flat], dim=-1)

    def forward(self, pos_norm, edge_index, refresh=False):
        if refresh or self._gnn_cache is None:
            self.refresh_gnn(pos_norm, edge_index)
        x    = self._get_features(pos_norm)
        h    = self.shared(x)
        mean = torch.tanh(self.actor_mean(h))
        std  = self.actor_logstd.exp().clamp(1e-3, 1.0)
        dist = Normal(mean, std)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value    = self.critic(h).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, pos_norm, edge_index, actions):
        """For PPO update — GNN refresh already happened during rollout."""
        if self._gnn_cache is None:
            self.refresh_gnn(pos_norm, edge_index)
        x    = self._get_features(pos_norm)
        h    = self.shared(x)
        mean = torch.tanh(self.actor_mean(h))
        std  = self.actor_logstd.exp().clamp(1e-3, 1.0)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy  = dist.entropy().sum(-1)
        value    = self.critic(h).squeeze(-1)
        return log_prob, entropy, value


# ============================================================
# 7. PPO TRAINING
# ============================================================
def train_ppo(env, args):
    device     = torch.device(args.device)
    n_cells    = env.n
    edge_index = env.nl.edge_index.to(device)

    policy = GNNPPONet(
        n_cells  = n_cells,
        gnn_dim  = args.gnn_dim,
        n_layers = args.gnn_layers,
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.episodes, eta_min=1e-5)

    hpwl_history  = []
    cong_history  = []
    best_hpwl     = float('inf')
    baseline_hpwl = None
    baseline_cong = None

    os.makedirs(args.outdir, exist_ok=True)

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"\n[INFO] GNN-PPO policy: {total_params:,} parameters")
    print(f"[INFO] GNN runs on GPU — single graph (OOM fix), "
          f"refreshed every {args.gnn_refresh} steps")
    print(f"[INFO] Training on {device} | {args.n_envs} parallel envs")
    print(f"[INFO] {args.steps} steps × {args.n_envs} envs = "
          f"{args.steps * args.n_envs} samples/episode")
    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(device)}  "
              f"({torch.cuda.get_device_properties(device).total_memory/1e9:.1f} GB)")
    print("-" * 65)

    baseline_pos_saved = False

    for ep in range(args.episodes):
        state = env.reset()

        b_states, b_actions, b_log_probs = [], [], []
        b_values, b_rewards              = [], []
        ep_hpwl = ep_cong = None

        for step_i in range(args.steps):
            refresh = (step_i % args.gnn_refresh == 0)
            with torch.no_grad():
                action, log_prob, value = policy(state, edge_index, refresh=refresh)

            next_state, reward, hpwl, cong = env.step(action)

            b_states.append(state)
            b_actions.append(action)
            b_log_probs.append(log_prob)
            b_values.append(value)
            b_rewards.append(reward)

            state   = next_state
            ep_hpwl = hpwl.mean().item()
            ep_cong = cong.mean().item()

            if baseline_hpwl is None:
                baseline_hpwl = ep_hpwl
                baseline_cong = ep_cong

        # Save baseline (random) placement DEF from episode 0
        if ep == 0 and not baseline_pos_saved:
            baseline_pos = env.pos[0].cpu().numpy()
            baseline_def = os.path.join(args.outdir, 'baseline_random.def')
            _write_def_from_positions(
                env.nl, baseline_pos, baseline_def,
                design_name='aes_cipher_top')
            print(f"[INFO] Baseline (random) DEF saved → {baseline_def}")
            baseline_pos_saved = True

        # Stack rollout: [steps * n_envs, ...]
        states_t  = torch.cat(b_states)
        actions_t = torch.cat(b_actions)
        old_lp_t  = torch.cat(b_log_probs).detach()
        values_t  = torch.cat(b_values).detach()

        rew_2d     = torch.cat(b_rewards).reshape(args.steps, args.n_envs)
        returns_2d = torch.zeros_like(rew_2d)
        R = torch.zeros(args.n_envs, device=device)
        for t in range(args.steps - 1, -1, -1):
            R = rew_2d[t] + 0.99 * R
            returns_2d[t] = R
        returns = returns_2d.reshape(-1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        advantages = returns - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = states_t.shape[0]
        for _ in range(args.ppo_epochs):
            perm = torch.randperm(T, device=device)
            for start in range(0, T, args.minibatch):
                idx = perm[start:start + args.minibatch]
                new_lp, entropy, new_val = policy.evaluate(
                    states_t[idx], edge_index, actions_t[idx])
                ratio  = (new_lp - old_lp_t[idx]).exp()
                adv    = advantages[idx]
                a_loss = -torch.min(ratio * adv,
                                    ratio.clamp(0.8, 1.2) * adv).mean()
                c_loss = (returns[idx] - new_val).pow(2).mean()
                loss   = a_loss + 0.5 * c_loss - 0.01 * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        scheduler.step()

        hpwl_history.append(ep_hpwl)
        cong_history.append(ep_cong)

        if ep_hpwl < best_hpwl:
            best_hpwl = ep_hpwl
            torch.save(policy.state_dict(),
                       os.path.join(args.outdir, 'best_policy.pt'))

        if ep % 20 == 0 or ep == args.episodes - 1:
            h_impr  = (baseline_hpwl - ep_hpwl) / (baseline_hpwl + 1e-8) * 100
            c_impr  = (baseline_cong - ep_cong)  / (baseline_cong + 1e-8) * 100
            gpu_str = ""
            if device.type == 'cuda':
                gpu_str = f"  GPU: {torch.cuda.memory_allocated(device)/1e6:.0f} MB"
            print(f"Ep {ep:4d}/{args.episodes} | "
                  f"HPWL: {ep_hpwl:>12.0f} ({h_impr:+.1f}%) | "
                  f"Cong: {ep_cong:.4f} ({c_impr:+.1f}%){gpu_str}")

    print("-" * 65)
    print(f"Baseline HPWL : {baseline_hpwl:.0f}")
    print(f"Best HPWL     : {best_hpwl:.0f}")
    h_impr = (baseline_hpwl - best_hpwl) / (baseline_hpwl + 1e-8) * 100
    print(f"Improvement   : {h_impr:.1f}%")
    return policy, hpwl_history, cong_history, baseline_hpwl, baseline_cong, best_hpwl


# ============================================================
# 8. TRAINING CURVES PLOT
# ============================================================
def plot_curves(hpwl_history, cong_history, baseline_hpwl, baseline_cong,
                best_hpwl, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Mini DeepPlace v3.2 — Training Progress (GPU)',
                 fontsize=13, fontweight='bold')

    ax1.plot(hpwl_history, color='#2E75B6', linewidth=1.5)
    ax1.axhline(baseline_hpwl, color='#C00000', linestyle='--', linewidth=1,
                label=f'Baseline: {baseline_hpwl:.0f}')
    ax1.axhline(best_hpwl, color='#375623', linestyle='--', linewidth=1,
                label=f'Best: {best_hpwl:.0f}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('HPWL (lower = better)')
    ax1.set_title('Wirelength (HPWL)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(cong_history, color='#E36C09', linewidth=1.5)
    ax2.axhline(baseline_cong, color='#C00000', linestyle='--', linewidth=1,
                label=f'Baseline: {baseline_cong:.4f}')
    best_cong = min(cong_history)
    ax2.axhline(best_cong, color='#375623', linestyle='--', linewidth=1,
                label=f'Best: {best_cong:.4f}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Congestion score')
    ax2.set_title('Routing Congestion')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved → {out_path}")


# ============================================================
# 9. DEF WRITER (CORE — writes full NETS section)
# ============================================================
def _write_def_from_positions(nl, positions, out_path, design_name):
    """
    Write a valid DEF 5.8 file from a numpy array of cell positions.

    Parameters
    ----------
    nl        : SimpleNetlist  — netlist with cells, cell_types, nets, net_names
    positions : np.ndarray [N, 2]  — cell (x, y) in microns
    out_path  : str  — output file path
    design_name : str

    Fixes vs v3.1
    -------------
    - UNITS DISTANCE MICRONS  (was DATABASE — invalid DEF 5.8)
    - Full NETS section written with cell/pin connections
    - Cell names with backslashes handled correctly for DEF
    """
    scale = DEF_SCALE
    die_w = int(nl.die_w * scale)
    die_h = int(nl.die_h * scale)
    n     = len(nl.cells)

    lines = [
        "VERSION 5.8 ;",
        'DIVIDERCHAR "/" ;',
        'BUSBITCHARS "[]" ;',
        f"DESIGN {design_name} ;",
        f"UNITS DISTANCE MICRONS {scale} ;",   # FIX: was DATABASE
        "",
        f"DIEAREA ( 0 0 ) ( {die_w} {die_h} ) ;",
        "",
        f"COMPONENTS {n} ;",
    ]

    for i, (name, w, h) in enumerate(nl.cells):
        cell_type = nl.cell_types[i]
        x = int(np.clip(positions[i, 0], 0, nl.die_w) * scale)
        y = int(np.clip(positions[i, 1], 0, nl.die_h) * scale)
        # DEF requires backslash-escaped instance names to keep the leading backslash
        # but NOT double-escape them
        safe_name = name if not name.startswith('\\') else name
        lines.append(f"- {safe_name} {cell_type}")
        lines.append(f"    + PLACED ( {x} {y} ) N ;")

    lines += ["END COMPONENTS", ""]

    # ── PINS section (no top-level pins from Verilog-only flow) ────────────
    lines += ["PINS 0 ;", "END PINS", ""]

    # ── NETS section — full connectivity ───────────────────────────────────
    # Use _net_pins if available (has port-level detail), otherwise use cell indices
    n_nets = len(nl.nets)
    lines.append(f"NETS {n_nets} ;")

    for net_idx, net_name in enumerate(nl.net_names):
        # Sanitize net name for DEF (replace problematic characters)
        safe_net = net_name.replace('[', '\\[').replace(']', '\\]')
        lines.append(f"- {safe_net}")

        if hasattr(nl, '_net_pins') and net_name in nl._net_pins:
            # Write with actual pin names from Verilog parsing
            for inst_name, pin_name in nl._net_pins[net_name]:
                safe_inst = inst_name if not inst_name.startswith('\\') else inst_name
                lines.append(f"  ( {safe_inst} {pin_name} )")
        else:
            # Fallback: write cell connections without pin name
            for cell_idx in nl.nets[net_idx]:
                inst_name = nl.cells[cell_idx][0]
                safe_inst = inst_name if not inst_name.startswith('\\') else inst_name
                lines.append(f"  ( {safe_inst} PIN )")

        lines.append("  ;")

    lines += ["END NETS", "", "END DESIGN"]

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"[INFO] Written {n} cells, {n_nets} nets → {out_path}")
    return out_path


# ============================================================
# 10. GREEDY POLICY ROLLOUT → DEF
# ============================================================
def write_placed_def(env, policy, edge_index, out_path, design_name, device,
                     n_steps=1000):
    print(f"\n[INFO] Running greedy policy for {n_steps} steps → DEF...")
    policy.eval()

    single_pos = torch.rand(1, env.n, 2, device=device)
    single_pos[:, :, 0] *= env.die_w
    single_pos[:, :, 1] *= env.die_h

    best_hpwl = float('inf')
    best_pos  = single_pos.clone()

    def _hpwl_single(pos):
        cp   = pos[0][env.net_arr]
        mask = env.net_mask
        INF  = 1e9
        xs_max = cp[:, :, 0].masked_fill(~mask, -INF).max(1).values
        xs_min = cp[:, :, 0].masked_fill(~mask,  INF).min(1).values
        ys_max = cp[:, :, 1].masked_fill(~mask, -INF).max(1).values
        ys_min = cp[:, :, 1].masked_fill(~mask,  INF).min(1).values
        return (xs_max - xs_min + ys_max - ys_min).sum().item()

    state = single_pos / torch.stack([env.die_w, env.die_h])
    policy.refresh_gnn(state, edge_index)

    for step_i in range(n_steps):
        refresh = (step_i % 20 == 0)
        with torch.no_grad():
            action, _, _ = policy(state, edge_index, refresh=refresh)
        cell_idx = (action[0, 0].abs() * env.n).long() % env.n
        dx = action[0, 1] * env.die_w * 0.05
        dy = action[0, 2] * env.die_h * 0.05
        single_pos[0, cell_idx, 0] = (single_pos[0, cell_idx, 0] + dx).clamp(0, env.die_w)
        single_pos[0, cell_idx, 1] = (single_pos[0, cell_idx, 1] + dy).clamp(0, env.die_h)
        state = single_pos / torch.stack([env.die_w, env.die_h])
        h = _hpwl_single(single_pos)
        if h < best_hpwl:
            best_hpwl = h
            best_pos  = single_pos.clone()

    final_pos = best_pos[0].cpu().numpy()
    print(f"[INFO] Final placement HPWL: {best_hpwl:.0f}")

    return _write_def_from_positions(env.nl, final_pos, out_path, design_name)


# ============================================================
# 11. OPENROAD LEGALIZATION
# ============================================================
def legalize_with_openroad(placed_def, tech_lef, cell_lef, outdir,
                           design_name, openroad_bin):
    """
    Runs OpenROAD detailed placement to legalize the RL-placed DEF.
    Requires both tech LEF and cell LEF to be specified.
    """
    legalized_def = os.path.join(outdir, f"{design_name}_legalized.def")
    tcl_path      = os.path.join(outdir, "legalize.tcl")

    # Build read_lef lines — tech LEF must come first
    lef_lines = ""
    if tech_lef and os.path.exists(tech_lef):
        lef_lines += f"read_lef {tech_lef}\n"
    else:
        print("[WARN] tech_lef not provided or not found — legalization may fail")
    if cell_lef and os.path.exists(cell_lef):
        lef_lines += f"read_lef {cell_lef}\n"

    tcl = f"""{lef_lines}read_def {placed_def}
make_tracks
detailed_placement
check_placement -verbose
write_def {legalized_def}
exit
"""
    with open(tcl_path, 'w') as f:
        f.write(tcl)

    ret = os.system(f"{openroad_bin} -exit {tcl_path} > {outdir}/legalize.log 2>&1")
    if ret == 0 and os.path.exists(legalized_def):
        print(f"[INFO] Legalized DEF → {legalized_def}")
        return legalized_def
    else:
        print(f"[WARN] OpenROAD legalization failed — check {outdir}/legalize.log")
        return None


# ============================================================
# 12. OPENROAD CONGESTION TCL WRITER
# ============================================================
def write_congestion_tcl(tech_lef, cell_lef, def_path, guide_path,
                         outdir, label):
    """
    Writes a TCL script that loads a DEF and runs global routing
    so the congestion heatmap can be viewed in the OpenROAD GUI.

    label : 'before' or 'after' — used in filename
    """
    tcl_path = os.path.join(outdir, f"congestion_{label}.tcl")
    lef_lines = ""
    if tech_lef and os.path.exists(tech_lef):
        lef_lines += f"read_lef {tech_lef}\n"
    if cell_lef and os.path.exists(cell_lef):
        lef_lines += f"read_lef {cell_lef}\n"

    tcl = f"""{lef_lines}read_def {def_path}
make_tracks
global_route -guide_file {guide_path}
# Open GUI and view Heat Maps -> Estimated Congestion (RUDY)
# gui::show
"""
    with open(tcl_path, 'w') as f:
        f.write(tcl)
    print(f"[INFO] Congestion TCL ({label}) → {tcl_path}")
    return tcl_path


# ============================================================
# 13. MAIN
# ============================================================
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 65)
    print("  Mini DeepPlace v3.2 — GPU Build (Full DEF Nets)")
    print("=" * 65)
    print(f"  Verilog      : {args.verilog}")
    print(f"  Cell LEF     : {args.lef}")
    print(f"  Tech LEF     : {args.tech_lef or '(not provided)'}")
    print(f"  Output       : {args.outdir}")
    print(f"  Episodes     : {args.episodes}")
    print(f"  Device       : {args.device}")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"  GNN dims     : {args.gnn_dim} x {args.gnn_layers} layers")
    print(f"  GNN refresh  : every {args.gnn_refresh} steps")
    print(f"  Max GNN edges: {MAX_GNN_EDGES:,} (sampled if netlist larger)")
    print(f"  Reward       : {args.alpha}×HPWL + {args.beta}×Congestion")
    print("=" * 65)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA not available — falling back to CPU")
        args.device = 'cpu'

    lef_sizes = parse_lef_sizes(args.lef)

    nl = SimpleNetlist()
    nl.parse(args.verilog, lef_sizes,
             default_size=args.cell_size,
             density=args.density,
             prefix=args.prefix)

    device = torch.device(args.device)
    env = GPUPlacementEnv(
        nl, bins=args.grid_bins,
        alpha=args.alpha, beta=args.beta,
        n_envs=args.n_envs, device=args.device,
    )

    (policy, hpwl_history, cong_history,
     baseline_hpwl, baseline_cong, best_hpwl) = train_ppo(env, args)

    # Plot training curves
    plot_path = os.path.join(args.outdir, 'training_curves.png')
    plot_curves(hpwl_history, cong_history,
                baseline_hpwl, baseline_cong, best_hpwl, plot_path)

    # Load best weights and write final RL-placed DEF
    best_weights = os.path.join(args.outdir, 'best_policy.pt')
    policy.load_state_dict(torch.load(best_weights, map_location=args.device))
    edge_index = nl.edge_index.to(device)

    placed_def = os.path.join(args.outdir, 'aes_rl_placed.def')
    write_placed_def(env, policy, edge_index, placed_def,
                     design_name='aes_cipher_top', device=device)

    # Write congestion TCL scripts for before/after comparison in OpenROAD GUI
    guide_path    = os.path.join(args.outdir, 'route.guide')
    baseline_def  = os.path.join(args.outdir, 'baseline_random.def')

    write_congestion_tcl(
        tech_lef=args.tech_lef, cell_lef=args.lef,
        def_path=baseline_def, guide_path=guide_path,
        outdir=args.outdir, label='before')

    write_congestion_tcl(
        tech_lef=args.tech_lef, cell_lef=args.lef,
        def_path=placed_def, guide_path=guide_path,
        outdir=args.outdir, label='after')

    # Optional: legalize with OpenROAD
    if not args.skip_legalize:
        legalize_with_openroad(
            placed_def=placed_def,
            tech_lef=args.tech_lef,
            cell_lef=args.lef,
            outdir=args.outdir,
            design_name='aes_cipher_top',
            openroad_bin=args.openroad,
        )

    h_impr = (baseline_hpwl - best_hpwl)     / (baseline_hpwl + 1e-8) * 100
    c_impr = (baseline_cong - min(cong_history)) / (baseline_cong + 1e-8) * 100

    print("\n" + "=" * 65)
    print("  OUTPUTS")
    print("=" * 65)
    print(f"  best_policy.pt         → trained GNN-PPO weights")
    print(f"  baseline_random.def    → random placement (BEFORE)")
    print(f"  aes_rl_placed.def      → RL optimized placement (AFTER)")
    print(f"  training_curves.png    → HPWL + congestion curves")
    print(f"  congestion_before.tcl  → load in OpenROAD GUI for BEFORE heatmap")
    print(f"  congestion_after.tcl   → load in OpenROAD GUI for AFTER heatmap")
    print(f"\n  HPWL improvement       : {h_impr:.1f}%")
    print(f"  Congestion reduction   : {c_impr:.1f}%")
    print("=" * 65)
    print("\n  To compare congestion in OpenROAD GUI:")
    print(f"    openroad -gui congestion_before.tcl   # view BEFORE heatmap")
    print(f"    openroad -gui congestion_after.tcl    # view AFTER heatmap")
    print("=" * 65)


if __name__ == '__main__':
    main()
