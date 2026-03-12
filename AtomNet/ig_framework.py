# ig_framework.py
import logging
import os
from typing import List, Optional, Literal
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from captum.attr import IntegratedGradients
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch_geometric.graphgym.config import cfg
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
import logging
# Avoid printing warning messages due to font issues of matplotlib
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def get_baseline(
    data: Data,
    mode: Literal["zeros", "batch_mean", "dataset_mean"] = "zeros",
    dataset_mean_vec: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Return a baseline of the same shape as data.x.
    - zeros: all zeros
    - batch_mean: The feature-wise mean vector of the current batch (broadcast to each node)
    - dataset_mean: Pass in the pre-calculated mean vector of the dataset (broadcast to each node)
    """
    if mode == "zeros":
        return torch.zeros_like(data.x)
    elif mode == "batch_mean":
        mean_vec = data.x.mean(dim=0, keepdim=True)  # [1, F]
        return mean_vec.expand_as(data.x).clone()
    elif mode == "dataset_mean":
        assert dataset_mean_vec is not None, "The dataset_mean pattern requires the provision of dataset_mean_vec"
        mean_vec = dataset_mean_vec.view(1, -1).to(data.x.device)  # [1, F]
        return mean_vec.expand_as(data.x).clone()
    else:
        raise ValueError(f"Unknown baseline mode: {mode}")


@torch.no_grad()
def estimate_dataset_feature_mean(loader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Estimate the feature-wise mean of all node features in the dataset (for dataset_mean baseline)
    """
    n = 0
    running_sum = None
    for data in loader:
        data = data.to(device)
        x = data.x
        if running_sum is None:
            running_sum = x.sum(dim=0)
        else:
            running_sum += x.sum(dim=0)
        n += x.size(0)
    return running_sum / max(n, 1)


# ============== The main process of IG: Batch → Node attribution → Graph aggregation → Data aggregation ==============
def ig_attribution_for_batch(
    model: nn.Module,
    data: Data,
    target: Optional[int] = 0,  # None
    baseline_mode: Literal["zeros", "batch_mean", "dataset_mean"] = "zeros",
    dataset_mean_vec: Optional[torch.Tensor] = None,
    ig_steps: int = 64,
) -> torch.Tensor:
    """
    Calculate the IG attribution (node level) for a batch and aggregate it to the graph level.
    return: graph_attr [num_graphs_in_batch, num_features]
    """
    model.eval()

    data.x = data.x.detach()
    data.x.requires_grad_(True)

    # Construct the forward function: only accept x, with the edge_indexbatch provided by the closure
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, data):
            super().__init__()
            self.model = model
            self.data = data  # Save the original Data (including edge_index, batch, etc.)

        def forward(self, x):
            # print("x:", x.shape, "batch:", self.data.batch.shape)
            # Make a copy of the data to avoid damaging the original batch
            data = self.data.clone()
            data.x = x
            pred, _ = model(data)  # [num_graphs, out_dim]

            return pred

    wrapped_model = ModelWrapper(model, data)

    ig = IntegratedGradients(wrapped_model)

    # baseline
    baseline = get_baseline(data, baseline_mode, dataset_mean_vec)

    # Calculate IG
    attributions = ig.attribute(
        inputs=data.x,  # IntegratedGradients requires that inputs must be tensors or tuples of tensors, and they should be capable of calculating gradients over the inputs. And data is not a tensor
        baselines=baseline,
        target=target,  # None
        n_steps=ig_steps,
        internal_batch_size=1,  # Forced gradual 'forward'
    )  # [num_nodes_in_batch, num_features]

    # Node -> Graph: Take the mean attribute for each node in the graph (summedian can also be used)
    graph_attr = scatter_mean(attributions, data.batch, dim=0)  # [num_graphs, num_features]
    return graph_attr


def compute_ig_importance_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    *,
    target: Optional[int] = 0,  # None
    baseline_mode: Literal["zeros", "batch_mean", "dataset_mean"] = "zeros",
    ig_steps: int = 64,
    use_abs: bool = True,
    feature_names: Optional[List[str]] = None,
    save_figure_dir: Optional[str] = None,
    save_csv_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Calculate the global importance of the "feature dimension" across the entire dataset.
    Return: feature_importance [num_features]
    - use_abs =True: Use |IG| re-aggregation, common practice, more stable
    - baseline_mode: zeros / batch_mean / dataset_mean
    - feature_names: If provided, the column names will be included when saving the CSV
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset_mean_vec = None
    if baseline_mode == "dataset_mean":
        dataset_mean_vec = estimate_dataset_feature_mean(loader, device)

    all_graph_attr = []
    for batch in tqdm(loader, ncols=100, desc="IG over dataset"):
        batch = batch.to(device)
        graph_attr = ig_attribution_for_batch(
            model=model,
            data=batch,
            target=target,  # None
            baseline_mode=baseline_mode,
            dataset_mean_vec=dataset_mean_vec,
            ig_steps=ig_steps,
        )  # [batch_size, num_features]

        if use_abs:
            graph_attr = graph_attr.abs()
        all_graph_attr.append(graph_attr.detach().cpu())

    all_graph_attr = torch.cat(all_graph_attr, dim=0)  # [num_graphs_total, num_features]
    feature_importance = all_graph_attr.mean(dim=0)    # [num_features]

    if save_csv_path is not None:
        os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(feature_importance.numel())]
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": feature_importance.numpy().tolist(),
        }).sort_values("importance", ascending=False)

        # Use the part before the last underline as the basis for grouping
        df['group'] = df['feature'].str.rsplit('_', n=1).str[0]
        # Sum by group
        tmp_df = df.groupby('group')['importance'].sum().reset_index()
        grouped_df = tmp_df.sort_values(by="importance", ascending=False)
        grouped_df = grouped_df.reset_index(drop=True)

        # print("----- IG feature importance ranking -----")
        # for idx, row in grouped_df.iterrows():
        #     print(f"#{idx + 1:<2d} {row['group']:<30s}: {row['importance']:.6f}")

        top_k = 20
        df_top = df.head(top_k)

        plt.figure(figsize=(12, 6))
        bars = plt.barh(
            df_top["feature"],
            df_top["importance"],
            color="#e69f00",
        )
        # plt.barh(df_top["feature"], df_top["importance"], color="skyblue")
        plt.gca().invert_yaxis()
        plt.xlabel("Feature Importance (mean |IG|)", fontsize=12)
        plt.ylabel("")
        # plt.title(f"ig_steps = {ig_steps}, Top {top_k} Features of {cfg.name}", fontsize=14, weight="bold")
        # Add a numerical label for each column (displayed on the right)
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.0005,
                     bar.get_y() + bar.get_height() / 2,
                     f"{width:.6f}",
                     va='center', ha='left', fontsize=10)

        plt.grid(axis='x', linestyle='--', alpha=0.7)
        # plt.grid(axis='y', linestyle='--', color='gray', alpha=0.6)

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        plt.savefig(os.path.join(save_figure_dir, f"top20_{ig_steps}_{cfg.name}.pdf"), format="pdf", bbox_inches="tight")
        # plt.savefig(f"./dataset/ig/img/top20_{ig_steps}_{cfg.name}.jpeg", dpi=1000, bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(12, 6))
        bars = plt.barh(
            grouped_df["group"],
            grouped_df["importance"],
            color="#e69f00",
        )
        plt.gca().invert_yaxis()
        plt.xlabel("Group Feature Importance (mean |IG|)", fontsize=12)
        plt.ylabel("")
        # plt.title(f"IG Feature Importance — {cfg.name} (Jarvis DFT-3D)", fontsize=16, weight="bold", pad=15)

        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.0005,
                     bar.get_y() + bar.get_height() / 2,
                     f"{width:.6f}",
                     va='center', ha='left', fontsize=10)

        plt.grid(axis='x', linestyle='--', alpha=0.7)
        # plt.grid(axis='y', linestyle='--', color='gray', alpha=0.6)

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        plt.savefig(os.path.join(save_figure_dir, f"IG_{ig_steps}_{cfg.name}.pdf"), format="pdf", bbox_inches="tight")
        # plt.savefig("./dataset/ig/img/IG_{}_{}.jpeg".format(ig_steps, cfg.name),
        #             dpi=1000, bbox_inches="tight")
        plt.show()

        df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        grouped_df.to_csv(save_csv_path.replace(".csv", "_grouped.csv"), index=False, encoding="utf-8-sig")
        print(f"[IG] Feature importance saved to: {save_csv_path}")

    return np.array(tmp_df["group"]), np.array(tmp_df["importance"])


def IG_metric(model, loader):
    ckpt_path = f'{cfg.run_dir}/ckpt/best.ckpt'
    assert os.path.exists(ckpt_path), "model parameter file does not exist. ❌"

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state"])

    atom_init = cfg.atom_init
    feature_file_path = f'./dataset/csv/{atom_init}.csv'
    assert os.path.exists(feature_file_path), f"{atom_init} file does not exist. ❌"

    save_dir = "./dataset/ig/img"
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(feature_file_path, index_col=0)
    feature_names = df.columns.tolist()
    assert len(feature_names) == 116, "The number of feature names does not match!"

    results = {}  # n_steps -> (group_names, group_scores, per_graph_group)
    n_steps_list = [20, 32, 64, 128]
    for n in n_steps_list:
        # Calculate the feature importance (IG) at the dataset level
        group_names, group_scores = compute_ig_importance_dataset(
            model=model,
            loader=loader,  # train_loader
            target=None,
            baseline_mode="zeros",  #  "zeros" / "batch_mean" / "dataset_mean"
            ig_steps=n,
            use_abs=True,
            feature_names=feature_names,
            save_figure_dir=save_dir,
            save_csv_path=f"./dataset/ig/ig_{n}_{cfg.name}.csv",
        )
        results[n] = (group_names, group_scores)

    def topk_overlap(a, b, k=5):
        # a,b are arrays of group scores (same order), returns Jaccard overlap for top-k sets
        topa = set(np.array(group_names)[np.argsort(-a)[:k]])
        topb = set(np.array(group_names)[np.argsort(-b)[:k]])
        inter = len(topa & topb)
        return inter / k, inter, topa, topb

    # build DataFrame summary
    summary_rows = []
    keys = list(results.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ni = keys[i]
            nj = keys[j]
            _, ai = results[ni]
            _, aj = results[nj]
            rho, p_rho = spearmanr(ai, aj)
            tau, p_tau = kendalltau(ai, aj)
            top8_overlap, inter8, _, _ = topk_overlap(ai, aj, k=8)
            # percent change in scores
            pct_change = np.abs(ai - aj) / (np.abs(ai) + 1e-12)
            mean_pct_change = pct_change.mean()
            summary_rows.append({
                "n1": ni, "n2": nj,
                "spearman": rho, "spearman_p": p_rho,
                "kendall": tau, "kendall_p": p_tau,
                "top8_overlap": top8_overlap,
                "mean_pct_change": mean_pct_change
            })

    summary_df = pd.DataFrame(summary_rows)
    print("\nPairwise comparison summary:")
    print(summary_df)

    plt.figure(figsize=(10, 6))

    colors = ['#9ecae1', '#fdae6b', '#a1d99b', '#fc9272']

    x = np.arange(len(group_names))
    width = 0.18
    for idx, (n, color) in enumerate(zip(keys, colors)):
        _, scores = results[n]
        # normalize to sum=1 for visual comparability if desired
        # scores_norm = scores / scores.sum()
        scores_norm = scores

        plt.bar(x + (idx - len(keys) / 2) * width, scores_norm, width=width, label=f"n={n}",
                color=color, alpha=0.9)

    plt.xticks(x, group_names, rotation=45, ha='right')
    plt.ylabel("Group importance (mean |IG|)")
    # plt.title("IG group importance for different n_steps")

    plt.grid(axis='y', linestyle='--', color='gray', alpha=0.7)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.legend()
    plt.tight_layout()

    plt.savefig(f"./dataset/ig/img/seed_{cfg.seed}_{cfg.figshare_target}_IG.pdf", format="pdf", bbox_inches="tight")
    # plt.savefig(f"./dataset/ig/img/seed_{cfg.seed}_{cfg.figshare_target}_IG.jpeg", dpi=1000, bbox_inches="tight")
    plt.show()

    for n in keys:
        _, scores = results[n]
        order = np.argsort(-scores)
        print(f"\nRanking for n_steps={n}:")
        for rank, idx in enumerate(order, 1):
            print(f"{rank:2d}. {group_names[idx]:30s} {scores[idx]:.6f}")
