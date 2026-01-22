import numpy as np
import matplotlib.pyplot as plt


def collect_edge_lengths(data_list, attr_name='cart_dist'):
    """
    从 data_list 中收集所有 edge length 值，返回 1D numpy array。
    支持 data.attr 为 torch.Tensor / numpy array / list。
    没有该属性会跳过该 data。
    """
    # 如果你的 data 是 torch.Tensor，请导入 torch
    try:
        import torch
        _HAS_TORCH = True
    except Exception:
        _HAS_TORCH = False

    arrs = []
    for i, data in enumerate(data_list):
        if not hasattr(data, attr_name):
            # 跳过没有该属性的 data
            continue
        vals = getattr(data, attr_name)
        # torch Tensor -> numpy
        if _HAS_TORCH and isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
        else:
            vals = np.asarray(vals)
        # 保证是一维
        vals = vals.ravel()
        if vals.size:
            arrs.append(vals)
    if not arrs:
        raise ValueError(f"在 data_list 中未找到任何 `{attr_name}` 或者所有都是空的。")
    return np.concatenate(arrs)


# 为了示范，如果没有数据，可以取消下面两行注释并创建示例数据：
# import numpy as np
# all_dists = np.concatenate([np.random.exponential(scale=1.2, size=1000), np.random.normal(loc=1.0, scale=0.1, size=200)])

def plot_hist_and_annotate(all_dists, n_bins=50, value_to_mark=1.0, tol=1e-2, bar_color='#c5e9f7',
                           title='Edge length histogram', savepath='./edgeHistogram.pdf', xlabel='Edge length', ylabel='Count'):
    """
    绘制纯柱状图（每个柱子黑色边框，柱身为浅蓝色），并标注：
      - 所在 bin 的总计数（bin_count）
      - 严格等于 value_to_mark（在 tol 容差内）的点数 (exact_count)
    参数：
      all_dists: 1D array-like of edge lengths
      n_bins: int 或 bins sequence
      value_to_mark: 要标注的长度值（例如 1.0）
      tol: 计算“严格等于”的容差（abs(x - value_to_mark) <= tol）
      bar_color: 柱子填充颜色（hex 或 matplotlib 可识别的颜色）
    """
    all_dists = np.asarray(all_dists).ravel()
    if all_dists.size == 0:
        raise ValueError("传入的 all_dists 为空。")

    counts, edges = np.histogram(all_dists, bins=n_bins)
    widths = edges[1:] - edges[:-1]
    centers = edges[:-1] + widths / 2.0

    fig, ax = plt.subplots(figsize=(9, 5))
    # 每根柱子黑色边框： edgecolor='k' linewidth=0.6；facecolor 用浅蓝色 bar_color
    bars = ax.bar(centers, counts, width=widths, align='center',
                  color=bar_color, edgecolor='k', linewidth=0.6, alpha=0.95, label='Histogram (counts)')

    # 画 x=value_to_mark 的竖线（视觉定位）
    ax.axvline(value_to_mark, color='k', linestyle='--', linewidth=1.0, label=f'x = {value_to_mark}')

    # 计算 value_to_mark 属于哪个 bin（digitize 返回 1..len(edges)-1）
    bin_idx = np.digitize([value_to_mark], edges)[0] - 1
    # 边界保护
    if bin_idx < 0 or bin_idx >= len(counts):
        bin_count = 0
    else:
        bin_count = int(counts[bin_idx])

    # 计算严格等于（或在 tol 内）的点数
    # 对浮点数，采用 abs diff <= tol 的近似判断；如果你希望“严格等于”取消 tol，请设 tol=0.0
    exact_count = int(np.sum(np.abs(all_dists - value_to_mark) <= tol))

    # 注释文本：同时显示两个计数
    note_text = f'bin count = {bin_count}\nexact count = {exact_count}'
    if bin_idx < 0:
        # value 在 left of histogram
        x_annot = edges[0] + 0.02 * (edges[-1] - edges[0])
        y_annot = counts.max() * 0.85 if counts.max() > 0 else 1.0
        ax.annotate(note_text, xy=(value_to_mark, y_annot),
                    xytext=(x_annot, y_annot),
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', alpha=0.25))
    elif bin_idx >= len(counts):
        # value 在 right of histogram
        x_annot = edges[-1] - 0.02 * (edges[-1] - edges[0])
        y_annot = counts.max() * 0.85 if counts.max() > 0 else 1.0
        ax.annotate(note_text, xy=(value_to_mark, y_annot),
                    xytext=(x_annot, y_annot),
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', alpha=0.25))
    else:
        # value 在某个 bin 内：把注释放在该柱顶部上方，并画箭头指向柱顶
        x_annot = centers[bin_idx]
        y_annot = counts[bin_idx]
        y_offset = max(counts) * 0.05 if counts.max() > 0 else 1.0
        ax.annotate(note_text, xy=(x_annot, y_annot),
                    xytext=(x_annot, y_annot + y_offset),
                    arrowprops=dict(arrowstyle='->', lw=1.0),
                    ha='center', va='bottom',
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.2', alpha=0.2))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, format="pdf", bbox_inches="tight")
        # plt.savefig(savepath, dpi=1200)
    plt.show()


# ----------------------------
# all_dists = collect_edge_lengths(data_list)   # <-- 把 data_list 换成你的变量名
# plot_hist_kde_and_annotate(all_dists, n_bins=50, value_to_mark=1.0, title='Edge lengths')

