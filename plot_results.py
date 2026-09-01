"""
plot_results.py
================
Sinh biểu đồ + bảng số liệu cho bản revise FedHST (PLOS ONE), đọc trực tiếp từ các
CSV do federated_experiments.py xuất ra. Chia làm 2 nhóm:

  A. MAIN BODY  : hội tụ (Accuracy/F1/Loss theo Round), bảng so sánh thuật toán
                  (mean ± std, có kiểm định ý nghĩa nếu có dữ liệu multi-seed),
                  bảng chi phí tính toán/truyền tải.
  B. ABLATION   : gamma, beta1/beta2, số lượng client, tỉ lệ tham gia, số local
                  epoch, mức non-IID (alpha).

Mọi hàm đều: (1) tự phát hiện có cột 'Seed' hay không để tính std đúng cách,
(2) in bảng ra console, (3) lưu PNG (300dpi) + CSV bảng số + đoạn LaTeX (booktabs)
để dán thẳng vào bản thảo, (4) không yêu cầu các cột không tồn tại (an toàn với
CSV cũ chưa có Round_Time_Sec/Seed/...).

Toàn bộ chữ trên hình (title, axis label, tick/giá trị trục, legend) được in
ĐẬM và tăng cỡ chữ (+1 mức so với mặc định matplotlib) để dễ đọc khi in vào
phụ lục (appendix) của bài báo — xem khối FONT SETTINGS bên dưới.

Cách dùng nhanh: xem khối `if __name__ == "__main__":` ở cuối file — chạy 1 lần
sẽ tự động sinh TOÀN BỘ các hình (3 dataset chính + 6 ablation + tổng hợp đa
dataset) từ các CSV có sẵn trong cùng thư mục.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white", palette="tab10")

# ==========================================
# --- FONT SETTINGS: chữ đậm + lớn hơn mặc định 1 cỡ ---
# (mặc định matplotlib: title~12, axis label~10, tick~10, legend~10)
# ==========================================
TITLE_FS = 16
LABEL_FS = 14
TICK_FS = 13
LEGEND_FS = 13

plt.rcParams.update({
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titlesize': TITLE_FS,
    'axes.labelsize': LABEL_FS,
    'xtick.labelsize': TICK_FS,
    'ytick.labelsize': TICK_FS,
    'legend.fontsize': LEGEND_FS,
    'legend.title_fontsize': LEGEND_FS,
    'font.size': TICK_FS,
})


def _style_axis(ax):
    """Áp dụng đậm + cỡ chữ lớn hơn cho tick labels của 1 axes (phòng khi rcParams
    không áp dụng đủ, ví dụ với set_xticklabels gọi thủ công sau đó)."""
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
        lbl.set_fontsize(TICK_FS)


def _bold_legend(legend):
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(LEGEND_FS)
    if legend.get_title() is not None:
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_fontsize(LEGEND_FS)


FIG_DIR = "figures"
TAB_DIR = "tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

STEADY_STATE_WINDOW = 15  # số round cuối dùng để tính steady-state (khớp bài gốc)


# ==========================================
# --- TIỆN ÍCH CHUNG ---
# ==========================================
def _steady_state_per_seed(df, metric, window=STEADY_STATE_WINDOW):
    """
    Trả về DataFrame [Algorithm, Seed, <metric>] = giá trị trung bình trên `window`
    round cuối, tính RIÊNG cho từng seed nếu có cột 'Seed', nếu không thì coi như
    1 seed ảo (Seed=0) để các hàm downstream dùng chung logic.

    An toàn với dữ liệu chạy eval_every>1 (một số round không được đánh giá, metric
    = None/NaN): các dòng đó bị loại trước khi lấy window, và nếu window quá hẹp so
    với eval_every (không đủ điểm đã đánh giá), tự nới rộng window để luôn có dữ liệu.
    """
    df = df.copy()
    if 'Seed' not in df.columns:
        df['Seed'] = 0
    df = df.dropna(subset=[metric])
    if len(df) == 0:
        raise ValueError(f"Không có round nào có giá trị hợp lệ cho '{metric}' — kiểm tra "
                          f"lại eval_every khi chạy run_federated (có thể window quá hẹp).")
    max_round = df['Round'].max()
    window_df = df[df['Round'] >= max_round - window]
    if window_df.empty:
        # window quá hẹp so với khoảng cách giữa các round có đánh giá (eval_every lớn)
        # -> lấy N round-đã-đánh-giá gần cuối nhất thay vì lọc theo khoảng cách Round
        eval_rounds_sorted = sorted(df['Round'].unique())
        n_points = min(len(eval_rounds_sorted), max(3, window // 5 or 3))
        keep_rounds = set(eval_rounds_sorted[-n_points:])
        window_df = df[df['Round'].isin(keep_rounds)]
    return window_df.groupby(['Algorithm', 'Seed'])[metric].mean().reset_index()


def _fmt_mean_std(mean, std, decimals=2):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _df_to_latex_booktabs(df, caption, label, float_cols=None):
    """Xuất bảng dạng LaTeX booktabs tối giản, tương thích style bài gốc (threeparttable)."""
    cols = df.columns.tolist()
    col_spec = "l" + "c" * (len(cols) - 1)
    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(cols) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        cells = [str(row[c]) for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ==========================================
# A1. MAIN BODY — HỘI TỤ (Accuracy / F1 / Loss theo Round)
# ==========================================
def plot_main_convergence(csv_path, algos=None, dataset_label="", out_name="main_convergence"):
    """
    3 panel: Accuracy, F1, Loss theo Round, mỗi đường 1 thuật toán.
    Nếu CSV có cột 'Seed' (multi-seed), vẽ đường trung bình + dải std giữa các seed
    (shaded band) — đúng chuẩn "mean ± std across independent runs" reviewer yêu cầu.
    Nếu không có 'Seed', vẽ đường đơn (giống hành vi bài gốc).
    """
    df = pd.read_csv(csv_path)
    if algos is not None:
        df = df[df['Algorithm'].isin(algos)]
    has_seed = 'Seed' in df.columns and df['Seed'].nunique() > 1

    algo_list = algos if algos is not None else sorted(df['Algorithm'].unique())
    palette = sns.color_palette("tab10", n_colors=len(algo_list))
    color_map = dict(zip(algo_list, palette))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [('Top1_Accuracy_Percent', 'Validation Accuracy (%)'),
               ('F1_Score', 'Macro F1-Score'),
               ('Global_Validation_Loss', 'Validation Loss')]

    handles, labels = [], []
    for ax, (metric, ylabel) in zip(axes, metrics):
        for algo in algo_list:
            sub_algo = df[df['Algorithm'] == algo]
            if len(sub_algo) == 0:
                continue
            if has_seed:
                agg = sub_algo.groupby('Round')[metric].agg(['mean', 'std']).reset_index()
                line, = ax.plot(agg['Round'], agg['mean'], label=algo, linewidth=2.5, color=color_map[algo])
                ax.fill_between(agg['Round'], agg['mean'] - agg['std'], agg['mean'] + agg['std'],
                                 alpha=0.15, color=color_map[algo])
            else:
                sub_sorted = sub_algo.sort_values('Round')
                line, = ax.plot(sub_sorted['Round'], sub_sorted[metric], label=algo, linewidth=2.5, color=color_map[algo])
            if ax is axes[0]:
                handles.append(line)
                labels.append(algo)
        ax.set_xlabel('Communication Round', fontsize=LABEL_FS, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=LABEL_FS, fontweight='bold')
        ax.set_title(ylabel.split(' (')[0] + f' over Rounds {dataset_label}', fontsize=TITLE_FS, fontweight='bold')
        _style_axis(ax)

    if handles:
        leg = fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 8),
                          bbox_to_anchor=(0.5, -0.08), prop={'weight': 'bold', 'size': LEGEND_FS})
        _bold_legend(leg)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"{out_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close(fig)
    return out_path


# ==========================================
# A2. MAIN BODY — BẢNG SO SÁNH (mean ± std qua seed, hoặc qua round cuối)
# ==========================================
def main_body_comparison_table(csv_path, order=None, out_name="main_comparison_table"):
    """
    Bảng chính giống Table 1/2/3 trong bài gốc nhưng CHUẨN HÓA lại: nếu có multi-seed,
    std tính giữa các lần chạy độc lập (đúng ý nghĩa thống kê hơn std giữa các round
    cuối của MỘT lần chạy). In ra console, lưu CSV + đoạn LaTeX.
    """
    df = pd.read_csv(csv_path)
    has_seed = 'Seed' in df.columns and df['Seed'].nunique() > 1

    rows = []
    for metric, key in [('Top1_Accuracy_Percent', 'Accuracy (%)'),
                         ('F1_Score', 'Macro F1-Score'),
                         ('Global_Validation_Loss', 'Validation Loss')]:
        per_seed = _steady_state_per_seed(df, metric)  # đã an toàn với None/NaN (eval_every>1)
        if has_seed:
            # std giữa các seed độc lập (đúng ý nghĩa thống kê hơn)
            agg = per_seed.groupby('Algorithm')[metric].agg(['mean', 'std']).reset_index()
        else:
            # fallback: std giữa các round trong cửa sổ steady-state (1 lần chạy)
            max_round = df['Round'].max()
            window_df = df.dropna(subset=[metric])
            window_df = window_df[window_df['Round'] >= max_round - STEADY_STATE_WINDOW]
            agg = window_df.groupby('Algorithm')[metric].agg(['mean', 'std']).reset_index()
        agg = agg.rename(columns={'mean': f'{key}_mean', 'std': f'{key}_std'})
        rows.append(agg.set_index('Algorithm'))

    table = pd.concat(rows, axis=1).reset_index()
    if order is not None:
        table['__order'] = table['Algorithm'].apply(lambda a: order.index(a) if a in order else 999)
        table = table.sort_values('__order').drop(columns='__order')
    else:
        table = table.sort_values('Accuracy (%)_mean', ascending=False)

    display_df = pd.DataFrame({'Algorithm': table['Algorithm']})
    display_df['Accuracy (%)'] = table.apply(lambda r: _fmt_mean_std(r['Accuracy (%)_mean'], r['Accuracy (%)_std']), axis=1)
    display_df['Macro F1-Score'] = table.apply(lambda r: _fmt_mean_std(r['Macro F1-Score_mean'], r['Macro F1-Score_std'], 4), axis=1)
    display_df['Validation Loss'] = table.apply(lambda r: _fmt_mean_std(r['Validation Loss_mean'], r['Validation Loss_std'], 4), axis=1)

    note = "std giữa các seed độc lập" if has_seed else "std giữa các round cuối (1 seed — khuyến nghị chạy multi-seed để có std đáng tin cậy hơn)"
    print(f"\n=== Bảng so sánh chính ({note}) ===")
    print(display_df.to_string(index=False))

    csv_out = os.path.join(TAB_DIR, f"{out_name}.csv")
    display_df.to_csv(csv_out, index=False, encoding="utf-8")
    latex_out = os.path.join(TAB_DIR, f"{out_name}.tex")
    with open(latex_out, "w", encoding="utf-8") as f:
        f.write(_df_to_latex_booktabs(
            display_df,
            caption=f"Performance comparison across federated optimization algorithms ({note}).",
            label="tab:main_comparison"
        ))
    print(f"[Saved] {csv_out}\n[Saved] {latex_out}")
    return display_df


# ==========================================
# A3. MAIN BODY — KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ
# ==========================================
def significance_table(csv_path, target_algo="FedHST", baseline_algos=None,
                        metric='Top1_Accuracy_Percent', out_name="significance_table"):
    from scipy import stats as scipy_stats
    df = pd.read_csv(csv_path)
    if 'Seed' not in df.columns:
        print("[Bỏ qua] CSV không có cột 'Seed' — cần chạy run_multi_seed() trước.")
        return None

    per_seed = _steady_state_per_seed(df, metric)
    if baseline_algos is None:
        baseline_algos = [a for a in per_seed['Algorithm'].unique() if a != target_algo]

    target_vals = per_seed[per_seed['Algorithm'] == target_algo].sort_values('Seed')[metric].values

    rows = []
    for base in baseline_algos:
        base_vals = per_seed[per_seed['Algorithm'] == base].sort_values('Seed')[metric].values
        if len(base_vals) != len(target_vals) or len(base_vals) < 2:
            continue
        t_stat, t_p = scipy_stats.ttest_rel(target_vals, base_vals)
        try:
            w_stat, w_p = scipy_stats.wilcoxon(target_vals, base_vals)
        except ValueError:
            w_stat, w_p = float('nan'), float('nan')
        rows.append({
            'Baseline': base,
            'N_seeds': len(base_vals),
            'Target_Algo': target_algo,
            'Target_mean': f"{target_vals.mean():.2f}",
            'Baseline_mean': f"{base_vals.mean():.2f}",
            'p (paired t-test)': f"{t_p:.4f}",
            'p (Wilcoxon)': f"{w_p:.4f}" if not np.isnan(w_p) else "n/a (n quá nhỏ)",
        })

    table = pd.DataFrame(rows)
    print(f"\n=== Kiểm định ý nghĩa thống kê: {target_algo} vs. baselines (n_seeds={per_seed['Seed'].nunique()}) ===")
    print(table.to_string(index=False))
    if per_seed['Seed'].nunique() < 5:
        print("[Lưu ý] n_seeds < 5 → power kiểm định rất thấp; nêu rõ giới hạn phần cứng "
              "trong Response to Reviewers / mục Limitations, không diễn giải p-value như bằng chứng mạnh.")

    csv_out = os.path.join(TAB_DIR, f"{out_name}.csv")
    table.to_csv(csv_out, index=False, encoding="utf-8")
    print(f"[Saved] {csv_out}")
    return table


# ==========================================
# A4. MAIN BODY — CHI PHÍ TÍNH TOÁN / TRUYỀN TẢI
# ==========================================
def overhead_table_and_plot(csv_path, payload_csv_path=None, out_name="overhead"):
    """
    Đọc Round_Time_Sec / Server_Time_Sec / Client_Time_Sec / Peak_Memory_MB từ CSV
    (yêu cầu bản federated_experiments.py đã cập nhật instrumentation).
    payload_csv_path: CSV xuất từ communication_payload_report() (tuỳ chọn).
    """
    df = pd.read_csv(csv_path)
    needed = {'Round_Time_Sec', 'Client_Time_Sec', 'Server_Time_Sec'}
    if not needed.issubset(df.columns):
        print(f"[Bỏ qua] CSV thiếu cột thời gian {needed - set(df.columns)} — cần chạy lại bằng "
              f"bản federated_experiments.py đã cập nhật instrumentation.")
        return None

    time_stats = df.groupby('Algorithm').agg(
        Round_Time_mean=('Round_Time_Sec', 'mean'),
        Client_Time_mean=('Client_Time_Sec', 'mean'),
        Server_Time_mean=('Server_Time_Sec', 'mean'),
    ).reset_index()

    if 'Peak_Memory_MB' in df.columns and df['Peak_Memory_MB'].notna().any():
        mem_stats = df.groupby('Algorithm')['Peak_Memory_MB'].mean().reset_index()
        time_stats = time_stats.merge(mem_stats, on='Algorithm', how='left')

    if payload_csv_path is not None and os.path.exists(payload_csv_path):
        payload = pd.read_csv(payload_csv_path)[['Algorithm', 'MB_Per_Client_Per_Round']]
        time_stats = time_stats.merge(payload, on='Algorithm', how='left')

    time_stats = time_stats.sort_values('Round_Time_mean')
    print("\n=== Chi phí tính toán / truyền tải trung bình mỗi round ===")
    print(time_stats.to_string(index=False))
    csv_out = os.path.join(TAB_DIR, f"{out_name}.csv")
    time_stats.to_csv(csv_out, index=False, encoding="utf-8")
    print(f"[Saved] {csv_out}")

    # Biểu đồ cột: thời gian client vs server (stacked)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(time_stats))
    ax.bar(x, time_stats['Client_Time_mean'], label='Client compute time', color='#4C72B0')
    ax.bar(x, time_stats['Server_Time_mean'], bottom=time_stats['Client_Time_mean'],
           label='Server aggregation time', color='#DD8452')
    ax.set_xticks(x)
    ax.set_xticklabels(time_stats['Algorithm'], rotation=30, ha='right', fontsize=TICK_FS, fontweight='bold')
    ax.set_ylabel('Thời gian trung bình / round (giây)', fontsize=LABEL_FS, fontweight='bold')
    ax.set_title('So sánh chi phí tính toán mỗi round', fontsize=TITLE_FS, fontweight='bold')
    _style_axis(ax)
    leg = ax.legend()
    _bold_legend(leg)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"{out_name}_time.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close(fig)

    return time_stats


# ==========================================
# B1. ABLATION — GAMMA (Spatial Damper Weight)
# ==========================================
def plot_gamma_ablation(csv_path, accuracy_threshold=80.0, out_name="ablation_gamma"):
    """
    3 panel: (1) gamma vs accuracy steady-state, (2) gamma vs tốc độ hội tụ (số round
    đầu tiên đạt ngưỡng accuracy_threshold), (3) gamma vs mức giảm phương sai
    (%) so với gamma=0 (tương đương FedAdam thuần).
    """
    df = pd.read_csv(csv_path)
    has_seed = 'Seed' in df.columns and df['Seed'].nunique() > 1

    gamma_vals = sorted(df['Gamma'].dropna().unique())

    rows = []
    for g in gamma_vals:
        sub = df[df['Gamma'] == g]
        per_seed = _steady_state_per_seed(sub, 'Top1_Accuracy_Percent')
        acc_mean = per_seed['Top1_Accuracy_Percent'].mean()
        acc_std = per_seed['Top1_Accuracy_Percent'].std() if has_seed else \
            sub[sub['Round'] >= sub['Round'].max() - STEADY_STATE_WINDOW]['Top1_Accuracy_Percent'].std()

        # Tốc độ hội tụ: round đầu tiên đạt ngưỡng (trung bình qua seed nếu có)
        conv_rounds = []
        for seed_id, seed_sub in sub.groupby(sub['Seed'] if 'Seed' in sub.columns else [0] * len(sub)):
            hit = seed_sub[seed_sub['Top1_Accuracy_Percent'] >= accuracy_threshold]
            conv_rounds.append(hit['Round'].min() if len(hit) > 0 else np.nan)
        conv_round_mean = np.nanmean(conv_rounds) if len(conv_rounds) and not all(np.isnan(conv_rounds)) else np.nan

        rows.append({'Gamma': g, 'Acc_mean': acc_mean, 'Acc_std': acc_std, 'Conv_Round': conv_round_mean})

    gamma_df = pd.DataFrame(rows).sort_values('Gamma')
    base_std = gamma_df.loc[gamma_df['Gamma'] == 0.0, 'Acc_std']
    base_std = base_std.values[0] if len(base_std) else np.nan
    gamma_df['Variance_Reduction_%'] = (1 - gamma_df['Acc_std'] / base_std) * 100 if base_std and base_std > 0 else np.nan

    print("\n=== Ablation Gamma (Spatial Damper Weight) ===")
    print(gamma_df.to_string(index=False))
    gamma_df.to_csv(os.path.join(TAB_DIR, f"{out_name}.csv"), index=False, encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].errorbar(gamma_df['Gamma'], gamma_df['Acc_mean'], yerr=gamma_df['Acc_std'],
                      marker='o', capsize=4, linewidth=2.5, color='#4C72B0')
    axes[0].set_xlabel(r'Spatial Damper Weight $\gamma$', fontsize=LABEL_FS, fontweight='bold')
    axes[0].set_ylabel('Steady-State Accuracy (%)', fontsize=LABEL_FS, fontweight='bold')
    axes[0].set_title(r'$\gamma$ vs. Accuracy', fontsize=TITLE_FS, fontweight='bold')

    axes[1].plot(gamma_df['Gamma'], gamma_df['Conv_Round'], marker='o', linewidth=2.5, color='#DD8452')
    axes[1].set_xlabel(r'Spatial Damper Weight $\gamma$', fontsize=LABEL_FS, fontweight='bold')
    axes[1].set_ylabel(f'Round đạt ngưỡng {accuracy_threshold}% Acc', fontsize=LABEL_FS, fontweight='bold')
    axes[1].set_title(r'$\gamma$ vs. Tốc độ hội tụ', fontsize=TITLE_FS, fontweight='bold')

    axes[2].plot(gamma_df['Gamma'], gamma_df['Variance_Reduction_%'], marker='o', linewidth=2.5, color='#55A868')
    axes[2].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[2].set_xlabel(r'Spatial Damper Weight $\gamma$', fontsize=LABEL_FS, fontweight='bold')
    axes[2].set_ylabel('Giảm phương sai so với γ=0 (%)', fontsize=LABEL_FS, fontweight='bold')
    axes[2].set_title(r'$\gamma$ vs. Variance Reduction', fontsize=TITLE_FS, fontweight='bold')

    for ax in axes:
        _style_axis(ax)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"{out_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close(fig)
    return gamma_df


# ==========================================
# B2. ABLATION — BETA1 / BETA2
# ==========================================
def plot_beta_ablation(csv_path, out_name="ablation_beta"):
    df = pd.read_csv(csv_path)
    if 'Sweep_Type' not in df.columns:
        raise ValueError("CSV cần có cột 'Sweep_Type' — hãy dùng dữ liệu từ run_beta_ablation().")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, sweep, xcol, xlabel in [(axes[0], 'beta1', 'Beta1', r'$\beta_1$'),
                                     (axes[1], 'beta2', 'Beta2', r'$\beta_2$')]:
        sub = df[df['Sweep_Type'] == sweep]
        per_val = _steady_state_per_seed(sub, 'Top1_Accuracy_Percent')
        # gắn lại giá trị beta tương ứng (per_seed mất cột Beta1/Beta2 sau groupby Algorithm)
        beta_map = sub.groupby('Algorithm')[xcol].first() if xcol in sub.columns else None
        stats = sub.groupby(xcol).apply(
            lambda g: pd.Series({'mean': g['Top1_Accuracy_Percent'].mean(), 'std': g['Top1_Accuracy_Percent'].std()})
        ).reset_index()
        ax.errorbar(stats[xcol], stats['mean'], yerr=stats['std'].fillna(0), marker='o', capsize=4, linewidth=2.5)
        ax.set_xlabel(xlabel, fontsize=LABEL_FS, fontweight='bold')
        ax.set_ylabel('Steady-State Accuracy (%)', fontsize=LABEL_FS, fontweight='bold')
        ax.set_title(f'Sensitivity theo {xlabel}', fontsize=TITLE_FS, fontweight='bold')
        _style_axis(ax)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"{out_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close(fig)


# ==========================================
# B3/B4/B5/B6. ABLATION TỔNG QUÁT: config_col (Num_Clients_Config,
# Participation_Frac, Local_Epochs_Config, Alpha_Config) x Algorithm
# ==========================================
def plot_generic_ablation(csv_path, config_col, xlabel, title, out_name):
    """Dùng chung cho: client count, participation ratio, local epochs, alpha level."""
    df = pd.read_csv(csv_path)
    if config_col not in df.columns:
        raise ValueError(f"CSV cần có cột '{config_col}'.")

    per_seed = _steady_state_per_seed(df, 'Top1_Accuracy_Percent')
    # map lại config_col vào per_seed (bị mất khi groupby theo Algorithm/Seed nếu config_col
    # không đồng nhất trong 1 (Algorithm,Seed) — nên group luôn theo config_col ở đây)
    df2 = df.copy()
    if 'Seed' not in df2.columns:
        df2['Seed'] = 0
    max_round = df2['Round'].max()
    window = df2[df2['Round'] >= max_round - STEADY_STATE_WINDOW]
    per_seed_cfg = window.groupby(['Algorithm', config_col, 'Seed'])['Top1_Accuracy_Percent'].mean().reset_index()
    stats = per_seed_cfg.groupby(['Algorithm', config_col])['Top1_Accuracy_Percent'].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, sub in stats.groupby('Algorithm'):
        sub = sub.sort_values(config_col)
        # config_col có thể là chuỗi (Participation_Frac) -> dùng vị trí thứ tự thay vì giá trị số
        x_vals = range(len(sub)) if sub[config_col].dtype == object else sub[config_col]
        ax.errorbar(x_vals, sub['mean'], yerr=sub['std'].fillna(0), marker='o', capsize=4,
                    linewidth=2.5, label=algo)
        if sub[config_col].dtype == object:
            ax.set_xticks(list(x_vals))
            ax.set_xticklabels(sub[config_col].tolist(), fontsize=TICK_FS, fontweight='bold')

    ax.set_xlabel(xlabel, fontsize=LABEL_FS, fontweight='bold')
    ax.set_ylabel('Steady-State Accuracy (%)', fontsize=LABEL_FS, fontweight='bold')
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold')
    _style_axis(ax)
    leg = ax.legend()
    _bold_legend(leg)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"{out_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close(fig)

    csv_out = os.path.join(TAB_DIR, f"{out_name}.csv")
    stats.to_csv(csv_out, index=False, encoding="utf-8")
    print(f"[Saved] {csv_out}")
    return stats


def plot_client_count_ablation(csv_path, out_name="ablation_client_count"):
    return plot_generic_ablation(csv_path, 'Num_Clients_Config', 'Số lượng client (tổng)',
                                  'Effect of Client Number', out_name)


def plot_participation_ablation(csv_path, out_name="ablation_participation"):
    return plot_generic_ablation(csv_path, 'Participation_Frac', 'Tỉ lệ tham gia mỗi round',
                                  'Effect of Participation Ratio', out_name)


def plot_local_epochs_ablation(csv_path, out_name="ablation_local_epochs"):
    return plot_generic_ablation(csv_path, 'Local_Epochs_Config', 'Số local epoch (E)',
                                  'Effect of Local Epochs', out_name)


def plot_alpha_level_ablation(csv_path, out_name="ablation_alpha_level"):
    return plot_generic_ablation(csv_path, 'Alpha_Config', r'Dirichlet $\alpha$ (nhỏ = non-IID mạnh hơn)',
                                  'Effect of Non-IID Severity (IID → Extreme)', out_name)


def plot_cross_dataset_summary(csv_paths_by_dataset, order=None, out_name="cross_dataset_summary"):
    """
    Gộp nhiều CSV (mỗi cái 1 dataset, cùng cấu trúc thuật toán) thành 1 bảng + 1 biểu đồ
    cột nhóm duy nhất — dùng cho bảng "generalization across datasets" trong bài báo.

    csv_paths_by_dataset : dict {tên_hiển_thị_dataset: đường_dẫn_csv}, ví dụ
        {"FashionMNIST": "fashionmnist_tuned_suite_epochs3.csv",
         "CIFAR-10": "cifar10_tuned_suite_epochs2.csv",
         "CIFAR-100": "cifar100_tuned_suite_epochs2.csv"}
    """
    rows = []
    for dataset_label, path in csv_paths_by_dataset.items():
        df = pd.read_csv(path)
        per_seed = _steady_state_per_seed(df, 'Top1_Accuracy_Percent')
        stats = per_seed.groupby('Algorithm')['Top1_Accuracy_Percent'].agg(['mean', 'std']).reset_index()
        for _, r in stats.iterrows():
            rows.append({'Dataset': dataset_label, 'Algorithm': r['Algorithm'],
                         'Accuracy_mean': r['mean'], 'Accuracy_std': r['std']})

    summary = pd.DataFrame(rows)
    pivot_display = summary.copy()
    pivot_display['Accuracy'] = pivot_display.apply(
        lambda r: _fmt_mean_std(r['Accuracy_mean'], r['Accuracy_std']), axis=1)
    table = pivot_display.pivot(index='Algorithm', columns='Dataset', values='Accuracy')
    if order is not None:
        table = table.reindex([a for a in order if a in table.index])
    dataset_order = list(csv_paths_by_dataset.keys())
    table = table[dataset_order]

    print("\n=== Bảng tổng hợp đa dataset (mean ± std giữa các seed) ===")
    print(table.to_string())
    csv_out = os.path.join(TAB_DIR, f"{out_name}.csv")
    table.to_csv(csv_out, encoding="utf-8")
    latex_out = os.path.join(TAB_DIR, f"{out_name}.tex")
    latex_df = table.reset_index()
    with open(latex_out, "w", encoding="utf-8") as f:
        f.write(_df_to_latex_booktabs(
            latex_df,
            caption="Generalization of FedHST across FashionMNIST, CIFAR-10, and CIFAR-100 "
                    "(mean $\\pm$ std over 3 independent seeds; tuned hyperparameters "
                    "$\\gamma{=}0.5, \\beta_1{=}0.7, \\beta_2{=}0.9999$).",
            label="tab:cross_dataset_summary"
        ))
    print(f"[Saved] {csv_out}\n[Saved] {latex_out}")

    # Biểu đồ cột nhóm: 1 nhóm/dataset, 1 cột/thuật toán
    algos = order if order is not None else sorted(summary['Algorithm'].unique())
    algos = [a for a in algos if a in summary['Algorithm'].unique()]
    x = np.arange(len(dataset_order))
    width = 0.8 / max(len(algos), 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = sns.color_palette("tab10", n_colors=len(algos))
    for i, algo in enumerate(algos):
        means, stds = [], []
        for ds in dataset_order:
            sub = summary[(summary['Dataset'] == ds) & (summary['Algorithm'] == algo)]
            means.append(sub['Accuracy_mean'].values[0] if len(sub) else np.nan)
            stds.append(sub['Accuracy_std'].values[0] if len(sub) else 0)
        ax.bar(x + i * width - 0.4 + width / 2, means, width, yerr=stds, capsize=3,
               label=algo, color=palette[i])
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_order, fontsize=TICK_FS, fontweight='bold')
    ax.set_ylabel('Steady-State Accuracy (%)', fontsize=LABEL_FS, fontweight='bold')
    ax.set_title('FedHST Generalization Across Datasets', fontsize=TITLE_FS, fontweight='bold')
    _style_axis(ax)
    leg = ax.legend()
    _bold_legend(leg)
    plt.tight_layout()
    fig_out = os.path.join(FIG_DIR, f"{out_name}.png")
    plt.savefig(fig_out, dpi=300, bbox_inches='tight')
    print(f"[Saved] {fig_out}")
    plt.close(fig)

    return table


# ==========================================
# CHẠY TOÀN BỘ TRONG 1 LẦN (dùng cho phụ lục bài báo)
# ==========================================
if __name__ == "__main__":
    ORDER = ["FedHST", "FedAdam", "FedDyn"]

    # ---- MAIN BODY: hội tụ + bảng so sánh + kiểm định + overhead, cho CẢ 3 dataset ----
    MAIN_CSVS = {
        "CIFAR-10": ("cifar10_tuned_suite_epochs2.csv", "cifar10"),
        "CIFAR-100": ("cifar100_tuned_suite_epochs2.csv", "cifar100"),
        "FashionMNIST": ("fashionmnist_tuned_suite_epochs3.csv", "fashionmnist"),
    }
    for dataset_label, (csv_name, slug) in MAIN_CSVS.items():
        if os.path.exists(csv_name):
            plot_main_convergence(csv_name, algos=ORDER, dataset_label=f"({dataset_label})",
                                   out_name=f"{slug}_main_convergence")
            main_body_comparison_table(csv_name, order=ORDER, out_name=f"{slug}_main_comparison_table")
            significance_table(csv_name, target_algo="FedHST",
                                baseline_algos=[a for a in ORDER if a != "FedHST"],
                                out_name=f"{slug}_significance_table")
            overhead_table_and_plot(csv_name, out_name=f"{slug}_overhead")
        else:
            print(f"[Bỏ qua] Không tìm thấy {csv_name}")

    # ---- Tổng hợp đa dataset (bar chart gộp) ----
    cross_paths = {ds: name for ds, (name, _) in MAIN_CSVS.items() if os.path.exists(name)}
    if len(cross_paths) >= 2:
        plot_cross_dataset_summary(cross_paths, order=ORDER)

    # ---- ABLATION (tất cả trên FashionMNIST, alpha=0.1 trừ khi có ghi chú khác) ----
    if os.path.exists("ablation_study/fashionmnist_gamma_ablation_clients100_rounds50_alpha0_1.csv"):
        plot_gamma_ablation("ablation_study/fashionmnist_gamma_ablation_clients100_rounds50_alpha0_1.csv")

    if os.path.exists("ablation_study/fashionmnist_beta_ablation_alpha0_1.csv"):
        plot_beta_ablation("ablation_study/fashionmnist_beta_ablation_alpha0_1.csv")

    if os.path.exists("ablation_study/fashionmnist_client_count_ablation_alpha0_1.csv"):
        plot_client_count_ablation("ablation_study/fashionmnist_client_count_ablation_alpha0_1.csv")

    if os.path.exists("ablation_study/fashionmnist_participation_ablation_alpha0_1.csv"):
        plot_participation_ablation("ablation_study/fashionmnist_participation_ablation_alpha0_1.csv")

    if os.path.exists("ablation_study/fashionmnist_local_epochs_ablation_alpha0_1.csv"):
        plot_local_epochs_ablation("ablation_study/fashionmnist_local_epochs_ablation_alpha0_1.csv")

    if os.path.exists("ablation_study/fashionmnist_alpha_level_ablation.csv"):
        plot_alpha_level_ablation("ablation_study/fashionmnist_alpha_level_ablation.csv")

    print("\n[DONE] Đã sinh xong toàn bộ biểu đồ/bảng có dữ liệu tương ứng "
          f"trong thư mục '{FIG_DIR}/' và '{TAB_DIR}/' (font đậm, cỡ chữ +1 mức).")
