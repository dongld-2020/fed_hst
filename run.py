from federated_experiments import run_tuned_suite, compute_significance

df_tuned = run_tuned_suite(
    dataset_name="CIFAR100",
    rounds=500,
    epochs=2,              # tăng từ 1 lên 3 theo yêu cầu
    alpha=0.5,
    seeds=[0, 1, 2],   # khớp 3 seed đã dùng ở lần chạy trước
    fedhst_gamma=0.5,
    fedhst_beta1=0.7,      # tốt nhất từ beta1-ablation (thay vì 0.9 mặc định)
    fedhst_beta2=0.9999,   # tốt nhất từ beta2-ablation (thay vì 0.999 mặc định)
    out_csv="cifar10_tuned_suite_epochs2_feddyn_seed12.csv",
)
compute_significance(
    df_tuned,
    target_algo="FedHST",
    baseline_algos=list(baseline for baseline in df_tuned['Algorithm'].unique() if baseline != "FedHST"),
    metric="Top1_Accuracy_Percent",
)
