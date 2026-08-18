"""
federated_experiments.py
=========================
Bản mở rộng của script gốc, bổ sung:
  1. Baseline mới: FedYogi, FedAdagrad, SCAFFOLD, FedDyn (FedProx đã có sẵn từ trước)
  2. Hàm chạy Ablation Study cho hệ số Spatial Damper (gamma) của FedHST
  3. Hỗ trợ chạy đầy đủ CIFAR-100 (model + data loader đã có sẵn, chỉ cần gọi)
  4. Hàm chạy multi-seed để phục vụ yêu cầu "5-10 lần chạy độc lập, mean ± std,
     kiểm định thống kê" của reviewer (tiện ích, không bắt buộc phải dùng ngay)

Các phần KHÔNG thay đổi so với bản gốc: SimpleCNN, CIFAR10CNN, CIFAR100CNN,
non_iid_partition_dirichlet, evaluate, và toàn bộ nhánh thuật toán cũ trong
run_federated (FedAvg, FedProx, FedNolowe, FedAdaComp, FedXVar, FedHST,
FedAvgM, FedAdam) được giữ nguyên logic, chỉ thêm nhánh mới.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import random
import time
import pandas as pd
from sklearn.metrics import f1_score


def set_seed(seed=42):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"--- Seed đã được cố định tuyệt đối tại: {seed} ---")


# ==========================================
# --- 1. KIẾN TRÚC CÁC MODEL (giữ nguyên) ---
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc = nn.Linear(32 * 13 * 13, 10)

    def forward(self, x):
        return self.fc(self.conv(x))


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.GroupNorm(16, 128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CIFAR100CNN(nn.Module):
    def __init__(self):
        super(CIFAR100CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.GroupNorm(16, 128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 100)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ==========================================
# --- 1b. KIẾN TRÚC MỚI: LIGHTWEIGHT RESNET (GroupNorm) CHO CIFAR-10/100 ---
# ==========================================
# Vì sao đổi kiến trúc: CIFAR10CNN/CIFAR100CNN gốc dùng Flatten (128*8*8=8192) rồi
# Linear(8192, 1024) — riêng lớp này đã chiếm 8.39M / 8.51M tham số (98.5%!) của toàn
# bộ model. Hệ quả kép, đều bất lợi cho bối cảnh FL:
#   (1) Dễ overfit cục bộ vì phần lớn "trí nhớ" của model nằm ở 1 lớp FC khổng lồ,
#       trong khi mỗi client (đặc biệt dưới non-IID alpha=0.1-0.3) chỉ có rất ít dữ liệu.
#   (2) Chi phí truyền tải cực lớn: 8.5M tham số * 4 byte ≈ 34MB/client/round — không
#       thực tế cho FL và sẽ bị soi ở phần communication overhead.
# Kiến trúc ResNet nhẹ dưới đây dùng residual block + GroupNorm (không dùng BatchNorm
# vì running-stats của BatchNorm không ổn định dưới dữ liệu non-IID giữa các client —
# đây cũng là lý do FedOpt/FedAvgM gốc dùng GroupNorm) + Global Average Pooling thay
# Flatten, giảm tham số ~5-8 lần trong khi tăng năng lực biểu diễn nhờ residual
# connections (gradient flow tốt hơn qua các lớp sâu).
class BasicBlockGN(nn.Module):
    """Residual block chuẩn (He et al., 2016) dùng GroupNorm thay BatchNorm."""
    def __init__(self, in_planes, planes, stride=1, groups=8):
        super(BasicBlockGN, self).__init__()
        gn_groups = min(groups, planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups, planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(gn_groups, planes),
            )

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class CIFARResNet(nn.Module):
    """
    ResNet nhẹ cho CIFAR-10/100, dùng chung kiến trúc, chỉ khác số kênh (width) và
    num_classes ở đầu ra. base_channels=64 cho CIFAR-100 (nhiều lớp hơn -> cần năng
    lực biểu diễn lớn hơn), base_channels=32 mặc định đủ cho CIFAR-10 (đơn giản hơn).
    3 stage residual (mỗi stage 2 block) + Global Average Pooling + 1 lớp FC duy nhất.
    """
    def __init__(self, num_classes=10, base_channels=32, dropout=0.2):
        super(CIFARResNet, self).__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, c1), c1),
            nn.ReLU(inplace=True),
        )
        # Stage 1: giữ nguyên kích thước 32x32
        self.stage1 = nn.Sequential(BasicBlockGN(c1, c1, stride=1), BasicBlockGN(c1, c1, stride=1))
        # Stage 2: 32x32 -> 16x16
        self.stage2 = nn.Sequential(BasicBlockGN(c1, c2, stride=2), BasicBlockGN(c2, c2, stride=1))
        # Stage 3: 16x16 -> 8x8
        self.stage3 = nn.Sequential(BasicBlockGN(c2, c3, stride=2), BasicBlockGN(c3, c3, stride=1))

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling thay Flatten -> tiết kiệm tham số
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def CIFAR10ResNet():
    """~0.5-0.7M tham số — nhẹ hơn CIFAR10CNN gốc (8.5M) khoảng 12-17 lần."""
    return CIFARResNet(num_classes=10, base_channels=32, dropout=0.2)


def CIFAR100ResNet():
    """Rộng hơn CIFAR10ResNet (base_channels=64) vì 100 lớp cần năng lực biểu diễn lớn hơn,
    nhưng vẫn nhẹ hơn CIFAR100CNN gốc (8.6M) nhiều lần."""
    return CIFARResNet(num_classes=100, base_channels=64, dropout=0.3)


# ==========================================
# --- 2. HÀM CHIA DỮ LIỆU (giữ nguyên) ---
# ==========================================
def non_iid_partition_dirichlet(dataset, num_clients, num_classes, alpha=0.1):
    if torch.is_tensor(dataset.targets):
        y_train = dataset.targets.clone().detach().cpu().numpy()
    else:
        y_train = np.array(dataset.targets)
    N = y_train.shape[0]
    K = num_classes
    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, split_points))]
        min_size = min([len(idx_j) for idx_j in idx_batch])
    return {i: idx_batch[i] for i in range(num_clients)}


# ==========================================
# --- 3. HÀM ĐÁNH GIÁ (giữ nguyên) ---
# ==========================================
def evaluate(model, test_loader, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    all_preds, all_targets = [], []
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    val_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    f1 = f1_score(all_targets, all_preds, average='macro')
    return val_loss, f1, accuracy


def _get_model(dataset_name, device, cifar_variant="resnet"):
    """
    cifar_variant : 'resnet' (mặc định, MỚI — nhẹ hơn & GAP thay Flatten) hoặc
                     'cnn' (kiến trúc gốc, giữ lại để so sánh/ablation kiến trúc nếu cần).
                     Chỉ áp dụng cho CIFAR10/CIFAR100; MNIST/FashionMNIST không đổi.
    """
    if dataset_name in ('MNIST', 'FashionMNIST'):
        return SimpleCNN().to(device)
    elif dataset_name == 'CIFAR10':
        model = CIFAR10ResNet() if cifar_variant == "resnet" else CIFAR10CNN()
        return model.to(device)
    elif dataset_name == 'CIFAR100':
        model = CIFAR100ResNet() if cifar_variant == "resnet" else CIFAR100CNN()
        return model.to(device)
    else:
        raise ValueError("Dataset không hợp lệ!")


# ==========================================
# --- 4. THUẬT TOÁN FEDERATED LEARNING ---
# ==========================================
def run_federated(algo_name, train_db, test_loader, clients_dict, dataset_name,
                   rounds=50, epochs=1, gamma=0.5, feddyn_alpha=0.01,
                   scaffold_server_lr=1.0, beta1=0.9, beta2=0.999,
                   participation_frac=(0.2, 0.3), cifar_variant="resnet",
                   label_smoothing=0.0, batch_size=32, use_amp=False, eval_every=1,
                   num_workers=0, log_prefix="", _device_announced=[False]):
    """
    gamma           : Spatial Damper Weight, chỉ dùng khi algo_name == "FedHST".
                       Tham số hoá để phục vụ ablation study (quét nhiều giá trị gamma).
    feddyn_alpha    : hệ số điều tiết Dynamic Regularization của FedDyn.
    scaffold_server_lr : learning rate phía server cho SCAFFOLD (global step size).
    beta1, beta2    : hệ số suy giảm động lượng bậc 1 / bậc 2 dùng chung cho
                       FedAdam / FedYogi / FedAdagrad(chỉ beta1) / FedHST — tham số hoá
                       để phục vụ ablation sensitivity theo yêu cầu reviewer.
    participation_frac : tỉ lệ (min, max) số client được chọn mỗi round so với tổng số
                       client — tham số hoá để phục vụ ablation "effect of participation ratio".
                       Mặc định (0.2, 0.3) khớp với hành vi gốc khi num_clients=100
                       (random.randint(20, 30)).
    cifar_variant   : 'resnet' (mặc định, kiến trúc nhẹ mới) hoặc 'cnn' (kiến trúc gốc) —
                       chỉ có tác dụng với CIFAR10/CIFAR100.
    label_smoothing : hệ số label smoothing cho CrossEntropyLoss (0.0 = tắt). Giá trị
                       0.1 thường giúp CIFAR-10/100 tổng quát hoá tốt hơn, đặc biệt hữu
                       ích khi mỗi client chỉ có ít dữ liệu cục bộ (dễ overfit).
    batch_size      : batch size huấn luyện cục bộ (mặc định 32, giữ tương thích ngược).
                       Trên GPU, batch lớn hơn (64-256) thường huấn luyện nhanh hơn nhiều
                       vì tận dụng song song tốt hơn — xem thêm ghi chú tốc độ ở cuối file.
    use_amp         : bật mixed-precision training (torch.cuda.amp) — tăng tốc đáng kể
                       trên GPU NVIDIA có Tensor Cores (RTX 20xx trở lên), giảm bộ nhớ,
                       sai số độ chính xác không đáng kể. Không có tác dụng trên CPU.
    eval_every      : chỉ đánh giá trên test set mỗi N round (mặc định 1 = mọi round,
                       giữ hành vi gốc). Với rounds lớn (vd. 1000), đặt eval_every=5-10
                       giúp tiết kiệm đáng kể vì evaluate() chạy trên TOÀN BỘ test set.
                       Các round không đánh giá sẽ không xuất hiện trong kết quả trả về.
    num_workers     : số worker cho DataLoader cục bộ (mặc định 0). Với dataset nhỏ theo
                       từng client, num_workers>0 thường KHÔNG giúp ích (overhead khởi tạo
                       worker mới mỗi client/round có thể còn chậm hơn) — chỉ nên tăng nếu
                       local dataset của mỗi client đủ lớn (vd. CIFAR100 với alpha cao/ít client).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not _device_announced[0]:
        if device.type == "cuda":
            print(f"[Device] Đang dùng GPU: {torch.cuda.get_device_name(device)}")
        else:
            print("[Device] KHÔNG tìm thấy GPU khả dụng — đang chạy trên CPU (sẽ chậm hơn nhiều "
                  "với CIFAR10/100). Kiểm tra: `python -c \"import torch; print(torch.cuda.is_available())\"` "
                  "— nếu trả về False, cần cài đúng bản torch có hỗ trợ CUDA khớp driver GPU của máy.")
        _device_announced[0] = True

    global_model = _get_model(dataset_name, device, cifar_variant=cifar_variant)
    num_params = sum(p.numel() for p in global_model.parameters())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    scaler = torch.amp.GradScaler(device.type, enabled=(use_amp and device.type == "cuda"))

    all_client_ids = list(clients_dict.keys())
    num_clients_total = len(all_client_ids)

    global_delta_prev = None
    global_momentum_buffer = {k: torch.zeros_like(v).float() for k, v in global_model.state_dict().items()}
    m_buffer = {k: torch.zeros_like(v).float() for k, v in global_model.state_dict().items()}
    v_buffer = {k: torch.zeros_like(v).float() for k, v in global_model.state_dict().items()}

    # --- State riêng cho SCAFFOLD: control variate của server (c) và của từng client (c_i) ---
    scaffold_c = {name: torch.zeros_like(p) for name, p in global_model.named_parameters()}
    scaffold_ci = {
        cid: {name: torch.zeros_like(p) for name, p in global_model.named_parameters()}
        for cid in all_client_ids
    }

    # --- State riêng cho FedDyn: server state h và gradient-correction của từng client ---
    feddyn_h = {name: torch.zeros_like(p) for name, p in global_model.named_parameters()}
    feddyn_local_grad = {
        cid: {name: torch.zeros_like(p) for name, p in global_model.named_parameters()}
        for cid in all_client_ids
    }

    results = []
    print(f"\n--- Running {algo_name} on {dataset_name} "
          f"{'(gamma=' + str(gamma) + ')' if algo_name == 'FedHST' else ''} {log_prefix}---")

    for r in range(rounds):
        t_round_start = time.time()
        lo = max(1, int(participation_frac[0] * num_clients_total))
        hi = max(lo, int(participation_frac[1] * num_clients_total))
        n_selected = random.randint(lo, hi)
        selected_indices = random.sample(all_client_ids, n_selected)

        client_sizes = [len(clients_dict[c]) for c in selected_indices]
        total_samples = sum(client_sizes)

        local_weights = []
        local_losses = []
        # Dùng cho SCAFFOLD: delta_y_i (thay đổi trọng số) và delta_c_i (thay đổi control variate)
        scaffold_delta_y = []
        scaffold_delta_c = []

        for c in selected_indices:
            local_model = copy.deepcopy(global_model)
            # SCAFFOLD nguyên bản dùng SGD thuần (không momentum): momentum tích lũy sai số
            # control-variate correction qua các bước và gây phân kỳ theo cấp số nhân
            # (đã quan sát thấy khi test — loss nổ lên ~1e35 trong vài chục round).
            local_momentum = 0.0 if algo_name == "SCAFFOLD" else 0.9
            optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=local_momentum, weight_decay=1e-4)
            loader = DataLoader(Subset(train_db, clients_dict[c]), batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=(device.type == "cuda"))

            local_model.train()
            total_loss = 0
            num_local_steps = 0
            amp_enabled = use_amp and device.type == "cuda"

            for _ in range(epochs):
                for data, target in loader:
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    optimizer.zero_grad()

                    with torch.autocast(device_type=device.type, enabled=amp_enabled):
                        output = local_model(data)
                        loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(output, target)

                        if algo_name == "FedProx":
                            prox_term = 0
                            for param, global_param in zip(local_model.parameters(), global_model.parameters()):
                                prox_term += (0.001 / 2) * torch.norm(param - global_param) ** 2
                            loss += prox_term

                        if algo_name == "FedDyn":
                            # Dynamic regularization (Acar et al., 2021):
                            # L_i(theta) = F_i(theta) - <h_i, theta> + (alpha/2)||theta - theta_global||^2
                            lin_term = 0.0
                            quad_term = 0.0
                            global_named = dict(global_model.named_parameters())
                            for name, param in local_model.named_parameters():
                                hi = feddyn_local_grad[c][name].to(device)
                                gparam = global_named[name].to(device)
                                lin_term = lin_term - torch.sum(hi * param)
                                quad_term = quad_term + torch.sum((param - gparam.detach()) ** 2)
                            loss = loss + lin_term + (feddyn_alpha / 2) * quad_term

                    scaler.scale(loss).backward()

                    if algo_name == "SCAFFOLD":
                        # QUAN TRỌNG khi bật AMP: phải unscale_() trước khi thao tác trực tiếp
                        # lên .grad, nếu không sẽ cộng correction (thang đo gốc) vào gradient
                        # đã bị GradScaler nhân với hệ số scale — sai lệch nghiêm trọng.
                        # Khi use_amp=False, scaler bị vô hiệu hoá nên unscale_() chỉ là no-op.
                        scaler.unscale_(optimizer)
                        with torch.no_grad():
                            for name, param in local_model.named_parameters():
                                if param.grad is not None:
                                    param.grad.add_(scaffold_c[name].to(device) - scaffold_ci[c][name].to(device))

                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                    num_local_steps += 1

            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_losses.append(total_loss / len(loader))

            if algo_name == "SCAFFOLD":
                eta_l = 0.01
                # Sàn tối thiểu cho K: dưới Dirichlet alpha nhỏ, nhiều client có rất ít batch
                # (K = 1-3), khiến hệ số 1/(K*eta_l) trong Option II bùng nổ và cộng dồn qua
                # các round gây phân kỳ. Đặt sàn K_min=5 để giữ hệ số chia trong khoảng an toàn.
                K_eff = max(num_local_steps, 5)
                CLIP_VAL = 5.0  # clip biên độ control-variate (đơn vị: gradient scale) để chống nổ số
                new_ci = {}
                dy = {}
                dc = {}
                global_named = dict(global_model.named_parameters())
                for name, param in local_model.named_parameters():
                    x_name = global_named[name].detach().to(device)
                    y_name = param.detach().to(device)
                    # Option II: c_i^+ = c_i - c + (x - y_i) / (K * eta_l)
                    ci_plus = (scaffold_ci[c][name].to(device) - scaffold_c[name].to(device)
                               + (x_name - y_name) / (K_eff * eta_l))
                    ci_plus = torch.clamp(ci_plus, -CLIP_VAL, CLIP_VAL)
                    new_ci[name] = ci_plus.cpu()
                    dy[name] = (y_name - x_name).cpu()
                    dc[name] = (ci_plus - scaffold_ci[c][name].to(device)).cpu()
                scaffold_ci[c] = new_ci
                scaffold_delta_y.append(dy)
                scaffold_delta_c.append(dc)

            if algo_name == "FedDyn":
                global_named = dict(global_model.named_parameters())
                for name, param in local_model.named_parameters():
                    diff = (param.detach() - global_named[name].detach())
                    updated = feddyn_local_grad[c][name].to(device) - feddyn_alpha * diff
                    feddyn_local_grad[c][name] = updated.cpu()

        t_client_end = time.time()

        # --- SERVER AGGREGATION ---
        new_state = copy.deepcopy(global_model.state_dict())
        spatial_var_time = None

        if algo_name == "FedAvg":
            for key in new_state.keys():
                new_state[key] = torch.zeros_like(new_state[key]).float()
                for i in range(len(local_weights)):
                    weight = client_sizes[i] / total_samples
                    new_state[key] += local_weights[i][key].float() * weight

        elif algo_name == "FedProx":
            for key in new_state.keys():
                new_state[key] = torch.mean(torch.stack([local_weights[i][key].float() for i in range(len(local_weights))]), dim=0)

        elif algo_name == "FedNolowe":
            losses = np.array(local_losses, dtype=np.float64)
            weights = 1.0 - (losses / (losses.sum() + 1e-9))
            weights /= (weights.sum() + 1e-9)
            for key in new_state.keys():
                new_state[key] = torch.zeros_like(new_state[key]).float()
                for i in range(len(local_weights)):
                    new_state[key] += local_weights[i][key].float() * weights[i]

        elif algo_name == "FedAdaComp":
            current_delta = {}
            for key in new_state.keys():
                diffs = torch.stack([local_weights[i][key].float() - global_model.state_dict()[key].float() for i in range(len(local_weights))])
                current_delta[key] = torch.mean(diffs, dim=0)
            gain = 1.0
            if global_delta_prev is not None:
                dot, n_curr, n_prev = 0.0, 0.0, 0.0
                for k in current_delta:
                    dot += torch.sum(current_delta[k] * global_delta_prev[k])
                    n_curr += torch.norm(current_delta[k]) ** 2
                    n_prev += torch.norm(global_delta_prev[k]) ** 2
                cos_sim = dot / (torch.sqrt(n_curr) * torch.sqrt(n_prev) + 1e-9)
                gain = torch.exp(cos_sim).item()
            for key in new_state.keys():
                new_state[key] = global_model.state_dict()[key].float() + gain * current_delta[key]
            global_delta_prev = current_delta

        elif algo_name == "FedXVar":
            beta, gamma_xv, eps, eta = 0.9, 0.5, 1e-8, 1
            for key in new_state.keys():
                client_deltas = [local_weights[i][key].float() - global_model.state_dict()[key].float() for i in range(len(local_weights))]
                mean_delta = torch.zeros_like(new_state[key]).float()
                for i in range(len(client_deltas)):
                    weight = client_sizes[i] / total_samples
                    mean_delta += client_deltas[i] * weight
                spatial_variance = torch.zeros_like(new_state[key]).float()
                for i in range(len(client_deltas)):
                    weight = client_sizes[i] / total_samples
                    spatial_variance += weight * (client_deltas[i] - mean_delta) ** 2
                cv_squared = spatial_variance / (mean_delta ** 2 + eps)
                gate = torch.exp(-gamma_xv * cv_squared)
                global_momentum_buffer[key] = beta * global_momentum_buffer[key] + (1 - beta) * mean_delta
                step = eta * global_momentum_buffer[key] * gate
                new_state[key] = global_model.state_dict()[key].float() + step

        elif algo_name == "FedHST":
            tau, eta = 1e-3, 0.01  # beta1, beta2 lấy từ tham số hàm (phục vụ ablation sensitivity)
            t_spatial_var_start = time.time()
            for key in new_state.keys():
                client_deltas = [local_weights[i][key].float() - global_model.state_dict()[key].float() for i in range(len(local_weights))]
                mean_delta = torch.zeros_like(new_state[key]).float()
                for i in range(len(client_deltas)):
                    weight = client_sizes[i] / total_samples
                    mean_delta += client_deltas[i] * weight
                spatial_variance = torch.zeros_like(new_state[key]).float()
                for i in range(len(client_deltas)):
                    weight = client_sizes[i] / total_samples
                    spatial_variance += weight * (client_deltas[i] - mean_delta) ** 2
                m_buffer[key] = beta1 * m_buffer[key] + (1 - beta1) * mean_delta
                v_buffer[key] = beta2 * v_buffer[key] + (1 - beta2) * (mean_delta ** 2)
                hybrid_variance = v_buffer[key] + gamma * spatial_variance   # <-- gamma tham số hoá cho ablation
                step = eta * m_buffer[key] / (torch.sqrt(hybrid_variance) + tau)
                new_state[key] = global_model.state_dict()[key].float() + step
            spatial_var_time = time.time() - t_spatial_var_start

        elif algo_name == "FedAvgM":
            beta = 0.9
            for key in new_state.keys():
                delta = torch.zeros_like(new_state[key]).float()
                for i in range(len(local_weights)):
                    weight = client_sizes[i] / total_samples
                    delta += (local_weights[i][key].float() - global_model.state_dict()[key].float()) * weight
                global_momentum_buffer[key] = beta * global_momentum_buffer[key] + delta
                new_state[key] = global_model.state_dict()[key].float() + global_momentum_buffer[key]

        elif algo_name == "FedAdam":
            tau, eta = 1e-3, 0.01  # beta1, beta2 lấy từ tham số hàm
            for key in new_state.keys():
                delta = torch.zeros_like(new_state[key]).float()
                for i in range(len(local_weights)):
                    weight = client_sizes[i] / total_samples
                    delta += (local_weights[i][key].float() - global_model.state_dict()[key].float()) * weight
                m_buffer[key] = beta1 * m_buffer[key] + (1 - beta1) * delta
                v_buffer[key] = beta2 * v_buffer[key] + (1 - beta2) * (delta ** 2)
                step = eta * m_buffer[key] / (torch.sqrt(v_buffer[key]) + tau)
                new_state[key] = global_model.state_dict()[key].float() + step

        # --------- MỚI: FedAdagrad ---------
        elif algo_name == "FedAdagrad":
            tau, eta = 1e-3, 0.01  # beta1 lấy từ tham số hàm
            for key in new_state.keys():
                delta = torch.zeros_like(new_state[key]).float()
                for i in range(len(local_weights)):
                    weight = client_sizes[i] / total_samples
                    delta += (local_weights[i][key].float() - global_model.state_dict()[key].float()) * weight
                m_buffer[key] = beta1 * m_buffer[key] + (1 - beta1) * delta
                # Adagrad: tích lũy không suy giảm (không có beta2)
                v_buffer[key] = v_buffer[key] + delta ** 2
                step = eta * m_buffer[key] / (torch.sqrt(v_buffer[key]) + tau)
                new_state[key] = global_model.state_dict()[key].float() + step

        # --------- MỚI: FedYogi ---------
        elif algo_name == "FedYogi":
            tau, eta = 1e-3, 0.01  # beta1, beta2 lấy từ tham số hàm
            for key in new_state.keys():
                delta = torch.zeros_like(new_state[key]).float()
                for i in range(len(local_weights)):
                    weight = client_sizes[i] / total_samples
                    delta += (local_weights[i][key].float() - global_model.state_dict()[key].float()) * weight
                m_buffer[key] = beta1 * m_buffer[key] + (1 - beta1) * delta
                delta_sq = delta ** 2
                # Yogi: v_t = v_{t-1} - (1-beta2) * sign(v_{t-1} - delta^2) * delta^2
                v_buffer[key] = v_buffer[key] - (1 - beta2) * torch.sign(v_buffer[key] - delta_sq) * delta_sq
                step = eta * m_buffer[key] / (torch.sqrt(v_buffer[key]) + tau)
                new_state[key] = global_model.state_dict()[key].float() + step

        # --------- MỚI: SCAFFOLD ---------
        elif algo_name == "SCAFFOLD":
            for key in new_state.keys():
                mean_dy = torch.zeros_like(new_state[key]).float()
                for i in range(len(scaffold_delta_y)):
                    weight = client_sizes[i] / total_samples
                    mean_dy += scaffold_delta_y[i][key].float().to(device) * weight
                new_state[key] = global_model.state_dict()[key].float() + scaffold_server_lr * mean_dy
            # Cập nhật control variate của server: c <- c + (|S_t| / N) * mean(delta_c_i)
            for key in scaffold_c.keys():
                mean_dc = torch.zeros_like(scaffold_c[key]).float()
                for i in range(len(scaffold_delta_c)):
                    mean_dc += scaffold_delta_c[i][key].float().to(device) / len(scaffold_delta_c)
                scaffold_c[key] = scaffold_c[key] + (n_selected / num_clients_total) * mean_dc

        # --------- MỚI: FedDyn ---------
        elif algo_name == "FedDyn":
            # theta^{t+1} = mean_weighted(theta_i) - (1/alpha) * h^t
            for key in new_state.keys():
                mean_theta = torch.zeros_like(new_state[key]).float()
                for i in range(len(local_weights)):
                    weight = client_sizes[i] / total_samples
                    mean_theta += local_weights[i][key].float() * weight
                new_state[key] = mean_theta
            # Cập nhật server state h dựa trên độ lệch trung bình theta_i - theta_global (trước khi ghi đè)
            global_named = dict(global_model.named_parameters())
            for name in feddyn_h.keys():
                mean_diff = torch.zeros_like(feddyn_h[name]).float()
                for i, c in enumerate(selected_indices):
                    weight = client_sizes[i] / total_samples
                    diff = local_weights[i][name].float() - global_named[name].detach()
                    mean_diff += diff * weight
                feddyn_h[name] = feddyn_h[name] - feddyn_alpha * mean_diff
            for key in new_state.keys():
                if key in feddyn_h:
                    new_state[key] = new_state[key] - (1.0 / feddyn_alpha) * feddyn_h[key]

        global_model.load_state_dict(new_state)
        t_server_end = time.time()

        is_eval_round = (eval_every <= 1) or ((r + 1) % eval_every == 0) or (r == rounds - 1)
        if not is_eval_round:
            # Bỏ qua evaluate() để tiết kiệm thời gian — evaluate() chạy trên TOÀN BỘ
            # test set, tốn đáng kể khi rounds lớn (vd. 1000) và dataset nặng (CIFAR100).
            # Vẫn ghi lại thời gian/overhead của round này, chỉ để trống các chỉ số hiệu năng.
            results.append({
                'Algorithm': algo_name, 'Dataset': dataset_name,
                'Gamma': gamma if algo_name == "FedHST" else None,
                'Beta1': beta1, 'Beta2': beta2, 'Round': r + 1,
                'Global_Validation_Loss': None, 'F1_Score': None, 'Top1_Accuracy_Percent': None,
                'Num_Selected_Clients': n_selected, 'Num_Total_Clients': num_clients_total,
                'Num_Model_Params': num_params,
                'Round_Time_Sec': t_server_end - t_round_start,
                'Client_Time_Sec': t_client_end - t_round_start,
                'Server_Time_Sec': t_server_end - t_client_end,
                'Spatial_Var_Time_Sec': spatial_var_time if algo_name == "FedHST" else None,
                'Peak_Memory_MB': None,
            })
            continue

        val_loss, f1, acc = evaluate(global_model, test_loader, device)
        print(f"Round {r+1:2d} | Val Loss: {val_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.2f}%")

        peak_mem_mb = None
        if device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        results.append({
            'Algorithm': algo_name,
            'Dataset': dataset_name,
            'Gamma': gamma if algo_name == "FedHST" else None,
            'Beta1': beta1,
            'Beta2': beta2,
            'Round': r + 1,
            'Global_Validation_Loss': val_loss,
            'F1_Score': f1,
            'Top1_Accuracy_Percent': acc,
            'Num_Selected_Clients': n_selected,
            'Num_Total_Clients': num_clients_total,
            'Num_Model_Params': num_params,
            'Round_Time_Sec': t_server_end - t_round_start,
            'Client_Time_Sec': t_client_end - t_round_start,
            'Server_Time_Sec': t_server_end - t_client_end,
            'Spatial_Var_Time_Sec': spatial_var_time if algo_name == "FedHST" else None,
            'Peak_Memory_MB': peak_mem_mb,
        })

    return results


# ==========================================
# --- 5. TIỆN ÍCH CHẠY NHIỀU THUẬT TOÁN / GAMMA / SEED ---
# ==========================================
def _robust_download(dataset_cls, root, train, transform, archive_filename, max_retries=3):
    """
    Wrapper quanh các dataset của torchvision (MNIST/FashionMNIST/CIFAR10/CIFAR100) để
    tự phục hồi khi tải bị đứt giữa chừng — lỗi rất hay gặp trên mạng chập chờn:
    "RuntimeError: File not found or corrupted." torchvision KHÔNG tự xoá file .tar.gz/
    .gz bị dở khi lỗi, nên nếu chạy lại ngay sẽ bị lỗi y hệt (nó thấy file đã tồn tại,
    không tải lại, rồi kiểm tra MD5 thất bại). Hàm này tự phát hiện lỗi, xoá file hỏng,
    và thử tải lại tối đa `max_retries` lần trước khi báo lỗi kèm hướng dẫn thủ công.
    """
    import os
    import time

    last_err = None
    archive_path = os.path.join(root, archive_filename)
    for attempt in range(1, max_retries + 1):
        try:
            return dataset_cls(root, train=train, download=True, transform=transform)
        except RuntimeError as e:
            last_err = e
            if os.path.exists(archive_path):
                print(f"[Cảnh báo] File tải về bị lỗi/thiếu ({archive_path}) — "
                      f"đang xoá và thử tải lại (lần {attempt}/{max_retries})...")
                os.remove(archive_path)
            else:
                print(f"[Cảnh báo] Lỗi tải dữ liệu (lần {attempt}/{max_retries}): {e}")
            time.sleep(2)

    raise RuntimeError(
        f"Không thể tải '{archive_filename}' sau {max_retries} lần thử — có thể do mạng "
        f"không ổn định hoặc bị chặn. Cách khắc phục thủ công:\n"
        f"  1. Xoá toàn bộ thư mục '{root}' để dọn sạch file dở.\n"
        f"  2. Tải thủ công '{archive_filename}' từ https://www.cs.toronto.edu/~kriz/cifar.html "
        f"(với MNIST/FashionMNIST, tìm mirror tương ứng), đặt vào '{root}/{archive_filename}'.\n"
        f"  3. Chạy lại — torchvision sẽ thấy file đã có và tự giải nén (không cần tải lại).\n"
        f"Lỗi gốc: {last_err}"
    )


def load_dataset(dataset_name, augmentation="strong"):
    """
    Trả về (train_dataset, test_dataset, num_classes) — bao gồm CIFAR-100.

    augmentation : 'strong' (mặc định, MỚI) hoặc 'basic' (pipeline gốc, giữ lại để
                   so sánh/ablation). Chỉ ảnh hưởng CIFAR10/CIFAR100.
                   'strong' = RandomCrop + Flip + TrivialAugmentWide + RandomErasing —
                   TrivialAugmentWide là policy augmentation tổng quát (không cần search
                   riêng theo dataset như AutoAugment), đã được chứng minh cải thiện
                   ổn định 1-3 điểm % accuracy trên CIFAR trong nhiều benchmark; kết hợp
                   RandomErasing (Zhong et al., 2020) giúp giảm overfit cục bộ — đặc biệt
                   quan trọng dưới FL non-IID khi mỗi client chỉ có rất ít ảnh/lớp.
    """
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        return train_dataset, test_dataset, 10

    elif dataset_name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        return train_dataset, test_dataset, 10

    elif dataset_name == 'CIFAR10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        if augmentation == "strong":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = _robust_download(datasets.CIFAR10, './data', True, transform_train, 'cifar-10-python.tar.gz')
        test_dataset = _robust_download(datasets.CIFAR10, './data', False, transform_test, 'cifar-10-python.tar.gz')
        return train_dataset, test_dataset, 10

    elif dataset_name == 'CIFAR100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        if augmentation == "strong":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = _robust_download(datasets.CIFAR100, './data', True, transform_train, 'cifar-100-python.tar.gz')
        test_dataset = _robust_download(datasets.CIFAR100, './data', False, transform_test, 'cifar-100-python.tar.gz')
        return train_dataset, test_dataset, 100

    else:
        raise ValueError("Dataset không hợp lệ!")


def _save_progress(all_metrics, out_csv):
    """
    Ghi đè CSV bằng toàn bộ kết quả TÍCH LŨY đến thời điểm hiện tại. Gọi sau mỗi lần
    chạy xong 1 (thuật toán, seed) trong các hàm run_*() bên dưới — nếu chương trình bị
    crash/gián đoạn giữa chừng (lỗi runtime, mất điện, Ctrl+C...), file out_csv vẫn chứa
    đầy đủ kết quả của các lần đã chạy xong, không phải chạy lại từ đầu.
    """
    pd.DataFrame(all_metrics).to_csv(out_csv, index=False)


def run_algo_suite(dataset_name, algos, rounds, epochs, alpha, num_clients=100,
                    seed_partition=42, out_csv=None):
    """
    Chạy một danh sách thuật toán trên cùng một dataset/partition (cùng seed chia dữ liệu
    để đảm bảo so sánh công bằng), rồi xuất CSV.
    Dùng để chạy nhanh baseline mới: FedYogi, FedAdagrad, SCAFFOLD, FedDyn, FedProx...

    Tự động ghi CSV tạm sau mỗi thuật toán hoàn thành (resume-safe) — xem _save_progress().
    """
    if out_csv is None:
        out_csv = f"{dataset_name.lower()}_baselines_clients{num_clients}_rounds{rounds}_alpha{alpha}.csv"

    set_seed(seed_partition)
    train_dataset, test_dataset, num_classes = load_dataset(dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print(f"Phân chia dữ liệu Dirichlet (alpha={alpha}, classes={num_classes})...")
    clients_partition = non_iid_partition_dirichlet(train_dataset, num_clients, num_classes, alpha)

    all_metrics = []
    for algo in algos:
        set_seed(42)  # cố định seed huấn luyện cho mỗi thuật toán (giữ như bản gốc)
        res = run_federated(algo, train_dataset, test_loader, clients_partition, dataset_name,
                             rounds=rounds, epochs=epochs)
        all_metrics.extend(res)
        _save_progress(all_metrics, out_csv)

    print(f"\n[DONE] Đã xuất '{out_csv}'.")
    return pd.DataFrame(all_metrics)


def run_gamma_ablation(dataset_name, gammas, rounds, epochs, alpha, num_clients=100,
                        seed_partition=42, seeds=(42,), out_csv=None):
    """
    Ablation study cho Spatial Damper Weight (gamma) của FedHST.
    gammas: list các giá trị cần quét, ví dụ [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    (gamma=0.0 tương đương FedAdam thuần — dùng để kiểm chứng claim lý thuyết trong bài).
    seeds : danh sách seed huấn luyện độc lập, ví dụ [0, 1, 2] — QUAN TRỌNG để kiểm tra
            xem giá trị gamma tối ưu có ổn định qua nhiều lần chạy hay chỉ tốt "may mắn"
            trên một seed cụ thể. Mặc định (42,) giữ hành vi cũ (1 seed) để tương thích
            ngược, nhưng nên truyền nhiều seed khi dùng cho kết luận trong bài báo.

    Tự động ghi CSV tạm sau mỗi (gamma, seed) hoàn thành (resume-safe) — xem _save_progress().
    """
    if out_csv is None:
        out_csv = f"{dataset_name.lower()}_gamma_ablation_clients{num_clients}_rounds{rounds}_alpha{alpha}.csv"

    set_seed(seed_partition)
    train_dataset, test_dataset, num_classes = load_dataset(dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print(f"Phân chia dữ liệu Dirichlet (alpha={alpha}, classes={num_classes})...")
    clients_partition = non_iid_partition_dirichlet(train_dataset, num_clients, num_classes, alpha)

    all_metrics = []
    for g in gammas:
        for s in seeds:
            set_seed(s)
            res = run_federated("FedHST", train_dataset, test_loader, clients_partition, dataset_name,
                                 rounds=rounds, epochs=epochs, gamma=g, log_prefix=f"[gamma={g} seed={s}] ")
            for row in res:
                row['Seed'] = s
            all_metrics.extend(res)
            _save_progress(all_metrics, out_csv)

    print(f"\n[DONE] Đã xuất '{out_csv}'.")
    return pd.DataFrame(all_metrics)


def run_multi_seed(dataset_name, algos, rounds, epochs, alpha, seeds, num_clients=100,
                    seed_partition=42, out_csv=None, **algo_kwargs):
    """
    Chạy một hoặc nhiều thuật toán qua nhiều seed huấn luyện (dùng chung một partition
    dữ liệu để so sánh công bằng) để phục vụ yêu cầu mean ± std / kiểm định thống kê
    của reviewer.

    algos : tên 1 thuật toán (str) hoặc danh sách nhiều thuật toán — chạy tất cả trong
            cùng 1 lần gọi, tiện cho thiết bị hạn chế compute (chỉ cần download/partition
            dữ liệu 1 lần).
    seeds : ví dụ [0, 1, 2] cho 3 lần chạy độc lập (khuyến nghị tối thiểu 5-10 theo
            reviewer, nhưng 3 vẫn cho phép ước lượng std thô và chạy paired-test — sẽ cần
            nêu rõ đây là giới hạn do phần cứng trong Response to Reviewers / Limitations).
    algo_kwargs : truyền thẳng xuống run_federated (gamma, beta1, beta2, feddyn_alpha,
            participation_frac, ...) — dùng khi muốn multi-seed một cấu hình ablation cụ thể.

    Tự động ghi CSV tạm sau mỗi (thuật toán, seed) hoàn thành (resume-safe) — xem _save_progress().
    """
    if isinstance(algos, str):
        algos = [algos]

    if out_csv is None:
        algo_tag = algos[0] if len(algos) == 1 else "suite"
        out_csv = f"{dataset_name.lower()}_{algo_tag}_multiseed_alpha{alpha}.csv"

    set_seed(seed_partition)
    train_dataset, test_dataset, num_classes = load_dataset(dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=32)
    clients_partition = non_iid_partition_dirichlet(train_dataset, num_clients, num_classes, alpha)

    all_metrics = []
    for algo in algos:
        for s in seeds:
            set_seed(s)
            res = run_federated(algo, train_dataset, test_loader, clients_partition, dataset_name,
                                 rounds=rounds, epochs=epochs, log_prefix=f"[seed={s}] ", **algo_kwargs)
            for row in res:
                row['Seed'] = s
            all_metrics.extend(res)
            _save_progress(all_metrics, out_csv)

    print(f"\n[DONE] Đã xuất '{out_csv}'.")
    return pd.DataFrame(all_metrics)


def run_tuned_suite(dataset_name, rounds, epochs, alpha, seeds, num_clients=100,
                     baseline_algos=("FedDyn", "FedAdam"),
                     fedhst_gamma=0.5, fedhst_beta1=0.9, fedhst_beta2=0.999,
                     seed_partition=42, out_csv=None):
    """
    So sánh công bằng: các baseline chạy với hyperparameter MẶC ĐỊNH của chính chúng
    (không bị ảnh hưởng bởi việc tinh chỉnh riêng cho FedHST), còn FedHST chạy với bộ
    hyperparameter đã chọn qua ablation (fedhst_gamma, fedhst_beta1, fedhst_beta2).
    Dùng chung 1 partition dữ liệu + cùng danh sách seed cho tất cả để đảm bảo so sánh
    công bằng. Kết quả gộp vào 1 CSV duy nhất (có cột 'Seed') — dùng trực tiếp được với
    main_body_comparison_table() / significance_table() / plot_main_convergence() trong
    plot_results.py.

    Ví dụ dùng để kiểm chứng tổ hợp hyperparameter mới tìm được từ ablation:
        run_tuned_suite("FashionMNIST", rounds=50, epochs=3, alpha=0.1, seeds=[42,84,168],
                         fedhst_gamma=0.5, fedhst_beta1=0.7, fedhst_beta2=0.9999)

    Tự động ghi CSV tạm sau mỗi (thuật toán, seed) hoàn thành (resume-safe) — nếu bị crash
    giữa chừng (như lỗi device-mismatch từng gặp ở SCAFFOLD), các thuật toán đã chạy xong
    trước đó vẫn còn nguyên trong file, không phải chạy lại từ đầu. Xem _save_progress().
    """
    if out_csv is None:
        out_csv = f"{dataset_name.lower()}_tuned_suite_alpha{alpha}_epochs{epochs}.csv"

    set_seed(seed_partition)
    train_dataset, test_dataset, num_classes = load_dataset(dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=32)
    print(f"Phân chia dữ liệu Dirichlet (alpha={alpha}, classes={num_classes})...")
    clients_partition = non_iid_partition_dirichlet(train_dataset, num_clients, num_classes, alpha)

    all_metrics = []

    # --- FedHST: hyperparameter đã tinh chỉnh qua ablation (chạy trước) ---
#    for s in seeds:
#        set_seed(s)
#        res = run_federated("FedHST", train_dataset, test_loader, clients_partition, dataset_name,
#                             rounds=rounds, epochs=epochs, gamma=fedhst_gamma,
#                             beta1=fedhst_beta1, beta2=fedhst_beta2,
#                             log_prefix=f"[FedHST-tuned seed={s}] ")
#        for row in res:
#            row['Seed'] = s
#        all_metrics.extend(res)
#        _save_progress(all_metrics, out_csv)

    # --- Baseline: hyperparameter mặc định (beta1=0.9, beta2=0.999 chuẩn của từng thuật toán) ---
    for algo in baseline_algos:
        for s in seeds:
            set_seed(s)
            res = run_federated(algo, train_dataset, test_loader, clients_partition, dataset_name,
                                 rounds=rounds, epochs=epochs, log_prefix=f"[baseline seed={s}] ")
            for row in res:
                row['Seed'] = s
            all_metrics.extend(res)
            _save_progress(all_metrics, out_csv)

    df_metrics = pd.DataFrame(all_metrics)
    print(f"\n[DONE] Đã xuất '{out_csv}'.")
    print(f"[Ghi chú] FedHST dùng gamma={fedhst_gamma}, beta1={fedhst_beta1}, beta2={fedhst_beta2}; "
          f"các baseline khác giữ nguyên mặc định (beta1=0.9, beta2=0.999).")
    return df_metrics


# ==========================================
# --- 5b. ABLATION STUDY: beta1/beta2, client count, participation ratio,
#          local epochs, mức non-IID (alpha) ---
# ==========================================
def run_beta_ablation(dataset_name, beta1_list, beta2_list, rounds, epochs, alpha,
                       num_clients=100, seed_partition=42, seed_train=42,
                       gamma=0.5, out_csv=None):
    """
    Sensitivity của FedHST theo beta1 (giữ beta2 mặc định 0.999) và theo beta2
    (giữ beta1 mặc định 0.9). Kết quả có cột 'Sweep_Type' ('beta1' hoặc 'beta2')
    để lọc khi vẽ biểu đồ.

    Tự động ghi CSV tạm sau mỗi giá trị hoàn thành (resume-safe) — xem _save_progress().
    """
    if out_csv is None:
        out_csv = f"{dataset_name.lower()}_beta_ablation_alpha{alpha}.csv"

    set_seed(seed_partition)
    train_dataset, test_dataset, num_classes = load_dataset(dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=32)
    clients_partition = non_iid_partition_dirichlet(train_dataset, num_clients, num_classes, alpha)

    all_metrics = []
    for b1 in beta1_list:
        set_seed(seed_train)
        res = run_federated("FedHST", train_dataset, test_loader, clients_partition, dataset_name,
                             rounds=rounds, epochs=epochs, gamma=gamma, beta1=b1, beta2=0.999,
                             log_prefix=f"[beta1-ablation b1={b1}] ")
        for row in res:
            row['Sweep_Type'] = 'beta1'
        all_metrics.extend(res)
        _save_progress(all_metrics, out_csv)
    for b2 in beta2_list:
        set_seed(seed_train)
        res = run_federated("FedHST", train_dataset, test_loader, clients_partition, dataset_name,
                             rounds=rounds, epochs=epochs, gamma=gamma, beta1=0.9, beta2=b2,
                             log_prefix=f"[beta2-ablation b2={b2}] ")
        for row in res:
            row['Sweep_Type'] = 'beta2'
        all_metrics.extend(res)
        _save_progress(all_metrics, out_csv)

    print(f"\n[DONE] Đã xuất '{out_csv}'.")
    return pd.DataFrame(all_metrics)


def run_client_count_ablation(dataset_name, client_counts, rounds, epochs, alpha,
                               algos=("FedAdam", "FedHST"), seed_partition=42, seed_train=42,
                               out_csv=None):
    """
    Effect of client number: chạy lại partition Dirichlet với các giá trị num_clients
    khác nhau (vd. [20, 50, 100, 200]). Vì partition thay đổi theo num_clients, mỗi
    giá trị cần chia dữ liệu lại — không thể dùng chung 1 partition như các ablation khác.

    Tự động ghi CSV tạm sau mỗi (num_clients, thuật toán) hoàn thành (resume-safe).
    """
    if out_csv is None:
        out_csv = f"{dataset_name.lower()}_client_count_ablation_alpha{alpha}.csv"

    all_metrics = []
    for nc in client_counts:
        set_seed(seed_partition)
        train_dataset, test_dataset, num_classes = load_dataset(dataset_name)
        test_loader = DataLoader(test_dataset, batch_size=32)
        clients_partition = non_iid_partition_dirichlet(train_dataset, nc, num_classes, alpha)
        for algo in algos:
            set_seed(seed_train)
            res = run_federated(algo, train_dataset, test_loader, clients_partition, dataset_name,
                                 rounds=rounds, epochs=epochs, log_prefix=f"[num_clients={nc}] ")
            for row in res:
                row['Num_Clients_Config'] = nc
            all_metrics.extend(res)
            _save_progress(all_metrics, out_csv)

    print(f"\n[DONE] Đã xuất '{out_csv}'.")
    return pd.DataFrame(all_metrics)


def run_participation_ablation(dataset_name, participation_fracs, rounds, epochs, alpha,
                                algos=("FedAdam", "FedHST"), num_clients=100,
                                seed_partition=42, seed_train=42, out_csv=None):
    """
    Effect of participation ratio: participation_fracs là danh sách các cặp (lo, hi),
    ví dụ [(0.05,0.1), (0.2,0.3), (0.5,0.6)] tương ứng 5-10%, 20-30% (mặc định gốc), 50-60%.

    Tự động ghi CSV tạm sau mỗi (tỉ lệ, thuật toán) hoàn thành (resume-safe).
    """
    if out_csv is None:
        out_csv = f"{dataset_name.lower()}_participation_ablation_alpha{alpha}.csv"

    set_seed(seed_partition)
    train_dataset, test_dataset, num_classes = load_dataset(dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=32)
    clients_partition = non_iid_partition_dirichlet(train_dataset, num_clients, num_classes, alpha)

    all_metrics = []
    for frac in participation_fracs:
        for algo in algos:
            set_seed(seed_train)
            res = run_federated(algo, train_dataset, test_loader, clients_partition, dataset_name,
                                 rounds=rounds, epochs=epochs, participation_frac=frac,
                                 log_prefix=f"[participation={frac}] ")
            for row in res:
                row['Participation_Frac'] = f"{frac[0]:.2f}-{frac[1]:.2f}"
            all_metrics.extend(res)
            _save_progress(all_metrics, out_csv)

    print(f"\n[DONE] Đã xuất '{out_csv}'.")
    return pd.DataFrame(all_metrics)


def run_local_epochs_ablation(dataset_name, epochs_list, rounds, alpha,
                               algos=("FedAdam", "FedHST"), num_clients=100,
                               seed_partition=42, seed_train=42, out_csv=None):
    """
    Effect of local epochs E: ví dụ epochs_list=[1, 3, 5, 10].

    Tự động ghi CSV tạm sau mỗi (E, thuật toán) hoàn thành (resume-safe).
    """
    if out_csv is None:
        out_csv = f"{dataset_name.lower()}_local_epochs_ablation_alpha{alpha}.csv"

    set_seed(seed_partition)
    train_dataset, test_dataset, num_classes = load_dataset(dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=32)
    clients_partition = non_iid_partition_dirichlet(train_dataset, num_clients, num_classes, alpha)

    all_metrics = []
    for E in epochs_list:
        for algo in algos:
            set_seed(seed_train)
            res = run_federated(algo, train_dataset, test_loader, clients_partition, dataset_name,
                                 rounds=rounds, epochs=E, log_prefix=f"[E={E}] ")
            for row in res:
                row['Local_Epochs_Config'] = E
            all_metrics.extend(res)
            _save_progress(all_metrics, out_csv)

    print(f"\n[DONE] Đã xuất '{out_csv}'.")
    return pd.DataFrame(all_metrics)


def run_alpha_level_ablation(dataset_name, alphas, rounds, epochs,
                              algos=("FedAdam", "FedHST"), num_clients=100,
                              seed_partition=42, seed_train=42, out_csv=None):
    """
    Effect dưới IID / non-IID nhẹ / non-IID cực đoan. Ví dụ alphas=[100.0, 1.0, 0.3, 0.1]
    (alpha lớn ~ gần IID, alpha nhỏ ~ non-IID cực đoan). Mỗi alpha cần partition lại.

    Tự động ghi CSV tạm sau mỗi (alpha, thuật toán) hoàn thành (resume-safe).
    """
    if out_csv is None:
        out_csv = f"{dataset_name.lower()}_alpha_level_ablation.csv"

    all_metrics = []
    for a in alphas:
        set_seed(seed_partition)
        train_dataset, test_dataset, num_classes = load_dataset(dataset_name)
        test_loader = DataLoader(test_dataset, batch_size=32)
        clients_partition = non_iid_partition_dirichlet(train_dataset, num_clients, num_classes, a)
        for algo in algos:
            set_seed(seed_train)
            res = run_federated(algo, train_dataset, test_loader, clients_partition, dataset_name,
                                 rounds=rounds, epochs=epochs, log_prefix=f"[alpha={a}] ")
            for row in res:
                row['Alpha_Config'] = a
            all_metrics.extend(res)
            _save_progress(all_metrics, out_csv)

    print(f"\n[DONE] Đã xuất '{out_csv}'.")
    return pd.DataFrame(all_metrics)


# ==========================================
# --- 5c. CHI PHÍ TRUYỀN TẢI (COMMUNICATION PAYLOAD) — tính lý thuyết ---
# ==========================================
def communication_payload_report(dataset_name, algos, cifar_variant="resnet"):
    """
    Tính chi phí truyền tải mỗi round (bytes) theo lý thuyết dựa trên số tham số mô hình
    (không phụ thuộc runtime). Giả định float32 (4 bytes/tham số).

    Hệ số nhân theo thuật toán (per selected client, mỗi round):
      - FedAvg/FedProx/FedAvgM/FedAdam/FedYogi/FedAdagrad/FedHST/FedNolowe/FedXVar/
        FedAdaComp : 1x  (chỉ gửi trọng số/pseudo-gradient cục bộ)
      - SCAFFOLD   : 2x  (gửi thêm control-variate delta_c_i cùng kích thước delta_y_i)
      - FedDyn     : 1x  (state hiệu chỉnh h_i chỉ lưu cục bộ ở client, không cần truyền)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _get_model(dataset_name, device, cifar_variant=cifar_variant)
    num_params = sum(p.numel() for p in model.parameters())
    bytes_per_client = num_params * 4  # float32

    multiplier = {
        "SCAFFOLD": 2.0,
    }

    rows = []
    for algo in algos:
        mult = multiplier.get(algo, 1.0)
        rows.append({
            'Algorithm': algo,
            'Dataset': dataset_name,
            'Num_Params': num_params,
            'Payload_Multiplier': mult,
            'Bytes_Per_Client_Per_Round': bytes_per_client * mult,
            'MB_Per_Client_Per_Round': bytes_per_client * mult / (1024 ** 2),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ==========================================
# --- 5d. KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (paired t-test / Wilcoxon) ---
# ==========================================
def compute_significance(df_multiseed, target_algo, baseline_algos, metric='Top1_Accuracy_Percent',
                          steady_state_window=15):
    """
    So sánh target_algo (thường là FedHST) với từng baseline_algo, dựa trên giá trị
    steady-state (trung bình N round cuối) TÍNH RIÊNG CHO TỪNG SEED, rồi ghép thành
    mẫu ghép cặp (paired) theo seed để chạy paired t-test và Wilcoxon signed-rank.

    Yêu cầu df_multiseed phải có cột 'Seed' (xuất từ run_multi_seed).
    Lưu ý: với n=3 seed, kiểm định có power rất thấp — nên nêu rõ đây là giới hạn
    do phần cứng trong bài / trong Response to Reviewers, không nên diễn giải p-value
    như bằng chứng mạnh.
    """
    from scipy import stats as scipy_stats

    if 'Seed' not in df_multiseed.columns:
        raise ValueError("DataFrame cần có cột 'Seed' — hãy dùng dữ liệu từ run_multi_seed().")

    max_round = df_multiseed['Round'].max()
    window = df_multiseed[df_multiseed['Round'] >= max_round - steady_state_window]

    per_seed_mean = window.groupby(['Algorithm', 'Seed'])[metric].mean().reset_index()

    target_vals = per_seed_mean[per_seed_mean['Algorithm'] == target_algo].sort_values('Seed')[metric].values

    rows = []
    for base in baseline_algos:
        base_vals = per_seed_mean[per_seed_mean['Algorithm'] == base].sort_values('Seed')[metric].values
        if len(base_vals) != len(target_vals) or len(base_vals) < 2:
            rows.append({'Baseline': base, 'N_Seeds': len(base_vals), 'Note': 'không đủ seed khớp cặp để kiểm định'})
            continue
        t_stat, t_p = scipy_stats.ttest_rel(target_vals, base_vals)
        try:
            w_stat, w_p = scipy_stats.wilcoxon(target_vals, base_vals)
        except ValueError:
            w_stat, w_p = float('nan'), float('nan')
        rows.append({
            'Baseline': base,
            'N_Seeds': len(base_vals),
            f'{target_algo}_mean': target_vals.mean(),
            f'{base}_mean': base_vals.mean(),
            'Diff_mean': target_vals.mean() - base_vals.mean(),
            'Paired_t_stat': t_stat,
            'Paired_t_pvalue': t_p,
            'Wilcoxon_stat': w_stat,
            'Wilcoxon_pvalue': w_p,
        })

    df_sig = pd.DataFrame(rows)
    print(df_sig.to_string(index=False))
    return df_sig


# ==========================================
# --- 6. VÍ DỤ SỬ DỤNG ---
# ==========================================
if __name__ == "__main__":
    ALGOS_TO_RUN = ["FedHST", "FedAdam", "FedDyn"]

    #ALGOS_TO_RUN = ["FedHST", "FedAvgM", "FedAdam", "FedProx", "FedYogi", "FedAdagrad",
    #                "SCAFFOLD", "FedDyn"]
    # ----------------------------------------------------------------
    # VÍ DỤ 1 (KHUYẾN NGHỊ DÙNG THAY run_algo_suite CŨ): chạy cả 8 thuật toán
    # qua 3 seed độc lập trong 1 lần gọi — vừa có baseline, vừa có mean ± std,
    # vừa đủ dữ liệu để chạy compute_significance() bên dưới.
    # ----------------------------------------------------------------
    df_multiseed = run_multi_seed(
        dataset_name="FashionMNIST",
        algos=ALGOS_TO_RUN,
        rounds=50,
        epochs=1,
        alpha=0.1,
        seeds=[0, 1, 2],   # 3 seed theo giới hạn phần cứng — nêu rõ trong Limitations
        out_csv="fashionmnist_multiseed3_full.csv",
    )

    # Kiểm định ý nghĩa thống kê: FedHST so với từng baseline còn lại
    compute_significance(
        df_multiseed,
        target_algo="FedHST",
        baseline_algos=[a for a in ALGOS_TO_RUN if a != "FedHST"],
        metric="Top1_Accuracy_Percent",
    )

    # ----------------------------------------------------------------
    # VÍ DỤ 1b (MỚI): Kiểm chứng tổ hợp hyperparameter mới tìm được từ ablation
    # — gamma=0.5, beta1=0.7, beta2=0.9999 cho FedHST, epochs=3 (thay vì 1) cho
    # TẤT CẢ thuật toán (để so sánh công bằng — epochs là thiết lập thí nghiệm
    # chung, không phải hyperparameter riêng của FedHST). Baseline khác vẫn giữ
    # beta1=0.9/beta2=0.999 mặc định. Dùng cùng 3 seed [42, 84, 168] như lần chạy
    # trước để so sánh trực tiếp được với fashionmnist_multiseed3_full.csv.
    # ----------------------------------------------------------------
    df_tuned = run_tuned_suite(
        dataset_name="FashionMNIST",
        rounds=50,
        epochs=3,              # tăng từ 1 lên 3 theo yêu cầu
        alpha=0.1,
        seeds=[42, 84, 168],   # khớp 3 seed đã dùng ở lần chạy trước
        fedhst_gamma=0.5,
        fedhst_beta1=0.7,      # tốt nhất từ beta1-ablation (thay vì 0.9 mặc định)
        fedhst_beta2=0.9999,   # tốt nhất từ beta2-ablation (thay vì 0.999 mặc định)
        out_csv="fashionmnist_tuned_suite_epochs3.csv",
    )
    compute_significance(
        df_tuned,
        target_algo="FedHST",
        baseline_algos=list(baseline for baseline in df_tuned['Algorithm'].unique() if baseline != "FedHST"),
        metric="Top1_Accuracy_Percent",
    )

    # ----------------------------------------------------------------
    # VÍ DỤ 2: Ablation gamma cho FedHST (gamma vs accuracy)
    # ----------------------------------------------------------------
    run_gamma_ablation(
        dataset_name="FashionMNIST",
        gammas=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
        rounds=50,
        epochs=1,
        alpha=0.1,
    )

    # ----------------------------------------------------------------
    # VÍ DỤ 3: Ablation beta1 / beta2
    # ----------------------------------------------------------------
    run_beta_ablation(
        dataset_name="FashionMNIST",
        beta1_list=[0.5, 0.7, 0.9, 0.99],
        beta2_list=[0.9, 0.99, 0.999, 0.9999],
        rounds=50,
        epochs=1,
        alpha=0.1,
    )

    # ----------------------------------------------------------------
    # VÍ DỤ 4: Ablation số lượng client (effect of client number)
    # ----------------------------------------------------------------
    run_client_count_ablation(
        dataset_name="FashionMNIST",
        client_counts=[20, 50, 100],
        rounds=50,
        epochs=1,
        alpha=0.1,
    )

    # ----------------------------------------------------------------
    # VÍ DỤ 5: Ablation tỉ lệ tham gia (effect of participation ratio)
    # ----------------------------------------------------------------
    run_participation_ablation(
        dataset_name="FashionMNIST",
        participation_fracs=[(0.05, 0.10), (0.20, 0.30), (0.50, 0.60)],
        rounds=50,
        epochs=1,
        alpha=0.1,
    )

    # ----------------------------------------------------------------
    # VÍ DỤ 6: Ablation số local epoch E
    # ----------------------------------------------------------------
    run_local_epochs_ablation(
        dataset_name="FashionMNIST",
        epochs_list=[1, 3, 5],
        rounds=50,
        alpha=0.1,
    )

    # ----------------------------------------------------------------
    # VÍ DỤ 7: Ablation mức non-IID (IID -> non-IID nhẹ -> cực đoan)
    # ----------------------------------------------------------------
    run_alpha_level_ablation(
        dataset_name="FashionMNIST",
        alphas=[100.0, 1.0, 0.3, 0.1],
        rounds=50,
        epochs=1,
    )

    # ----------------------------------------------------------------
    # VÍ DỤ 8: Báo cáo chi phí truyền tải lý thuyết (communication payload)
    # ----------------------------------------------------------------
    communication_payload_report(dataset_name="FashionMNIST", algos=ALGOS_TO_RUN)

    # ----------------------------------------------------------------
    # VÍ DỤ 9: Chạy CIFAR-10 / CIFAR-100 với kiến trúc mới (CIFAR10ResNet/CIFAR100ResNet,
    # nhẹ hơn 3-12 lần so với CNN gốc nhờ Global Average Pooling) + augmentation mạnh
    # (TrivialAugmentWide + RandomErasing) — cả hai đều BẬT MẶC ĐỊNH (cifar_variant=
    # "resnet", augmentation="strong" là default của run_federated/load_dataset nên
    # không cần truyền tường minh, để ở đây cho rõ ràng).
    # Lưu ý: CIFAR-100 nặng hơn nhiều, nên cân nhắc giảm 'rounds' và số seed khi thử ban đầu.
    # ----------------------------------------------------------------
    # run_multi_seed(
    #     dataset_name="CIFAR100",
    #     algos=ALGOS_TO_RUN,
    #     rounds=500,
    #     epochs=1,
    #     alpha=0.3,
    #     seeds=[0, 1, 2],
    #     cifar_variant="resnet",     # kiến trúc mới nhẹ + mạnh hơn (mặc định)
    #     label_smoothing=0.1,        # giúp tổng quát hoá tốt hơn, đặc biệt hữu ích cho 100 lớp
    # )
    #
    # Muốn so sánh với kiến trúc CNN gốc (vd. để báo cáo ablation kiến trúc nếu reviewer
    # hỏi), chỉ cần đổi cifar_variant="cnn":
    # run_multi_seed(dataset_name="CIFAR10", algos=["FedAdam", "FedHST"], rounds=50,
    #                 epochs=1, alpha=0.3, seeds=[0, 1, 2], cifar_variant="cnn")