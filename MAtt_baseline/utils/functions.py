import torch
import torch.nn as nn
import sys
import os
sys.path.append("..")
from mAtt.optimizer import MixOptimizer
from sklearn.metrics import roc_auc_score as ras, silhouette_score
import numpy as np


def _move_to_device(xb, yb, device):
    # GPU 训练用 pin_memory + non_blocking 实现 CPU→GPU 异步复制，
    # 避免每次 batch 阻塞等待拷贝完成（DataLoader 已开 pin_memory 时收益最大）。
    nb = device.type == 'cuda'
    return xb.float().to(device, non_blocking=nb), yb.long().to(device, non_blocking=nb)


def _silhouette_with_pca(feats, labels, max_components=15, var_ratio=0.9):
    """A22: 切空间特征 PCA 降维（保留 90% 方差、上限 15 维）后再算 silhouette。

    度量说明: 切空间向量在单位切点处的欧氏距离 = SPD 流形上的 Log-Euclidean
    距离（切空间映射即 logm），故降维后用欧氏 silhouette 计算的正是 LE 度量
    下的聚类质量；PCA 是该度量最优线性降维，避免高维欧氏距离的稀疏退化。
    """
    from sklearn.decomposition import PCA
    n_cap = min(max_components, feats.shape[0] - 1, feats.shape[1])
    if n_cap < 2:
        return silhouette_score(feats, labels)
    pca = PCA(n_components=n_cap, random_state=0)
    reduced = pca.fit_transform(feats)
    # 按 90% 累计方差截断
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_keep = max(2, int(np.searchsorted(cum, var_ratio) + 1))
    return silhouette_score(reduced[:, :n_keep], labels)


def trainNetwork(net, trainloader, validloader, testloader, model_path=None, iterations=500, lr=5*1e-4, wd=None, repeat=None, sub=None, epochs=None, dataset=None, patience=0, silhouette_interval=0, grad_clip=1.0, use_cosine_schedule=True):
    device = next(net.parameters()).device
    CE = nn.CrossEntropyLoss()
    base_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)
    optimizer = MixOptimizer(base_optimizer)
    # A15: 余弦退火 —— 前期大步长快速下降、后期小步长精细收敛
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=iterations) if use_cosine_schedule else None
    bestLoss = float('inf')
    bestAcc = 0.0
    best_loss_path = None
    best_acc_path = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'silhouette': []}
    patience_counter = 0
    enable_early_stop = (patience is not None and patience > 0)
    enable_silhouette = (silhouette_interval is not None and silhouette_interval > 0)

    for ite in range(iterations):
        net.train()
        acc_tr = 0
        tr_len = 0
        train_loss_sum = 0.0
        train_batches = 0

        for xb, yb in trainloader:
            xb, yb = _move_to_device(xb, yb, device)
            tr_len += yb.shape[0]
            out = net(xb)
            loss = CE(out, yb)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                # 注意力分数 α = -d² 不做压缩变换，靠裁剪梯度范数防止距离尺度爆炸导致训练崩溃
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1
            acc_tr += (torch.argmax(out, dim=1) == yb).sum().item()

        net.eval()
        TL = 0.0
        acc_val = 0
        val_len = 0
        with torch.no_grad():
            for xb, yb in validloader:
                xb, yb = _move_to_device(xb, yb, device)
                val_len += yb.shape[0]
                out = net(xb)
                TL += CE(out, yb).item()
                acc_val += (torch.argmax(out, dim=1) == yb).sum().item()

        avg_train_loss = train_loss_sum / max(train_batches, 1)
        avg_val_loss = TL / max(val_len, 1)
        train_acc = acc_tr / max(tr_len, 1)
        val_acc = acc_val / max(val_len, 1)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Silhouette Score（可选，每 N 轮计算一次）
        sil = float('nan')
        if enable_silhouette and ite % silhouette_interval == 0:
            try:
                feats, labels = [], []
                with torch.no_grad():
                    for xb_s, yb_s in validloader:
                        xb_s, yb_s = _move_to_device(xb_s, yb_s, device)
                        feats.append(net.get_features(xb_s).cpu().numpy())
                        labels.append(yb_s.cpu().numpy())
                feats = np.concatenate(feats, axis=0)
                labels = np.concatenate(labels, axis=0)
                if len(np.unique(labels)) >= 2:
                    sil = _silhouette_with_pca(feats, labels)
            except Exception:
                pass
        history['silhouette'].append(sil)

        if scheduler is not None:
            scheduler.step()

        print('')
        print(f'Iteration{ite}=====')
        print(f'train_loss:{avg_train_loss:.4f}    val_loss:{avg_val_loss:.4f}')
        print(f'train_acc:{train_acc:.4f}    val_acc:{val_acc:.4f}')
        if enable_silhouette:
            print(f'silhouette:{sil:.4f}')

        if not os.path.exists(model_path):
            os.makedirs(model_path)
            print(f'create {model_path}.......')

        improved = False

        # 保存验证 Loss 最低的模型（A17: 保存 state_dict，加载端重建模型）
        if avg_val_loss < bestLoss:
            bestLoss = avg_val_loss
            if dataset is not None:
                best_loss_path = os.path.join(model_path, f'{dataset}_repeat{repeat}_sub{sub}_epochs{epochs}_lr{lr}_wd{wd}_best_loss.pt')
            else:
                best_loss_path = os.path.join(model_path, f'repeat{repeat}_sub{sub}_epochs{epochs}_lr{lr}_wd{wd}_best_loss.pt')
            print(f'saving best loss model to {best_loss_path}')
            torch.save(net.state_dict(), best_loss_path)

            # 当前权重即所存权重，直接测试（省去整模型 load）
            net.eval()
            test_acc = testNetwork(net, testloader)
            print(f'test_acc (best loss):{test_acc}')
            improved = True

        # 保存验证准确率最高的模型
        if val_acc > bestAcc:
            bestAcc = val_acc
            if dataset is not None:
                best_acc_path = os.path.join(model_path, f'{dataset}_repeat{repeat}_sub{sub}_epochs{epochs}_lr{lr}_wd{wd}_best_acc.pt')
            else:
                best_acc_path = os.path.join(model_path, f'repeat{repeat}_sub{sub}_epochs{epochs}_lr{lr}_wd{wd}_best_acc.pt')
            print(f'saving best acc model to {best_acc_path}')
            torch.save(net.state_dict(), best_acc_path)

            net.eval()
            test_acc = testNetwork(net, testloader)
            print(f'test_acc (best acc):{test_acc}')
            improved = True

        # 早停
        if enable_early_stop:
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                print(f'no improvement for {patience_counter} iterations (patience={patience})')
                if patience_counter >= patience:
                    print(f'early stopping at iteration {ite}')
                    break

    return history, best_loss_path, best_acc_path


def testNetwork(net, testloader):
    device = next(net.parameters()).device
    net.eval()
    acc = 0
    total = 0
    with torch.no_grad():
        for xb, yb in testloader:
            xb, yb = _move_to_device(xb, yb, device)
            pred = net(xb)
            acc += (torch.argmax(pred, dim=1) == yb).sum().item()
            total += yb.shape[0]

    return acc / max(total, 1)


def testNetwork_auc(net, testloader):
    device = next(net.parameters()).device
    net.eval()
    y_pred = torch.empty(0, device=device)
    y_true = torch.empty(0, device=device)
    with torch.no_grad():
        for xb, yb in testloader:
            xb, yb = _move_to_device(xb, yb, device)
            pred = net(xb)
            y_pred = torch.cat((y_pred, pred[:, 1]), 0)
            y_true = torch.cat((y_true, yb.float()), 0)

    return ras(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())


# ============================================================
# 路径解析 & 可视化数据保存
# （从 run_matt_physionetmi.py 移入，消除跨文件导入依赖）
# ============================================================


BCI2A_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]


def resolve_path(path):
    """相对路径按脚本所在目录解析；绝对路径保持不变"""
    from pathlib import Path
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((Path(__file__).resolve().parent.parent / p).resolve())


def save_visualization_data(net, testloader, history, test_acc, args, criterion='best_loss'):
    """保存可视化所需数据到 npz（按 criterion 命名，覆盖前一次）"""
    from pathlib import Path
    import numpy as np

    figures_dir = Path(args['model_path']).resolve().parent / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_path = figures_dir / f'matt_visualization_data_{criterion}.npz'

    sub_str = str(args.get('sub', 'unknown'))
    title_suffix = (
        f'数据集: {args["dataset"].upper()} | '
        f'被试: {sub_str} | '
        f'E2R分段: {args["epochs"]} | '
        f'迭代: {args["iterations"]} | '
        f'wd: {args["wd"]} | '
        f'模型: {criterion}'
    )

    net.eval()
    device = next(net.parameters()).device

    # ---- 收集一个 batch 的中间结果 ----
    xb, yb = next(iter(testloader))
    xb_raw = xb.float().to(device)
    yb_raw = yb.long().to(device)

    with torch.no_grad():
        x1 = net.conv1(xb_raw); x1 = net.In1(x1)
        x2 = net.conv2(x1); x2 = net.In2(x2)
        spd_matrices = net.ract1(x2)

        # 注意力概率直接复用模型内部实现（α=-d²、稀疏对、行归一化），避免与 forward 不同步
        atten_prob = net.att2.attention_prob(spd_matrices)

        logits = net(xb_raw)

        x_attn, shape = net.att2(spd_matrices)
        x_rect = net.ract2(x_attn)
        x_tangent = net.tangent(x_rect)
        x_flat_batch = net.flat(x_tangent.view(shape[0], shape[1], -1))

    # ---- 收集整个测试集 ----
    all_x_flat = []
    all_y_true = []
    all_y_pred = []
    all_spd = []
    all_attn = []
    with torch.no_grad():
        for xb_all, yb_all in testloader:
            xb_all = xb_all.float().to(device)
            out_all = net(xb_all)
            pred_all = out_all.argmax(dim=1)

            x1_all = net.conv1(xb_all); x1_all = net.In1(x1_all)
            x2_all = net.conv2(x1_all); x2_all = net.In2(x2_all)
            spd_all = net.ract1(x2_all)

            ap_all = net.att2.attention_prob(spd_all)

            x_attn_all, shape_all = net.att2(spd_all)
            x_rect_all = net.ract2(x_attn_all)
            x_tan_all = net.tangent(x_rect_all)
            x_flat_all = net.flat(x_tan_all.view(shape_all[0], shape_all[1], -1))

            all_x_flat.append(x_flat_all.cpu())
            all_y_true.append(yb_all)
            all_y_pred.append(pred_all.cpu())
            all_spd.append(spd_all.cpu())
            all_attn.append(ap_all.cpu())

    all_x_flat = torch.cat(all_x_flat, dim=0)
    all_y_true = torch.cat(all_y_true, dim=0)
    all_y_pred = torch.cat(all_y_pred, dim=0)
    all_spd = torch.cat(all_spd, dim=0)
    all_attn = torch.cat(all_attn, dim=0)

    channel_names = BCI2A_CHANNELS if args['dataset'] == 'bci2a' else [f'Ch{i+1}' for i in range(22)]
    sfreq = 250

    np.savez(
        data_path,
        history_train_loss=np.array(history['train_loss'], dtype=np.float32),
        history_val_loss=np.array(history['val_loss'], dtype=np.float32),
        history_train_acc=np.array(history['train_acc'], dtype=np.float32),
        history_val_acc=np.array(history['val_acc'], dtype=np.float32),
        history_silhouette=np.array(history.get('silhouette', []), dtype=np.float32),
        test_acc=float(test_acc),
        xb=xb_raw.cpu().numpy().astype(np.float32),
        yb=yb_raw.cpu().numpy().astype(np.int64),
        x2=x2.cpu().numpy().astype(np.float32),
        spd_matrices=spd_matrices.cpu().numpy().astype(np.float32),
        atten_prob=atten_prob.cpu().numpy().astype(np.float32),
        x_flat_batch=x_flat_batch.cpu().numpy().astype(np.float32),
        logits=logits.cpu().numpy().astype(np.float32),
        all_x_flat=all_x_flat.numpy().astype(np.float32),
        all_y_true=all_y_true.numpy().astype(np.int64),
        all_y_pred=all_y_pred.numpy().astype(np.int64),
        all_spd=all_spd.numpy().astype(np.float32),
        all_attn=all_attn.numpy().astype(np.float32),
        conv1_weight=net.conv1.weight.data.cpu().numpy().astype(np.float32),
        channel_names=np.array(channel_names),
        sfreq=float(sfreq),
        title_suffix=title_suffix
    )
    print(f'可视化数据已保存: {data_path}')


