import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import time
import os
from model import LSLF
from loss import CMSCLoss

# -------------------------------------------------------------------------------
parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', help='dataset to use', default='houston2013')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='frequency of testing')
parser.add_argument('--patches', type=int, default=7, help='number of patches')
parser.add_argument('--band_patches', type=int, default=3, help='number of related band')
parser.add_argument('--epoches', type=int, default=600, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight_decay')
parser.add_argument('--dim_reduction', choices=['h2m', 'var', 'kmeans','cor','pca','spa'], default='h2m')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的设备：{device}")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"当前可用的GPU数量：{num_gpus}")
    if num_gpus > 0:
        gpu_id = 0
        torch.cuda.set_device(gpu_id)
        print(f"正在使用的GPU：{gpu_id}")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
# -------------------------------------------------------------------------------
# 定位训练和测试样本
def choose_train_and_test_point(train_data, test_data, true_data, num_classes):
    def get_pos_and_count(data, max_label, include_bg=False):
        pos = {}

        # 确定标签起始值
        start_label = 0 if include_bg else 1

        for i in range(start_label, max_label + 1):
            each_class_pos = np.argwhere(data == i)
            pos[i] = each_class_pos

        # 拼接所有类别的坐标（不包括背景，除非是全图数据）
        total_pos_list = [pos[i] for i in range(start_label, max_label + 1)]
        if total_pos_list:
            total_pos = np.concatenate(total_pos_list, axis=0).astype(int)
        else:
            total_pos = np.array([], dtype=int).reshape(0, 2)

        # 统计数量
        number_counts = [pos[i].shape[0] for i in range(start_label, max_label + 1)]

        return total_pos, number_counts

    # 训练集 (标签从1到num_classes)
    total_pos_train, number_train = get_pos_and_count(train_data, num_classes, include_bg=False)

    # 测试集 (标签从1到num_classes)
    total_pos_test, number_test = get_pos_and_count(test_data, num_classes, include_bg=False)

    # 全图数据 (标签从0到num_classes, 0为背景)
    total_pos_true, number_true = get_pos_and_count(true_data, num_classes, include_bg=True)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true
# -------------------------------------------------------------------------------
# 边界拓展/边缘填充
def mirror_hsi(height, width, bands, data, patch):
    padding = patch // 2
    mirror = np.zeros((height + 2 * padding, width + 2 * padding, bands), dtype=float)

    # 中心
    mirror[padding:padding + height, padding:padding + width, :] = data

    # 上下
    mirror[:padding, padding:padding + width, :] = data[:padding, :, :][::-1, :, :]
    mirror[padding + height:, padding:padding + width, :] = data[-padding:, :, :][::-1, :, :]

    # 左右
    mirror[:, :padding, :] = mirror[:, padding:2 * padding, :][:, ::-1, :]
    mirror[:, padding + width:, :] = mirror[:, width:width + padding, :][:, ::-1, :]

    return mirror
# -------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, number_true, num_classes):
    # 训练和测试集的标签从 0 到 num_classes - 1
    y_train = np.concatenate([np.repeat(i, number_train[i]) for i in range(num_classes)])
    y_test = np.concatenate([np.repeat(i, number_test[i]) for i in range(num_classes)])

    # 全图数据的标签从 0 到 num_classes (0是背景，第0个元素是背景的数量)
    y_true = np.concatenate([np.repeat(i, number_true[i]) for i in range(num_classes + 1)])

    print(f"y_train: shape = {y_train.shape} ,type = {y_train.dtype}")
    print(f"y_test: shape = {y_test.shape} ,type = {y_test.dtype}")
    print(f"y_true: shape = {y_true.shape} ,type = {y_true.dtype}")
    print("**************************************************")
    return y_train.astype(np.int64), y_test.astype(np.int64), y_true.astype(np.int64)
# -------------------------------------------------------------------------------
# AverageMeter
class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
# -------------------------------------------------------------------------------
# 准确率计算
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # 使用 .reshape(-1) 确保正确求和
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        # 返回 top-1 预测值
        top1_pred = pred[0].squeeze()
        return res, target, top1_pred
# -------------------------------------------------------------------------------
# 评估指标计算
def output_metric(tar, pre):
    # 确保输入是正确维度的数组
    if hasattr(tar, 'dim') and tar.dim() == 0:
        tar = tar.unsqueeze(0)
    if hasattr(pre, 'dim') and pre.dim() == 0:
        pre = pre.unsqueeze(0)

    # 如果是PyTorch张量，转换为numpy数组
    if hasattr(tar, 'cpu'):
        tar = tar.cpu().numpy()
    if hasattr(pre, 'cpu'):
        pre = pre.cpu().numpy()

    # 确保是一维数组
    tar = np.ravel(tar)
    pre = np.ravel(pre)
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
# -------------------------------------------------------------------------------
# 混淆矩阵结果计算
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum_rc = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    total_samples = np.sum(matrix)

    for i in range(shape[0]):
        # 对角线元素（正确分类）
        number += matrix[i, i]

        # 每一类的准确率 (AA)
        row_sum = np.sum(matrix[i, :])
        AA[i] = matrix[i, i] / row_sum if row_sum != 0 else 0

        # 计算 Kappa 的分母 pe 的分子部分
        col_sum = np.sum(matrix[:, i])
        sum_rc += row_sum * col_sum

    OA = number / total_samples
    AA_mean = np.mean(AA)

    pe = sum_rc / (total_samples ** 2)
    Kappa = (OA - pe) / (1 - pe) if 1 - pe != 0 else 0

    return OA, AA_mean, Kappa, AA
# -------------------------------------------------------------------------------
# 数据生成器
class HSI_LiDAR_DataGenerator(Data.Dataset):
    def __init__(self, mirror_hsi, mirror_lidar, points, labels, patch_size, transform=None):
        self.mirror_hsi = mirror_hsi
        self.mirror_lidar = mirror_lidar
        self.points = points
        self.labels = labels
        self.patch_size = patch_size
        self.transform = transform
        self.padding = patch_size // 2

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        point = self.points[idx]
        x, y = point[0], point[1]

        # 切片范围是 [x : x + patch_size, y : y + patch_size]
        hsi_patch = self.mirror_hsi[x: x + self.patch_size, y: y + self.patch_size, :]
        lidar_patch = self.mirror_lidar[x: x + self.patch_size, y: y + self.patch_size, :]
        label = self.labels[idx]

        hsi_patch = np.transpose(hsi_patch, (2, 0, 1))
        lidar_patch = np.transpose(lidar_patch, (2, 0, 1))

        hsi_patch = torch.from_numpy(hsi_patch).float()
        lidar_patch = torch.from_numpy(lidar_patch).float()
        label = torch.tensor(label).long()

        if self.transform:
            hsi_patch = self.transform(hsi_patch)
            lidar_patch = self.transform(lidar_patch)

        return hsi_patch, lidar_patch, label
# -------------------------------------------------------------------------------
# 加载数据
if args.dataset == 'houston2013':
    data = loadmat('Data/houston2013/hsi_houston2013.mat')['input']
    label_data = loadmat('Data/houston2013/hsi_houston2013.mat')
    lidar = loadmat('Data/houston2013/lidar_houston2013.mat')['LiDAR_data']
    args.epoches = 600
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

TR = label_data['TR']
TE = label_data['TE']
num_classes = np.max(TR)
label = TR + TE

print(f'data shape: {data.shape}')
print(f'lidar shape: {lidar.shape}')

# 归一化 HSI data
hsi_normalize = np.zeros(data.shape)
for i in range(data.shape[2]):
    input_max = np.max(data[:, :, i])
    input_min = np.min(data[:, :, i])
    hsi_normalize[:, :, i] = (data[:, :, i] - input_min) / (input_max - input_min) if (input_max - input_min) != 0 else 0

# 归一化 LiDAR data
lidar_band_0 = lidar[:, :, 0] if lidar.ndim == 3 else lidar
lidar_max = np.max(lidar_band_0)
lidar_min = np.min(lidar_band_0)
if lidar_max - lidar_min != 0:
    lidar_normalize = (lidar_band_0 - lidar_min) / (lidar_max - lidar_min)
else:
    lidar_normalize = np.zeros_like(lidar_band_0)

lidar_normalize = lidar_normalize[:, :, np.newaxis]  # (H, W, 1)

height, width, band = hsi_normalize.shape
print("height={0},width={1},band={2}".format(height, width, band))
# -------------------------------------------------------------------------------
# 划分数据集
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = choose_train_and_test_point(
    TR, TE, label, num_classes)

# 边界填充
mirror_hsi_img = mirror_hsi(height, width, data.shape[2], hsi_normalize, patch=args.patches)
mirror_lidar_img = mirror_hsi(height, width, 1, lidar_normalize, patch=args.patches)

# 标签生成
y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
# -------------------------------------------------------------------------------
# DataLoader
Label_train = HSI_LiDAR_DataGenerator(mirror_hsi_img, mirror_lidar_img, total_pos_train, y_train,
                                      patch_size=args.patches)
Label_test = HSI_LiDAR_DataGenerator(mirror_hsi_img, mirror_lidar_img, total_pos_test, y_test, patch_size=args.patches)
Label_true = HSI_LiDAR_DataGenerator(mirror_hsi_img, mirror_lidar_img, total_pos_true, y_true, patch_size=args.patches)

label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True,
                                     num_workers=0)  # num_workers 设为 0 以防Windows下多进程问题
label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
label_true_loader = Data.DataLoader(Label_true, batch_size=100, shuffle=False, num_workers=0)
# -------------------------------------------------------------------------------
# Model
model = LSLF(
    image_size=args.patches,
    near_band=args.band_patches,
    num_band=16,
    total_band=band,
    num_classes=num_classes,
    dim=16,
    depth=3,
    heads=4,
    mlp_dim=8,
    dropout=0.1,
    emb_dropout=0.1,
)
model = model.to(device)
# -------------------------------------------------------------------------------
# 训练 epoch
def train_epoch(model, train_loader, criterion, optimizer, CMSCLoss, device):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for hsi_data, lidar_data, label in train_loader:
        hsi_data = hsi_data.to(device)
        lidar_data = lidar_data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred, x_loss, l = model(hsi_data, lidar_data)
        loss = criterion(pred, label) + 0.01*CMSCLoss(x_loss, l)

        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(pred, label, topk=(1,))
        n = hsi_data.shape[0]
        objs.update(loss.item(), n)
        top1.update(prec1[0].item(), n)
        tar = np.append(tar, t.cpu().numpy())
        pre = np.append(pre, p.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
# -------------------------------------------------------------------------------
# 验证 epoch
def valid_epoch(model, valid_loader, criterion, CMSCLoss, device):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    model.eval()
    with torch.no_grad():
        for hsi_data, lidar_data, label in valid_loader:
            hsi_data = hsi_data.to(device)
            lidar_data = lidar_data.to(device)
            label = label.to(device)

            pred, x_loss, l = model(hsi_data, lidar_data)
            loss = criterion(pred, label) + 0.01*CMSCLoss(x_loss, l)

            prec1, t, p = accuracy(pred, label, topk=(1,))
            n = hsi_data.shape[0]
            objs.update(loss.item(), n)
            top1.update(prec1[0].item(), n)
            tar = np.append(tar, t.cpu().numpy())
            pre = np.append(pre, p.cpu().numpy())

    return tar, pre, top1.avg, objs.avg
# -------------------------------------------------------------------------------
#测试 epoch
def test_epoch(model, test_loader, device):
    pre = np.array([])
    model.eval()
    with torch.no_grad():
        for hsi_data, lidar_data, _ in test_loader:
            hsi_data = hsi_data.to(device)
            lidar_data = lidar_data.to(device)
            pred, x_loss, l = model(hsi_data, lidar_data)
            _, pred = pred.topk(1, 1, True, True)
            pre = np.append(pre, pred.squeeze().cpu().numpy())
    return pre
# -------------------------------------------------------------------------------
# loss function
criterion = nn.CrossEntropyLoss().to(device)
CMSCLoss = CMSCLoss().to(device)

# optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10 if args.epoches // 10 > 0 else 1,
                                            gamma=args.gamma)
# -------------------------------------------------------------------------------
if args.flag_test == 'test':
    # =========================  测试模式  =========================
    # 加载已训练权重
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model_path = os.path.join(log_dir, f'{args.dataset}.pt')

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        raise FileNotFoundError(f"{model_path} not found, please train model first.")

    tar_v, pre_v, top1, objs = valid_epoch(model, label_test_loader, criterion, CMSCLoss, device)
    OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

    tar_v_int = tar_v.astype(int)
    pre_v_int = pre_v.astype(int)
    target_names = [f'Class {i + 1}' for i in range(num_classes)]

    # === 输出结果 ===
    print("===================================================")
    print("Final Test Results:")
    print("Test OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
    print("===================================================")
    
    # 输出详细的类别报告
    print("\n Per-Class Metrics (Classification Report):")
    print(classification_report(tar_v_int, pre_v_int, target_names=target_names, digits=4, zero_division=0))

    # === 绘制预测图 ===
    pre_u = test_epoch(model, label_true_loader, device)
    prediction_matrix = np.zeros((height, width), dtype=float)

    for i in range(total_pos_true.shape[0]):
        prediction_matrix[total_pos_true[i, 0], total_pos_true[i, 1]] = pre_u[i]

    import matplotlib.colors as colors

    if args.dataset == 'houston2013':
        cmap_tr = ['#000000', '#aec7e8', '#ff7f0e', '#2ca02c', '#98df8a', '#d62728', '#9467bd', '#c5b0d5',
                   '#8c564b', '#e377c2', '#f7b6d2', '#7f7f7f', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
    elif args.dataset == 'MUUFL':
        cmap_tr = ['#000000', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#8c564b', '#e377c2',
                   '#7f7f7f', '#bcbd22', '#17becf', '#9edae5']
    elif args.dataset == 'Trento':
        cmap_tr = ['#000000', '#ff7f0e', '#d62728', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
    else:
        raise ValueError
    cmap = colors.LinearSegmentedColormap.from_list('custom_cmap', cmap_tr)

    plt.figure(figsize=(30, 30))
    plt.imshow(prediction_matrix, cmap=cmap, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'prediction map/lslf_{args.dataset}.png', bbox_inches='tight', pad_inches=0, transparent=True,dpi=300)
    plt.show()
else:
    # =========================  训练模式  =========================
    print("start training")
    tic = time.time()
    best_acc = 0
    best_epoch = 0

    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for epoch in range(args.epoches):
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer, CMSCLoss, device)
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
        print("Epoch: {:03d} train_loss: {:.4f}, train_OA: {:.2f}, train_AA: {:.2f}, train_Kappa: {:.4f}".format(
            epoch + 1, train_obj, OA1 * 100, AA_mean1 * 100, Kappa1))

        if (epoch % args.test_freq == 0) or (epoch == args.epoches - 1):
            tar_v, pre_v, val_acc, val_obj = valid_epoch(model, label_test_loader, criterion, CMSCLoss, device)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            print("Epoch: {:03d} test_loss: {:.4f}, test_OA: {:.2f}, test_AA: {:.2f}, test_Kappa: {:.4f}".format(
                epoch + 1, val_obj, OA2 * 100, AA_mean2 * 100, Kappa2))

            # 保存最佳模型
            if (AA_mean2 * 100) > best_acc:
                best_acc = AA_mean2 * 100
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(log_dir, f'{args.dataset}.pt'))
                print(f'>>> 保存最佳模型到 {log_dir}/{args.dataset}.pt')
                print(f'>>> 最佳模型在 epoch {best_epoch}, AA: {best_acc:.4f}')

            # 记录最新的验证结果
            final_OA = OA2
            final_AA_mean = AA_mean2
            final_Kappa = Kappa2
            final_AA = AA2
        scheduler.step()
    toc = time.time()
    print("Running Time: {:.2f}s".format(toc - tic))

    # 最终结果输出
    print("**************************************************")
    print("Final result (Last Epoch):")
    print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(final_OA, final_AA_mean, final_Kappa))
    print("Per-Class AA (Last Epoch):", final_AA)
    print("**************************************************")
    print("Parameter:")

def print_args(args):
    for k, v in vars(args).items():
        print(f"{k}: {v}")

print_args(args)