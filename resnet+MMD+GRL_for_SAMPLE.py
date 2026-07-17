import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from PIL import Image
import numpy as np
import os
import yaml
import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 非交互式后端，避免显示问题
from datetime import datetime


def plot_confusion_matrix(true_labels, pred_labels, class_names=None, save_path=None):
    """
    绘制混淆矩阵（显示正确率百分比）
    """
    if hasattr(true_labels, 'cpu'):
        true_labels = true_labels.cpu().numpy()
    if hasattr(pred_labels, 'cpu'):
        pred_labels = pred_labels.cpu().numpy()

    true_labels = np.array(true_labels).flatten()
    pred_labels = np.array(pred_labels).flatten()

    num_classes = len(np.unique(np.concatenate([true_labels, pred_labels])))

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    cm = confusion_matrix(true_labels, pred_labels)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_percentage, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Percentage %)')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    thresh = cm_percentage.max() / 2.
    for i in range(cm_percentage.shape[0]):
        for j in range(cm_percentage.shape[1]):
            if cm[i, j] > 0:
                text = f'{cm_percentage[i, j]:.1f}%'
            else:
                text = '0.0%'
            plt.text(j, i, text,
                     ha="center", va="center",
                     color="white" if cm_percentage[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")

    class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100
    print("\n各类别准确率:")
    for i, acc in enumerate(class_accuracy):
        print(f'{class_names[i]}: {acc:.2f}%')

    overall_accuracy = np.sum(cm.diagonal()) / np.sum(cm) * 100
    print(f"\n总体准确率: {overall_accuracy:.2f}%")
    plt.close()

    return cm, cm_percentage


class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层"""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GRL, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_)


class SARDomainAdaptationResNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.4):
        super(SARDomainAdaptationResNet, self).__init__()

        self.num_classes = num_classes

        # 编码器：ResNet18
        resnet = models.resnet18(pretrained=True)

        self.encoder_stage0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu
        )  # 224→112, 64ch

        self.encoder_stage1 = nn.Sequential(
            resnet.maxpool, resnet.layer1, nn.Dropout(p=dropout_rate)
        )  # 112→56, 64ch

        self.encoder_stage2 = nn.Sequential(
            resnet.layer2,
            nn.Dropout(p=dropout_rate)
        )  # 56→28, 128ch

        self.encoder_stage3 = nn.Sequential(
            resnet.layer3,
            nn.Dropout(p=dropout_rate)
        )  # 28→14, 256ch

        self.encoder_stage4 = nn.Sequential(
            resnet.layer4,
            nn.Dropout(p=dropout_rate)
        )  # 14→7, 512ch

        # 瓶颈层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck_fc = nn.Linear(512, 256)

        # 分类头
        self.classifier = nn.Linear(256, num_classes)

        # GRL层
        self.grl = GRL(lambda_=1.0)

        # 域分类器
        self.domain_classifier_e3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.domain_classifier_e4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def compute_mmd(self, source, target):
        """线性核MMD的无偏估计"""
        K_XX = torch.mm(source, source.t()).mean()
        K_YY = torch.mm(target, target.t()).mean()
        K_XY = torch.mm(source, target.t()).mean()
        mmd = torch.exp(torch.abs(K_XX + K_YY - 2 * K_XY)) - 1
        return mmd

    def forward(self, sim_data=None, real_data=None):
        """前向传播"""
        results = {'sim': None, 'real': None}

        # 处理仿真数据
        if sim_data is not None:
            e0 = self.encoder_stage0(sim_data)
            e1 = self.encoder_stage1(e0)
            e2 = self.encoder_stage2(e1)
            e3 = self.encoder_stage3(e2)
            e4 = self.encoder_stage4(e3)

            gap = self.global_avg_pool(e4)
            gap = gap.flatten(1)
            domain_feat = self.bottleneck_fc(gap)

            e3_grl = self.grl(e3)
            e4_grl = self.grl(e4)

            domain_logits_e3 = self.domain_classifier_e3(e3_grl)
            domain_logits_e4 = self.domain_classifier_e4(e4_grl)

            class_logits = self.classifier(domain_feat)

            results['sim'] = {
                'class_logits': class_logits,
                'domain_feat': domain_feat,
                'domain_logits_e3': domain_logits_e3,
                'domain_logits_e4': domain_logits_e4,
            }

        # 处理真实数据
        if real_data is not None:
            e0 = self.encoder_stage0(real_data)
            e1 = self.encoder_stage1(e0)
            e2 = self.encoder_stage2(e1)
            e3 = self.encoder_stage3(e2)
            e4 = self.encoder_stage4(e3)

            gap = self.global_avg_pool(e4)
            gap = gap.flatten(1)
            domain_feat = self.bottleneck_fc(gap)

            e3_grl = self.grl(e3)
            e4_grl = self.grl(e4)

            domain_logits_e3 = self.domain_classifier_e3(e3_grl)
            domain_logits_e4 = self.domain_classifier_e4(e4_grl)

            results['real'] = {
                'domain_feat': domain_feat,
                'domain_logits_e3': domain_logits_e3,
                'domain_logits_e4': domain_logits_e4,
            }

        return results


class Trainer:
    def __init__(self, model, device='cuda', lr=0.001, weights=None):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCEWithLogitsLoss()

        # 使用传入的权重或默认值
        if weights is None:
            self.weights = {
                'class': 1.0,
                'mmd': 0.75,
                'domain': 0.01
            }
        else:
            self.weights = weights

    def train_step(self, sim_data, sim_labels, real_data):
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()

        sim_data = sim_data.to(self.device)
        sim_labels = sim_labels.to(self.device)
        real_data = real_data.to(self.device)

        results = self.model(sim_data=sim_data, real_data=real_data)

        losses = {}

        # 1. 分类损失
        class_loss = self.class_criterion(results['sim']['class_logits'], sim_labels)
        losses['class'] = class_loss.item()

        # 2. MMD域自适应损失
        mmd_loss = self.model.compute_mmd(
            results['sim']['domain_feat'],
            results['real']['domain_feat']
        )
        losses['mmd'] = mmd_loss.item()

        # 3. 逐像素域对抗损失
        B_sim = sim_data.size(0)
        B_real = real_data.size(0)

        source_labels_e3 = torch.zeros(B_sim, 1, 14, 14).to(self.device)
        target_labels_e3 = torch.ones(B_real, 1, 14, 14).to(self.device)

        source_labels_e4 = torch.zeros(B_sim, 1, 7, 7).to(self.device)
        target_labels_e4 = torch.ones(B_real, 1, 7, 7).to(self.device)

        domain_loss_e3_source = self.domain_criterion(
            results['sim']['domain_logits_e3'], source_labels_e3
        )
        domain_loss_e3_target = self.domain_criterion(
            results['real']['domain_logits_e3'], target_labels_e3
        )
        domain_loss_e3 = (domain_loss_e3_source + domain_loss_e3_target) / 2

        domain_loss_e4_source = self.domain_criterion(
            results['sim']['domain_logits_e4'], source_labels_e4
        )
        domain_loss_e4_target = self.domain_criterion(
            results['real']['domain_logits_e4'], target_labels_e4
        )
        domain_loss_e4 = (domain_loss_e4_source + domain_loss_e4_target) / 2

        domain_loss = (domain_loss_e3 + domain_loss_e4) / 2
        losses['domain'] = domain_loss.item()

        # 总损失
        total_loss = (self.weights['class'] * class_loss +
                      self.weights['mmd'] * mmd_loss +
                      self.weights['domain'] * domain_loss)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        losses['total'] = total_loss.item()
        return losses


def create_real_loader_no_repeat(real_dataset, target_size, batch_size):
    """创建不重复的真实数据加载器，每个epoch重新采样"""
    original_size = len(real_dataset)

    class EpochAwareRealDataset(torch.utils.data.IterableDataset):
        def __init__(self, dataset, target_size, original_size):
            self.dataset = dataset
            self.target_size = target_size
            self.original_size = original_size

        def generate_indices(self):
            repeats = self.target_size // self.original_size
            remainder = self.target_size % self.original_size

            indices = []
            for _ in range(repeats):
                perm = np.random.permutation(self.original_size).tolist()
                indices.extend(perm)

            if remainder > 0:
                remaining = np.random.choice(self.original_size, remainder, replace=False).tolist()
                indices.extend(remaining)

            return indices

        def __iter__(self):
            current_indices = self.generate_indices()
            for idx in current_indices:
                yield self.dataset[idx]

        def __len__(self):
            return self.target_size

    dataset = EpochAwareRealDataset(real_dataset, target_size, original_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_single_run(config, run_id):
    """
    单次训练运行

    Args:
        config: 配置字典
        run_id: 运行编号

    Returns:
        best_accuracy: 最佳准确率
        final_accuracy: 最终准确率
    """
    # 从配置中读取参数
    real_data_path = config['data']['real_data_path']
    real_data_path_test = config['data']['real_data_path_test']
    sim_data_path = config['data']['sim_data_path']
    save_dir = config['data']['save_dir']
    result_dir = config['data']['result_dir']

    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    lr = config['training']['lr']
    test_interval = config['training']['test_interval']
    dropout_rate = config['training']['dropout_rate']

    num_classes = config['model']['num_classes']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 读取损失权重配置（如果存在）
    weights = {
        'class': config['weights']['class_weight'],
        'mmd': config['weights']['mmd_weight'],
        'domain': config['weights']['domain_weight']
    }


    # 为每次运行创建独立的子目录
    run_save_dir = os.path.join(save_dir, f'run_{run_id}')
    run_result_dir = os.path.join(result_dir, f'run_{run_id}')
    os.makedirs(run_save_dir, exist_ok=True)
    os.makedirs(run_result_dir, exist_ok=True)

    # 类别名称
    class_names = ['2S1', 'BMP2', 'BTR70', 'M1', 'M2', 'M35', 'M548', 'M60', 'T72', 'ZSU23']

    # 数据预处理
    img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),  # p=1.0 即100%概率
        transforms.RandomRotation(degrees=(-90, -90)),  # 固定旋转-90°
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"\n{'=' * 60}")
    print(f"运行 {run_id}: 开始训练")
    print(f"设备: {device}")
    print(f"损失权重: class={weights['class']}, mmd={weights['mmd']}, domain={weights['domain']}")
    print(f"{'=' * 60}")

    # 加载数据集
    full_sim_dataset = datasets.ImageFolder(sim_data_path, transform=img_transform)
    sim_loader = DataLoader(
        full_sim_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    real_dataset = datasets.ImageFolder(real_data_path, transform=img_transform)
    real_loader = create_real_loader_no_repeat(real_dataset, len(full_sim_dataset), batch_size)

    test_dataset = datasets.ImageFolder(real_data_path_test, transform=img_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"仿真训练数据: {len(full_sim_dataset)} 张")
    print(f"真实数据: {len(real_dataset)} 张")
    print(f"测试数据: {len(test_dataset)} 张")

    # 创建模型和训练器
    model = SARDomainAdaptationResNet(
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )

    trainer = Trainer(model, device=device, lr=lr, weights=weights)

    # 训练循环
    best_accuracy = 0.0
    final_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch_idx, (sim_batch, real_batch) in enumerate(zip(sim_loader, real_loader)):
            sim_data, sim_label = sim_batch
            real_data, _ = real_batch

            losses = trainer.train_step(
                sim_data=sim_data,
                sim_labels=sim_label,
                real_data=real_data,
            )
            epoch_losses.append(losses)

        # 计算平均损失
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in epoch_losses])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"运行{run_id} - Epoch {epoch + 1}/{epochs} - ", end='')
            print(f"class: {avg_losses['class']:.4f} ", end='')
            print(f"mmd: {avg_losses['mmd']:.4f} ", end='')
            print(f"domain: {avg_losses['domain']:.4f} ", end='')
            print(f"total: {avg_losses['total']:.4f}")

        trainer.scheduler.step()

        # 测试
        if (epoch + 1) % test_interval == 0 or epoch == epochs - 1:
            model.eval()
            correct = 0
            total = 0
            all_true_labels = []
            all_pred_labels = []

            with torch.no_grad():
                for test_data, test_labels in test_loader:
                    test_data = test_data.to(device)
                    test_labels = test_labels.to(device)

                    results = model(sim_data=test_data)
                    predictions = results['sim']['class_logits']
                    _, predicted = torch.max(predictions, 1)

                    all_pred_labels.append(predicted.cpu())
                    all_true_labels.append(test_labels.cpu())
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()

            accuracy = 100 * correct / total if total > 0 else 0
            print(f"运行{run_id} - Epoch {epoch + 1}: 准确率 = {accuracy:.2f}%")

            # 更新最佳准确率
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'accuracy': accuracy,
                }, os.path.join(run_save_dir, 'best_model.pth'))

            # 更新最终准确率
            final_accuracy = accuracy

            # 在最后一个epoch保存混淆矩阵
            if epoch == epochs - 1:
                all_true_labels = torch.cat(all_true_labels)
                all_pred_labels = torch.cat(all_pred_labels)
                cm_save_path = os.path.join(run_result_dir, f'confusion_matrix_final.png')
                plot_confusion_matrix(
                    true_labels=all_true_labels,
                    pred_labels=all_pred_labels,
                    class_names=class_names,
                    save_path=cm_save_path
                )

    print(f"运行 {run_id} 完成!")
    print(f"  最佳准确率: {best_accuracy:.2f}%")
    print(f"  最终准确率: {final_accuracy:.2f}%")

    return best_accuracy, final_accuracy


def main(config):
    """主函数：执行多次重复实验并统计结果"""

    # 获取重复次数
    num_repeats = config.get('training', {}).get('num_repeats', 20)

    result_dir = config['data']['result_dir']
    os.makedirs(result_dir, exist_ok=True)

    # 创建结果文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f'experiment_results_{timestamp}.txt')

    # 读取权重配置

    weights = config['weights']
    weights_str = f"  class_weight: {weights['class_weight']}\n" \
                  f"  mmd_weight: {weights['mmd_weight']}\n" \
                  f"  domain_weight: {weights['domain_weight']}"


    # 写入实验信息
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SAR域自适应分类实验 - 多次重复实验统计结果\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 80 + "\n")
        f.write(f"配置信息:\n")
        f.write(f"  仿真数据路径: {config['data']['sim_data_path']}\n")
        f.write(f"  真实训练数据: {config['data']['real_data_path']}\n")
        f.write(f"  真实测试数据: {config['data']['real_data_path_test']}\n")
        f.write(f"  批次大小: {config['training']['batch_size']}\n")
        f.write(f"  训练轮数: {config['training']['epochs']}\n")
        f.write(f"  学习率: {config['training']['lr']}\n")
        f.write(f"  Dropout率: {config['training']['dropout_rate']}\n")
        f.write(f"  重复次数: {num_repeats}\n")
        f.write(f"  损失权重:\n{weights_str}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'运行编号':<10} {'最佳准确率(%)':<15} {'最终准确率(%)':<15}\n")
        f.write("-" * 40 + "\n")

    # 存储结果
    best_accuracies = []
    final_accuracies = []

    # 执行多次重复实验
    for run_id in range(1, num_repeats + 1):
        print(f"\n{'#' * 60}")
        print(f"开始第 {run_id}/{num_repeats} 次实验")
        print(f"{'#' * 60}")

        # 单次训练
        best_acc, final_acc = train_single_run(config, run_id)

        best_accuracies.append(best_acc)
        final_accuracies.append(final_acc)

        # 写入当前结果
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(f"{run_id:<10} {best_acc:<15.4f} {final_acc:<15.4f}\n")

        # 实时显示统计信息
        if len(best_accuracies) > 1:
            print(f"\n当前统计 ({len(best_accuracies)}/{num_repeats} 次实验):")
            print(f"  最佳准确率 - Min: {np.min(best_accuracies):.4f}%, Max: {np.max(best_accuracies):.4f}%, "
                  f"Mean: {np.mean(best_accuracies):.4f}%, Std: {np.std(best_accuracies):.4f}%")
            print(f"  最终准确率 - Min: {np.min(final_accuracies):.4f}%, Max: {np.max(final_accuracies):.4f}%, "
                  f"Mean: {np.mean(final_accuracies):.4f}%, Std: {np.std(final_accuracies):.4f}%")

    # 最终统计结果
    print(f"\n{'=' * 80}")
    print("所有实验完成！最终统计结果:")
    print(f"{'=' * 80}")

    best_stats = {
        'min': np.min(best_accuracies),
        'max': np.max(best_accuracies),
        'mean': np.mean(best_accuracies),
        'std': np.std(best_accuracies)
    }

    final_stats = {
        'min': np.min(final_accuracies),
        'max': np.max(final_accuracies),
        'mean': np.mean(final_accuracies),
        'std': np.std(final_accuracies)
    }

    print(f"最佳准确率统计:")
    print(f"  Min: {best_stats['min']:.4f}%")
    print(f"  Max: {best_stats['max']:.4f}%")
    print(f"  Mean: {best_stats['mean']:.4f}%")
    print(f"  Std: {best_stats['std']:.4f}%")

    print(f"\n最终准确率统计:")
    print(f"  Min: {final_stats['min']:.4f}%")
    print(f"  Max: {final_stats['max']:.4f}%")
    print(f"  Mean: {final_stats['mean']:.4f}%")
    print(f"  Std: {final_stats['std']:.4f}%")

    # 写入最终统计结果
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("最终统计结果:\n")
        f.write("-" * 40 + "\n")
        f.write(f"最佳准确率:\n")
        f.write(f"  Min: {best_stats['min']:.4f}%\n")
        f.write(f"  Max: {best_stats['max']:.4f}%\n")
        f.write(f"  Mean: {best_stats['mean']:.4f}%\n")
        f.write(f"  Std: {best_stats['std']:.4f}%\n\n")
        f.write(f"最终准确率:\n")
        f.write(f"  Min: {final_stats['min']:.4f}%\n")
        f.write(f"  Max: {final_stats['max']:.4f}%\n")
        f.write(f"  Mean: {final_stats['mean']:.4f}%\n")
        f.write(f"  Std: {final_stats['std']:.4f}%\n")
        f.write("=" * 80 + "\n")
        f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\n结果已保存至: {result_file}")

    return best_stats, final_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAR Domain Adaptation Training with Multiple Runs')

    parser.add_argument('--config', type=str, default=r'secen1.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)


