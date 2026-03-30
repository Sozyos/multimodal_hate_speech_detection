#!/usr/bin/env python3
"""
多模态恶意言论检测模型评估脚本。
支持加载训练好的模型，在测试集上进行全面评估，生成指标报告和可视化图表。
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.dataset import M3Dataset
from models.multimodal_fusion import MultimodalFusionModel
from training.trainer import MultimodalTrainer

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def create_test_loader(config: Dict[str, Any]) -> DataLoader:
    """创建测试数据加载器"""
    data_config = config['data']

    # 创建测试数据集（禁用虚拟数据）
    test_dataset = M3Dataset(
        data_root=data_config['data_root'],
        split='test',
        image_size=tuple(data_config['image_size']),
        max_text_length=data_config['max_text_length'],
        use_dummy=False  # 强制使用真实数据
    )

    if len(test_dataset) == 0:
        raise ValueError("测试数据集为空！请检查数据路径和数据文件。")

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 2),
        pin_memory=True
    )

    print(f"测试集大小: {len(test_dataset)} 个样本")
    return test_loader

def load_model(checkpoint_path: str, config: Dict[str, Any]) -> torch.nn.Module:
    """加载模型和检查点"""
    model_config = config['model']

    # 创建模型结构
    model = MultimodalFusionModel(
        image_model_name=model_config['image_model_name'],
        text_model_name=model_config['text_model_name'],
        fusion_dim=model_config['fusion_dim'],
        num_binary_classes=model_config['num_binary_classes'],
        num_multilabel_classes=model_config['num_multilabel_classes'],
        dropout_rate=model_config.get('dropout_rate', 0.1)
    )

    # 加载检查点
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 如果检查点直接保存了模型状态
        model.load_state_dict(checkpoint)

    print(f"模型加载成功: {checkpoint_path}")

    # 如果有其他信息，显示出来
    if 'epoch' in checkpoint:
        print(f"训练轮数: {checkpoint['epoch']}")
    if 'best_val_score' in checkpoint:
        print(f"最佳验证分数: {checkpoint['best_val_score']:.4f}")

    return model

def save_evaluation_results(results: Dict[str, Any], output_dir: Path):
    """保存评估结果到文件"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存JSON格式的详细结果
    json_path = output_dir / 'evaluation_results.json'

    # 转换numpy数组为列表以便JSON序列化
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    json_results = convert_for_json(results)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"详细结果保存到: {json_path}")

    # 保存CSV格式的指标
    if 'metrics' in results:
        metrics_df = pd.DataFrame([results['metrics']])
        csv_path = output_dir / 'metrics.csv'
        metrics_df.to_csv(csv_path, index=False)
        print(f"指标表格保存到: {csv_path}")

    # 保存文本报告
    report_path = output_dir / 'evaluation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("多模态恶意言论检测模型评估报告\n")
        f.write("=" * 50 + "\n\n")

        # 基本信息
        f.write("1. 基本信息\n")
        f.write(f"   评估时间: {pd.Timestamp.now()}\n")
        f.write(f"   测试样本数: {results.get('num_samples', 'N/A')}\n\n")

        # 指标概览
        f.write("2. 主要指标\n")
        if 'metrics' in results:
            metrics = results['metrics']
            for key, value in sorted(metrics.items()):
                if isinstance(value, (int, float)):
                    f.write(f"   {key}: {value:.4f}\n")
        f.write("\n")

        # 详细报告
        if 'detailed_reports' in results:
            detailed = results['detailed_reports']
            f.write("3. 详细报告\n")
            if 'binary_report' in detailed:
                f.write("二元分类报告:\n")
                f.write(detailed['binary_report'])
                f.write("\n")
            if 'class_distribution' in detailed:
                f.write("类别分布:\n")
                f.write(str(detailed['class_distribution']))
                f.write("\n")

    print(f"评估报告保存到: {report_path}")

def generate_visualizations(results: Dict[str, Any], output_dir: Path):
    """生成可视化图表"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置matplotlib样式
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    charts_generated = 0

    # 1. 二元分类混淆矩阵热图
    if 'binary_confusion_matrix' in results:
        cm = results['binary_confusion_matrix']
        if isinstance(cm, (np.ndarray, list)) and len(cm) == 4:
            # 转换为2x2矩阵
            cm_matrix = np.array([[cm[0], cm[1]], [cm[2], cm[3]]])

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Normal', 'Hate'],
                       yticklabels=['Normal', 'Hate'])
            plt.title('二元分类混淆矩阵')
            plt.ylabel('真实标签')
            plt.xlabel('预测标签')
            plt.tight_layout()
            plt.savefig(output_dir / 'binary_confusion_matrix.png', dpi=300)
            plt.close()
            charts_generated += 1

    # 2. 二元分类ROC曲线
    if 'binary_probabilities' in results and 'labels' in results:
        try:
            # 获取二元标签和概率
            binary_probs = results['binary_probabilities']
            # 从labels中提取二元标签
            binary_labels_list = []
            for batch in results['labels']:
                if 'binary' in batch:
                    binary_labels_list.append(batch['binary'])

            if binary_labels_list and len(binary_probs) > 0:
                binary_labels = torch.cat(binary_labels_list).numpy()
                binary_probs = np.array(binary_probs)

                # 计算ROC曲线
                fpr, tpr, thresholds = roc_curve(binary_labels, binary_probs)
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC曲线 (AUC = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('假正率')
                plt.ylabel('真正率')
                plt.title('二元分类ROC曲线')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / 'binary_roc_curve.png', dpi=300)
                plt.close()
                charts_generated += 1

                # 保存ROC数据
                roc_data = pd.DataFrame({
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': np.append(thresholds, np.nan)  # 阈值比fpr/tpr少一个
                })
                roc_data.to_csv(output_dir / 'roc_curve_data.csv', index=False)
        except Exception as e:
            print(f"生成ROC曲线时出错: {e}")

    # 3. 二元分类PR曲线
    if 'binary_probabilities' in results and 'labels' in results:
        try:
            binary_probs = results['binary_probabilities']
            binary_labels_list = []
            for batch in results['labels']:
                if 'binary' in batch:
                    binary_labels_list.append(batch['binary'])

            if binary_labels_list and len(binary_probs) > 0:
                binary_labels = torch.cat(binary_labels_list).numpy()
                binary_probs = np.array(binary_probs)

                # 计算PR曲线
                precision, recall, thresholds = precision_recall_curve(binary_labels, binary_probs)
                avg_precision = average_precision_score(binary_labels, binary_probs)

                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, color='green', lw=2,
                        label=f'PR曲线 (AP = {avg_precision:.3f})')
                plt.xlabel('召回率')
                plt.ylabel('精确率')
                plt.title('二元分类精确率-召回率曲线')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.legend(loc="upper right")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / 'binary_pr_curve.png', dpi=300)
                plt.close()
                charts_generated += 1
        except Exception as e:
            print(f"生成PR曲线时出错: {e}")

    # 4. 多标签类别分布热图
    if 'multilabel_probabilities' in results and 'labels' in results:
        try:
            multilabel_probs = results['multilabel_probabilities']
            if len(multilabel_probs) > 0:
                # 计算每个类别的平均概率
                mean_probs = np.mean(multilabel_probs, axis=0)

                plt.figure(figsize=(10, 6))
                classes = [f'Class {i}' for i in range(len(mean_probs))]
                bars = plt.bar(range(len(mean_probs)), mean_probs)
                plt.xticks(range(len(mean_probs)), classes, rotation=45, ha='right')
                plt.title('多标签分类 - 每个类别的平均预测概率')
                plt.ylabel('平均概率')
                plt.ylim(0, 1.0)

                # 添加数值标签
                for bar, prob in zip(bars, mean_probs):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{prob:.3f}', ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig(output_dir / 'multilabel_class_probabilities.png', dpi=300)
                plt.close()
                charts_generated += 1
        except Exception as e:
            print(f"生成多标签类别分布图时出错: {e}")

    # 5. 指标柱状图
    if 'metrics' in results:
        metrics = results['metrics']
        # 筛选主要指标
        key_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and any(x in key for x in ['f1', 'accuracy', 'precision', 'recall', 'auc']):
                # 简化键名
                display_key = key.replace('binary_', '').replace('multilabel_', '')
                key_metrics[display_key] = value

        if key_metrics:
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(key_metrics)), list(key_metrics.values()))
            plt.xticks(range(len(key_metrics)), list(key_metrics.keys()), rotation=45, ha='right')
            plt.title('主要评估指标')
            plt.ylabel('分数')
            plt.ylim(0, 1.0)

            # 在柱子上方添加数值标签
            for bar, value in zip(bars, key_metrics.values()):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(output_dir / 'key_metrics.png', dpi=300)
            plt.close()
            charts_generated += 1

    # 6. 错误分析样本表格
    if 'samples' in results and len(results['samples']) > 0:
        try:
            # 提取错误分类的样本（二元分类错误）
            error_samples = []
            for sample in results['samples']:
                if 'binary_label' in sample and 'binary_pred' in sample:
                    if sample['binary_label'] != sample['binary_pred']:
                        error_samples.append(sample)

            if error_samples:
                # 创建错误分析表格
                error_df = pd.DataFrame(error_samples)
                # 只保留关键列
                keep_cols = ['ocr_text', 'post_text', 'binary_label', 'binary_pred', 'binary_prob']
                available_cols = [col for col in keep_cols if col in error_df.columns]
                error_df = error_df[available_cols]

                # 保存为CSV
                error_csv_path = output_dir / 'error_analysis.csv'
                error_df.to_csv(error_csv_path, index=False, encoding='utf-8-sig')
                print(f"错误分析表格保存到: {error_csv_path}")

                # 保存前10个错误样本为文本
                error_txt_path = output_dir / 'error_samples.txt'
                with open(error_txt_path, 'w', encoding='utf-8') as f:
                    f.write("错误分类样本分析（前10个）\n")
                    f.write("=" * 60 + "\n\n")

                    for i, sample in enumerate(error_samples[:10]):
                        f.write(f"样本 {i+1}:\n")
                        f.write(f"  OCR文本: {sample.get('ocr_text', 'N/A')}\n")
                        f.write(f"  帖子文本: {sample.get('post_text', 'N/A')}\n")
                        f.write(f"  真实标签: {sample.get('binary_label', 'N/A')}\n")
                        f.write(f"  预测标签: {sample.get('binary_pred', 'N/A')}\n")
                        if 'binary_prob' in sample:
                            f.write(f"  预测概率: {sample.get('binary_prob', 'N/A'):.4f}\n")
                        f.write("\n")

                charts_generated += 1
        except Exception as e:
            print(f"生成错误分析时出错: {e}")

    if charts_generated == 0:
        print("警告: 未生成任何可视化图表，请检查评估结果数据")
    else:
        print(f"已生成 {charts_generated} 个可视化图表到: {output_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估多模态恶意言论检测模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径 (如: checkpoints/best_model.pt)')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='评估结果输出目录 (默认使用配置中的output_dir)')
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (cuda, cpu, mps)，默认使用配置中的设置')
    parser.add_argument('--visualize', action='store_true', default=None,
                       help='是否生成可视化图表')

    args = parser.parse_args()

    print("=" * 60)
    print("多模态恶意言论检测模型评估")
    print("=" * 60)
    print(f"检查点: {args.checkpoint}")
    print(f"配置: {args.config}")

    # 加载配置
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return

    # 覆盖设备设置
    if args.device:
        config['training']['device'] = args.device

    # 覆盖可视化设置
    if args.visualize is not None:
        config['evaluation']['visualize'] = args.visualize

    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config['evaluation'].get('output_dir', 'evaluation_results'))

    # 创建测试数据加载器
    try:
        print("\n1. 加载测试数据...")
        test_loader = create_test_loader(config)
    except Exception as e:
        print(f"加载测试数据失败: {e}")
        return

    # 加载模型
    try:
        print("\n2. 加载模型...")
        model = load_model(args.checkpoint, config)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 创建训练器用于评估
    print("\n3. 创建训练器...")
    # 创建虚拟的训练和验证加载器（仅用于初始化训练器）
    dummy_loader = test_loader  # 复用测试加载器

    # 提取训练配置
    train_config = config['training']

    # 创建训练器实例
    trainer = MultimodalTrainer(
        model=model,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        config=train_config
    )

    # 设置设备
    device = torch.device(train_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    trainer.device = device
    trainer.model.to(device)

    # 执行评估
    print("\n4. 开始评估...")
    try:
        results = trainer.evaluate_on_test(test_loader)
        results['num_samples'] = len(test_loader.dataset)

        # 添加混淆矩阵信息
        if 'metrics' in results and 'binary_true_negative' in results['metrics']:
            results['binary_confusion_matrix'] = [
                results['metrics']['binary_true_negative'],
                results['metrics']['binary_false_positive'],
                results['metrics']['binary_false_negative'],
                results['metrics']['binary_true_positive']
            ]

    except Exception as e:
        print(f"评估失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 打印主要结果
    print("\n5. 评估结果概览:")
    print("-" * 40)
    if 'metrics' in results:
        metrics = results['metrics']
        # 打印主要指标
        main_metrics = [
            ('binary_accuracy', '二元准确率'),
            ('binary_f1', '二元F1分数'),
            ('binary_precision', '二元精确率'),
            ('binary_recall', '二元召回率'),
            ('multilabel_f1_macro', '多标签F1宏平均'),
            ('multilabel_hamming_loss', '多标签汉明损失'),
            ('overall_score', '综合分数')
        ]

        for key, desc in main_metrics:
            if key in metrics:
                print(f"  {desc}: {metrics[key]:.4f}")

    print(f"\n6. 保存结果到: {output_dir}")

    # 保存结果
    save_evaluation_results(results, output_dir)

    # 生成可视化
    if config['evaluation'].get('visualize', True):
        print("\n7. 生成可视化图表...")
        generate_visualizations(results, output_dir)

    print("\n" + "=" * 60)
    print("评估完成！")
    print(f"详细结果请查看: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
