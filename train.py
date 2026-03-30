#!/usr/bin/env python3
"""
多模态恶意言论检测模型训练主脚本。
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.dataset import M3Dataset
from models.multimodal_fusion import MultimodalFusionModel
from training.trainer import MultimodalTrainer

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def create_data_loaders(config: dict) -> tuple:
    """创建数据加载器"""
    data_config = config['data']

    # 创建数据集
    train_dataset = M3Dataset(
        data_root=data_config['data_root'],
        split='train',
        image_size=tuple(data_config['image_size']),
        max_text_length=data_config['max_text_length'],
        use_dummy=data_config.get('use_dummy', False)
    )

    val_dataset = M3Dataset(
        data_root=data_config['data_root'],
        split='val',
        image_size=tuple(data_config['image_size']),
        max_text_length=data_config['max_text_length'],
        use_dummy=data_config.get('use_dummy', False)
    )

    test_dataset = M3Dataset(
        data_root=data_config['data_root'],
        split='test',
        image_size=tuple(data_config['image_size']),
        max_text_length=data_config['max_text_length'],
        use_dummy=data_config.get('use_dummy', False)
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 2),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 2),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 2),
        pin_memory=True
    )

    print(f"数据集大小: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}")

    return train_loader, val_loader, test_loader

def create_model(config: dict) -> torch.nn.Module:
    """创建模型"""
    model_config = config['model']

    model = MultimodalFusionModel(
        image_model_name=model_config['image_model_name'],
        text_model_name=model_config['text_model_name'],
        fusion_dim=model_config['fusion_dim'],
        num_binary_classes=model_config['num_binary_classes'],
        num_multilabel_classes=model_config['num_multilabel_classes'],
        dropout_rate=model_config.get('dropout_rate', 0.1)
    )

    return model

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练多模态恶意言论检测模型')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅进行评估，不训练')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda, cpu, mps)')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 覆盖设备设置
    if args.device:
        config['training']['device'] = args.device

    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(config)

    # 创建模型
    print("创建模型...")
    model = create_model(config)

    # 如果有检查点，加载
    if args.resume:
        print(f"从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    # 仅评估模式
    if args.eval_only:
        print("仅进行评估...")
        trainer = MultimodalTrainer(model, train_loader, val_loader, config['training'])
        trainer.device = torch.device(config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        model.to(trainer.device)

        # 在测试集上评估
        test_results = trainer.evaluate_on_test(test_loader)
        print("\n测试结果:")
        for key, value in test_results['metrics'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")

        return

    # 创建训练器
    print("创建训练器...")
    trainer = MultimodalTrainer(model, train_loader, val_loader, config['training'])

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    print("开始训练...")
    trainer.train()

    # 训练完成后在测试集上评估
    print("\n在测试集上评估最终模型...")
    test_results = trainer.evaluate_on_test(test_loader)

    print("\n最终测试结果:")
    for key, value in test_results['metrics'].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")

    # 保存最终模型
    final_model_path = trainer.save_dir / 'final_model.pt'
    trainer.save_checkpoint('final_model.pt', trainer.current_epoch, test_results['metrics'])

    # 保存测试结果
    results_path = trainer.save_dir / 'test_results.yaml'
    with open(results_path, 'w') as f:
        yaml.dump({
            'metrics': test_results['metrics'],
            'config': config
        }, f, default_flow_style=False)

    print(f"\n训练完成！")
    print(f"最终模型保存到: {final_model_path}")
    print(f"测试结果保存到: {results_path}")

if __name__ == "__main__":
    main()
