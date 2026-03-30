#!/usr/bin/env python3
"""
集成测试：检查数据加载、模型前向传播等基本功能。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_data_loading():
    """测试数据加载"""
    print("测试数据加载...")
    try:
        from data.dataset import M3Dataset
        from data.preprocessing import ImagePreprocessor, TextPreprocessor, LabelEncoder
        
        # 测试预处理模块
        img_preprocessor = ImagePreprocessor()
        # 使用简单tokenizer以避免网络连接问题
        text_preprocessor = TextPreprocessor(use_simple_tokenizer=True)
        label_encoder = LabelEncoder()
        
        print("  预处理模块导入成功")
        
        # 测试数据集（使用虚拟数据）
        dataset = M3Dataset(data_root=project_root, split='train', use_dummy=True)
        print(f"  数据集创建成功，大小: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  样本键: {list(sample.keys())}")
            print(f"  图像形状: {sample['image'].shape}")
            print(f"  文本输入形状: {sample['input_ids'].shape}")
            print("  数据加载测试通过")
            return True
        else:
            print("  警告: 数据集为空")
            return False
            
    except Exception as e:
        print(f"  数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model():
    """测试模型"""
    print("测试模型...")
    try:
        from models.multimodal_fusion import MultimodalFusionModel
        
        model = MultimodalFusionModel()
        print(f"  模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 创建虚拟输入
        import torch
        batch_size = 2
        image_tensor = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 10000, (batch_size, 128))
        attention_mask = torch.ones(batch_size, 128)
        
        # 前向传播
        outputs = model(image_tensor, input_ids, attention_mask)
        print(f"  前向传播成功")
        print(f"  二元分类logits形状: {outputs['binary_logits'].shape}")
        print(f"  多标签分类logits形状: {outputs['multilabel_logits'].shape}")
        print("  模型测试通过")
        return True
        
    except Exception as e:
        print(f"  模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_components():
    """测试训练组件"""
    print("测试训练组件...")
    try:
        from training.losses import MultitaskLoss
        from training.metrics import MultimodalMetrics
        
        # 测试损失函数
        criterion = MultitaskLoss()
        print("  损失函数导入成功")
        
        # 测试评估指标
        metrics = MultimodalMetrics(num_multilabel_classes=8)
        print("  评估指标导入成功")
        
        # 测试训练器（仅导入）
        try:
            from training.trainer import MultimodalTrainer
            print("  训练器导入成功")
        except Exception as e:
            print(f"  训练器导入警告: {e}")
            print("  训练器导入跳过（可能缺少wandb或其他依赖）")
        
        print("  训练组件测试通过")
        return True
        
    except Exception as e:
        print(f"  训练组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始集成测试...")
    print("=" * 50)
    
    results = []
    
    # 测试数据加载
    results.append(("数据加载", test_data_loading()))
    
    # 测试模型
    results.append(("模型", test_model()))
    
    # 测试训练组件
    results.append(("训练组件", test_training_components()))
    
    print("=" * 50)
    print("测试结果:")
    
    all_passed = True
    for name, passed in results:
        status = "通过" if passed else "失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n所有测试通过！基本功能正常。")
        print("下一步: 运行实际训练 (python train.py --config configs/base_config.yaml)")
    else:
        print("\n部分测试失败，请检查错误信息。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
