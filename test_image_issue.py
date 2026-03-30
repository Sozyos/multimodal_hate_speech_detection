#!/usr/bin/env python3
"""
测试图像生成问题，检查"Unsupported Image"错误。
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_numpy_to_image():
    """测试numpy数组转换为PIL图像"""
    print("测试numpy数组转换为PIL图像...")

    # 创建随机RGB图像数据
    try:
        # 方法1: 使用0-255的uint8
        arr1 = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        img1 = Image.fromarray(arr1, 'RGB')
        print(f"  方法1成功: 数组形状={arr1.shape}, 数据类型={arr1.dtype}")
        print(f"  图像模式={img1.mode}, 尺寸={img1.size}")

        # 方法2: 使用0-1的float
        arr2 = np.random.rand(224, 224, 3)
        img2 = Image.fromarray((arr2 * 255).astype(np.uint8), 'RGB')
        print(f"  方法2成功: 数组形状={arr2.shape}, 数据类型={arr2.dtype}")

        # 方法3: 检查不同形状
        arr3 = np.random.rand(3, 224, 224) * 255  # CHW格式
        arr3_uint8 = arr3.astype(np.uint8)
        # 需要转置为HWC
        arr3_hwc = np.transpose(arr3_uint8, (1, 2, 0))
        img3 = Image.fromarray(arr3_hwc, 'RGB')
        print(f"  方法3成功: CHW转HWC后形状={arr3_hwc.shape}")

        return True
    except Exception as e:
        print(f"  numpy转图像失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_image_generation():
    """测试数据集中的图像生成"""
    print("\n测试数据集中的图像生成...")

    try:
        from data.dataset import M3Dataset

        # 创建虚拟数据集
        dataset = M3Dataset(
            data_root=project_root,
            split='train',
            use_dummy=True
        )

        print(f"  数据集创建成功，大小: {len(dataset)}")

        if len(dataset) > 0:
            # 获取第一个样本
            sample = dataset[0]
            print(f"  样本图像张量形状: {sample['image'].shape}")
            print(f"  图像张量数据类型: {sample['image'].dtype}")
            print(f"  图像张量值范围: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")

            # 检查是否归一化到合理范围
            if -3.0 < sample['image'].min() < 3.0 and -3.0 < sample['image'].max() < 3.0:
                print("  图像已正确归一化")
            else:
                print(f"  警告: 图像值范围异常")

            return True
        else:
            print("  警告: 数据集为空")
            return False

    except Exception as e:
        print(f"  数据集图像生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pil_version():
    """测试PIL/Pillow版本"""
    print("\n测试PIL/Pillow版本...")

    try:
        from PIL import Image
        import PIL

        print(f"  PIL版本: {PIL.__version__}")
        print(f"  Image模块: {Image.__file__}")

        # 测试支持的格式
        formats = Image.registered_extensions()
        print(f"  支持的图像格式数量: {len(formats)}")

        # 检查常见格式
        common_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        supported = [fmt for fmt in common_formats if fmt in formats]
        print(f"  常见格式支持: {supported}")

        return True
    except Exception as e:
        print(f"  PIL版本检查失败: {e}")
        return False

def test_opencv():
    """测试OpenCV（如果已安装）"""
    print("\n测试OpenCV...")

    try:
        import cv2
        print(f"  OpenCV版本: {cv2.__version__}")

        # 检查是否可以读取/写入图像
        test_array = np.random.rand(100, 100, 3) * 255
        test_array = test_array.astype(np.uint8)

        # 测试编码
        success, encoded = cv2.imencode('.jpg', test_array)
        if success:
            print("  OpenCV JPEG编码成功")
        else:
            print("  OpenCV JPEG编码失败")

        return True
    except ImportError:
        print("  OpenCV未安装")
        return False
    except Exception as e:
        print(f"  OpenCV测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("图像问题诊断测试")
    print("=" * 50)

    results = []

    results.append(("PIL版本检查", test_pil_version()))
    results.append(("numpy转图像测试", test_numpy_to_image()))
    results.append(("数据集图像生成", test_dataset_image_generation()))
    results.append(("OpenCV测试", test_opencv()))

    print("\n" + "=" * 50)
    print("测试结果:")

    all_passed = True
    for name, passed in results:
        status = "通过" if passed else "失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("所有图像相关测试通过！")
    else:
        print("部分测试失败，可能是'Unsupported Image'错误的原因")

    # 建议
    print("\n建议:")
    if all_passed:
        print("1. 基本图像处理功能正常")
        print("2. 如果仍有'Unsupported Image'错误，请检查具体错误堆栈")
        print("3. 确保有足够的磁盘空间和内存")
    else:
        print("1. 检查Pillow版本: pip install --upgrade Pillow")
        print("2. 检查numpy版本: pip install --upgrade numpy")
        print("3. 确保OpenCV正确安装: pip install opencv-python==4.5.5")
        print("4. 检查Python环境是否损坏")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)