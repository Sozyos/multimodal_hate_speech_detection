#!/usr/bin/env python3
"""
多模态融合模型：结合图像和文本特征进行恶意言论检测。
使用预训练的ViT和BERT编码器，通过注意力机制融合多模态特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, ViTConfig, BertConfig
from typing import Dict

class MultimodalFusionModel(nn.Module):
    """多模态融合模型"""

    def __init__(self,
                 image_model_name: str = "google/vit-base-patch16-224",
                 text_model_name: str = "bert-base-multilingual-cased",
                 fusion_dim: int = 768,
                 num_binary_classes: int = 2,
                 num_multilabel_classes: int = 8,
                 dropout_rate: float = 0.1):
        """
        初始化多模态融合模型

        Args:
            image_model_name: 预训练图像模型名称
            text_model_name: 预训练文本模型名称
            fusion_dim: 融合特征维度
            num_binary_classes: 二元分类类别数（恶意/正常）
            num_multilabel_classes: 多标签分类类别数（细粒度恶意类别）
            dropout_rate: Dropout率
        """
        super().__init__()

        # 图像编码器
        try:
            print(f"尝试加载预训练图像编码器: {image_model_name}")
            self.image_encoder = AutoModel.from_pretrained(image_model_name)
            print("预训练图像编码器加载成功")
            # 冻结部分层（可选）
            self._freeze_encoder(self.image_encoder, num_frozen_layers=6)
            self.image_encoder_pretrained = True
        except Exception as e:
            print(f"预训练图像编码器加载失败: {e}")
            print("使用随机初始化的图像编码器")
            # 使用默认的ViT配置
            config = ViTConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                image_size=224,
                patch_size=16,
                num_channels=3,
                qkv_bias=True,
                encoder_stride=16,
            )
            self.image_encoder = AutoModel.from_config(config)
            self.image_encoder_pretrained = False

        # 文本编码器
        try:
            print(f"尝试加载预训练文本编码器: {text_model_name}")
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            print("预训练文本编码器加载成功")
            self._freeze_encoder(self.text_encoder, num_frozen_layers=6)
            self.text_encoder_pretrained = True
        except Exception as e:
            print(f"预训练文本编码器加载失败: {e}")
            print("使用随机初始化的文本编码器")
            # 使用默认的BERT配置
            config = BertConfig(
                vocab_size=30522,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                pad_token_id=0,
                position_embedding_type="absolute",
                use_cache=True,
                classifier_dropout=None,
            )
            self.text_encoder = AutoModel.from_config(config)
            self.text_encoder_pretrained = False

        # 获取编码器输出维度
        # 使用默认维度（ViT和BERT通常为768）
        self.image_feature_dim = 768
        self.text_feature_dim = 768
        print(f"图像特征维度: {self.image_feature_dim}, 文本特征维度: {self.text_feature_dim}")

        # 特征投影层（将特征映射到统一维度）
        self.image_projection = nn.Linear(self.image_feature_dim, fusion_dim)
        self.text_projection = nn.Linear(self.text_feature_dim, fusion_dim)

        # 跨模态注意力融合
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )

        # 融合后的特征处理
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),  # 拼接了3个特征：fused_features, image_proj, text_proj
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # 分类头
        self.binary_classifier = nn.Sequential(
            nn.Linear(fusion_dim // 2, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_binary_classes)
        )

        self.multilabel_classifier = nn.Sequential(
            nn.Linear(fusion_dim // 2, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_multilabel_classes)
        )

        # 初始化权重
        self._init_weights()

    def _freeze_encoder(self, encoder, num_frozen_layers: int = 6):
        """冻结编码器的前几层"""
        if num_frozen_layers <= 0:
            return

        # 冻结嵌入层
        if hasattr(encoder, 'embeddings'):
            for param in encoder.embeddings.parameters():
                param.requires_grad = False

        # 冻结前num_frozen_layers个编码器层
        if hasattr(encoder, 'encoder'):
            layers = encoder.encoder.layer if hasattr(encoder.encoder, 'layer') else encoder.encoder.layers
            for i, layer in enumerate(layers[:num_frozen_layers]):
                for param in layer.parameters():
                    param.requires_grad = False

    def _init_weights(self):
        """初始化投影层和分类头的权重"""
        for module in [self.image_projection, self.text_projection,
                       self.binary_classifier, self.multilabel_classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 初始化MLP
        for layer in self.fusion_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, image_tensor: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            image_tensor: 图像张量 [batch_size, channels, height, width]
            input_ids: 文本token IDs [batch_size, seq_length]
            attention_mask: 注意力掩码 [batch_size, seq_length]

        Returns:
            包含分类logits的字典
        """
        batch_size = image_tensor.size(0)

        # 1. 提取图像特征
        image_features = self._extract_image_features(image_tensor)  # [batch_size, image_feature_dim]

        # 2. 提取文本特征
        text_features = self._extract_text_features(input_ids, attention_mask)  # [batch_size, text_feature_dim]

        # 3. 投影到统一维度
        image_proj = self.image_projection(image_features)  # [batch_size, fusion_dim]
        text_proj = self.text_projection(text_features)     # [batch_size, fusion_dim]

        # 4. 跨模态注意力融合
        # 将特征扩展为序列形式 [batch_size, 1, fusion_dim]
        image_seq = image_proj.unsqueeze(1)  # [batch_size, 1, fusion_dim]
        text_seq = text_proj.unsqueeze(1)    # [batch_size, 1, fusion_dim]

        # 拼接为多模态序列
        multimodal_seq = torch.cat([image_seq, text_seq], dim=1)  # [batch_size, 2, fusion_dim]

        # 自注意力融合
        attention_output, _ = self.cross_attention(
            multimodal_seq, multimodal_seq, multimodal_seq
        )  # [batch_size, 2, fusion_dim]

        # 池化：平均所有位置的特征
        fused_features = attention_output.mean(dim=1)  # [batch_size, fusion_dim]

        # 5. 拼接原始投影特征（残差连接）
        combined_features = torch.cat([fused_features, image_proj, text_proj], dim=1)  # [batch_size, fusion_dim * 3]

        # 6. MLP处理
        mlp_features = self.fusion_mlp(combined_features)  # [batch_size, fusion_dim // 2]

        # 7. 分类
        binary_logits = self.binary_classifier(mlp_features)  # [batch_size, num_binary_classes]
        multilabel_logits = self.multilabel_classifier(mlp_features)  # [batch_size, num_multilabel_classes]

        return {
            'binary_logits': binary_logits,
            'multilabel_logits': multilabel_logits,
            'image_features': image_features,
            'text_features': text_features,
            'fused_features': fused_features
        }

    def _extract_image_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """提取图像特征"""
        # ViT输出：last_hidden_state的形状为 [batch_size, seq_len, hidden_size]
        outputs = self.image_encoder(pixel_values=image_tensor)
        image_features = outputs.last_hidden_state

        # 使用CLS token的特征（索引0）
        cls_features = image_features[:, 0, :]  # [batch_size, hidden_size]
        return cls_features

    def _extract_text_features(self, input_ids: torch.Tensor,
                               attention_mask: torch.Tensor) -> torch.Tensor:
        """提取文本特征"""
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = outputs.last_hidden_state

        # 使用CLS token的特征（索引0）
        cls_features = text_features[:, 0, :]  # [batch_size, hidden_size]
        return cls_features

    def get_attention_weights(self, image_tensor: torch.Tensor,
                              input_ids: torch.Tensor,
                              attention_mask: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重（用于可视化）

        Returns:
            注意力权重 [batch_size, num_heads, 2, 2]
        """
        # 提取并投影特征
        image_features = self._extract_image_features(image_tensor)
        text_features = self._extract_text_features(input_ids, attention_mask)

        image_proj = self.image_projection(image_features).unsqueeze(1)
        text_proj = self.text_projection(text_features).unsqueeze(1)
        multimodal_seq = torch.cat([image_proj, text_proj], dim=1)

        # 获取注意力权重
        _, attention_weights = self.cross_attention(
            multimodal_seq, multimodal_seq, multimodal_seq,
            need_weights=True
        )

        return attention_weights

class SimpleMultimodalModel(nn.Module):
    """简化的多模态模型（用于快速原型）"""

    def __init__(self,
                 image_feature_dim: int = 512,
                 text_feature_dim: int = 768,
                 hidden_dim: int = 256,
                 num_binary_classes: int = 2,
                 num_multilabel_classes: int = 8):
        """初始化简化模型"""
        super().__init__()

        # 特征投影层
        self.image_projection = nn.Linear(image_feature_dim, hidden_dim)
        self.text_projection = nn.Linear(text_feature_dim, hidden_dim)

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 分类头
        self.binary_classifier = nn.Linear(hidden_dim // 2, num_binary_classes)
        self.multilabel_classifier = nn.Linear(hidden_dim // 2, num_multilabel_classes)

    def forward(self, image_features: torch.Tensor,
                text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 投影
        image_proj = F.relu(self.image_projection(image_features))
        text_proj = F.relu(self.text_projection(text_features))

        # 拼接
        combined = torch.cat([image_proj, text_proj], dim=1)

        # 融合
        fused = self.fusion_layer(combined)

        # 分类
        binary_logits = self.binary_classifier(fused)
        multilabel_logits = self.multilabel_classifier(fused)

        return {
            'binary_logits': binary_logits,
            'multilabel_logits': multilabel_logits
        }

def test_model():
    """测试模型"""
    print("测试多模态融合模型...")

    # 创建虚拟输入
    batch_size = 4
    image_tensor = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 10000, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)

    # 测试完整模型
    model = MultimodalFusionModel()
    outputs = model(image_tensor, input_ids, attention_mask)

    print(f"输入形状: 图像={image_tensor.shape}, 文本={input_ids.shape}")
    print(f"输出形状:")
    print(f"  binary_logits: {outputs['binary_logits'].shape}")
    print(f"  multilabel_logits: {outputs['multilabel_logits'].shape}")
    print(f"  image_features: {outputs['image_features'].shape}")
    print(f"  text_features: {outputs['text_features'].shape}")
    print(f"  fused_features: {outputs['fused_features'].shape}")

    # 测试简化模型
    simple_model = SimpleMultimodalModel()
    image_features = torch.randn(batch_size, 512)
    text_features = torch.randn(batch_size, 768)
    simple_outputs = simple_model(image_features, text_features)

    print(f"\n简化模型输出:")
    print(f"  binary_logits: {simple_outputs['binary_logits'].shape}")
    print(f"  multilabel_logits: {simple_outputs['multilabel_logits'].shape}")

    # 测试注意力权重
    attention_weights = model.get_attention_weights(image_tensor, input_ids, attention_mask)
    print(f"\n注意力权重形状: {attention_weights.shape}")

    print("\n模型测试完成!")

if __name__ == "__main__":
    test_model()