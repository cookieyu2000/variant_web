import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class BioMedBERTClassifier(nn.Module):
    """
    使用 BioMedBERT 作為 encoder，並在多段落上做 Transformer 聚合或平均後分類。
    Args:
        pretrained_model_name_or_path (str): 預訓練模型的名稱或路徑。
        num_labels (int): 分類標籤數量。
        dropout_rate (float): Dropout 比率。
        transformer_config (dict): Transformer 聚合配置。
        use_transformer (bool): 是否使用 Transformer 聚合，False 則平均池化。
    """
    def __init__(
        self,
        pretrained_model_name_or_path: str = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
        num_labels: int = 2,
        dropout_rate: float = 0.3,
        transformer_config: dict = None,
        use_transformer: bool = True
    ):
        super().__init__()
        cfg = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_labels
        )
        self.bert = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            config=cfg
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.use_transformer = use_transformer

        if self.use_transformer:
            transformer_config = transformer_config or {}
            hidden_size = transformer_config.get('hidden_size', cfg.hidden_size)
            num_heads = transformer_config.get('num_heads', 8)
            num_layers = transformer_config.get('num_layers', 2)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                activation='relu'
            )
            self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.classifier = nn.Linear(cfg.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: torch.LongTensor = None
    ) -> torch.Tensor:
        B, P, L = input_ids.size()
        flat_ids = input_ids.view(B * P, L)
        flat_mask = attention_mask.view(B * P, L)
        flat_token = token_type_ids.view(B * P, L) if token_type_ids is not None else None

        outputs = self.bert(
            input_ids=flat_ids,
            attention_mask=flat_mask,
            token_type_ids=flat_token
        )
        pooled = outputs.pooler_output.view(B, P, -1)

        if self.use_transformer:
            t_in = pooled.permute(1, 0, 2) # [P, B, H]
            t_out = self.transformer_encoder(t_in)
            repr = t_out.permute(1, 0, 2).mean(dim=1) # [B, P, H]
        else:
            repr = pooled.mean(dim=1)

        repr = self.dropout(repr)
        logits = self.classifier(repr)
        return logits
