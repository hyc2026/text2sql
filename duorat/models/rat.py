from typing import Tuple, Callable, Optional

import torch
from torch import nn
import torch.nn.functional as F


class RelationAwareMultiheadAttention(nn.Module):
    """
    Relation-Aware Multi-Headed Attention (RAMHA).
    这是 DuoRAT 的核心网络层，涉及到 relation embedding 关系的建模
    它在 Encoder 中仅作为自注意力层，在 Decoder 中先进行自注意力计算、再进行互注意力计算 (与 Memory 交互)
    """

    def __init__(
        self,
        embed_dim: int,  # Query
        k_embed_dim: int,  # Key 在互注意力计算中，K、V 为 Memory Embedding
        v_embed_dim: int,  # Value
        num_heads: int,  # 默认设置中，Encoder 和 Decoder 的 num_heads 均为 8
        attention_dropout: float,  # 默认设置中，Encoder 和 Decoder 的 attention_dropout 均为 0.1
    ) -> None:
        super(RelationAwareMultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.k_embed_dim = k_embed_dim
        self.v_embed_dim = v_embed_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        self.scaling = self.head_dim ** -0.5

        self.q_in_proj = nn.Linear(embed_dim, embed_dim)
        self.k_in_proj = nn.Linear(k_embed_dim, embed_dim)
        self.v_in_proj = nn.Linear(v_embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        """末维度拆成多头 (embed_dim == num_heads * head_dim)，
        并让末尾两个维度 (会参与 bmm 矩阵计算) 改为 seq_len, head_dim
            Input shapes:
                - x:    (batch_size, seq_len, embed_dim)

            Output shapes:
                - x:    (batch_size, self.num_heads, seq_len, self.head_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    def _mask_attention(
        self, attn_weights: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply additive attention mask, broadcasting over attention heads. 设置注意力掩码
            Input shapes:
                - attn_weights:     (batch_size, self.num_heads, seq_b_len, seq_a_len)
                - attention_mask:   (batch_size, seq_b_len, seq_a_len), optional

            Output shapes:
                - attn_weights:     (batch_size, seq_b_len, seq_a_len)
        """

        if attention_mask is not None:
            batch_size, num_heads, seq_b_len, seq_a_len = attn_weights.shape
            assert num_heads == self.num_heads

            assert attention_mask.shape == (batch_size, seq_b_len, seq_a_len)
            # assert attention_mask.dtype == attn_weights.dtype

            # 注意力掩码 attention_mask 在 dim=1 扩展一个维度，并通过广播机制扩展到与 attn_weights 形状相同，然后进行掩码处理
            # 需要掩盖的地方，attention_mask 的值为 -inf；不需要掩盖的地方，attention_mask 的值为 0。于是直接相加就完成掩码处理
            return attn_weights + attention_mask.unsqueeze(1)
        else:
            return attn_weights

    def _mask_key_paddings(
        self, attn_weights: torch.Tensor, key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Mask attention weights corresponding to <pad> tokens. 把填充符号 <pad> 对应的位置掩盖掉
            Input shapes:
                - attn_weights:     (batch_size, self.num_heads, seq_b_len, seq_a_len)
                - key_padding_mask: (batch_size, seq_a_len), optional

            Output shapes:
                - attn_weights:     (batch_size, self.num_heads, seq_b_len, seq_a_len)
        """

        if key_padding_mask is not None:
            batch_size, num_heads, seq_b_len, seq_a_len = attn_weights.shape
            assert num_heads == self.num_heads

            assert key_padding_mask.shape == (batch_size, seq_a_len)
            # assert key_padding_mask.dtype == torch.bool

            # 填充符号掩码 key_padding_mask 扩展两个维度，并通过广播机制扩展到与 attn_weights 形状相同，然后进行掩码处理
            # 需要掩盖的地方，attention_mask 的值为 -inf；不需要掩盖的地方，attention_mask 的值为 0。于是直接相加就完成掩码处理
            return torch.masked_fill(
                attn_weights,
                mask=key_padding_mask.unsqueeze(1).unsqueeze(2),
                value=-float("inf"),
            )
        else:
            return attn_weights

    def _attn_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        relations_k: Optional[torch.Tensor],  # K 的关系嵌入
    ) -> torch.Tensor:
        """
        获得注意力权重张量，如果 K 的关系嵌入存在，则 Q 不仅会和 K 计算注意力，还会和 K 的关系嵌入计算注意力，两个结果张量会加起来
            Input shapes:
                - query:            (batch_size, seq_b_len, embed_dim)
                - key:              (batch_size, seq_a_len, k_embed_dim)
                - relations_k:      (batch_size, seq_b_len, seq_a_len, head_dim), optional

            Output shapes:
                - attn_weights:     (batch_size, self.num_heads, seq_b_len, seq_a_len)
        """
        batch_size, seq_b_len, _ = query.shape
        _batch_size, seq_a_len, _ = key.shape
        assert _batch_size == batch_size

        # 对 query 进行输入的全连接映射，然后进行 element-wise 的缩放，最后把张量 _reshape 为方便多头注意力计算的形状
        q = self._reshape(self.q_in_proj(query) * self.scaling)
        assert q.shape == (batch_size, self.num_heads, seq_b_len, self.head_dim)

        # 对 key 进行输入的全连接映射，然后把张量 _reshape 为方便多头注意力计算的形状
        k = self._reshape(self.k_in_proj(key))
        assert k.shape == (batch_size, self.num_heads, seq_a_len, self.head_dim)

        # 此时 q 和 k 的张量形状均为 (batch_size, self.num_heads, seq_len, self.head_dim)
        # 而 k_t 为 k 在最末两个维度的转置
        k_t = k.transpose(2, 3)
        assert k_t.shape == (batch_size, self.num_heads, self.head_dim, seq_a_len)

        # 矩阵乘法 (仅作用于高维张量的最末两个维度)
        attn_weights = torch.matmul(q, k_t)
        assert attn_weights.shape == (batch_size, self.num_heads, seq_b_len, seq_a_len)

        # 如果存在 K 的关系嵌入 relations_k，则进行交互计算
        # relations_k 的张量形状为 (batch_size, seq_b_len, seq_a_len, head_dim)
        if relations_k is not None:
            q_t = q.transpose(1, 2)
            assert q_t.shape == (batch_size, seq_b_len, self.num_heads, self.head_dim)

            relations_k_t = relations_k.transpose(2, 3)
            assert relations_k_t.shape == (batch_size, seq_b_len, self.head_dim, seq_a_len)

            # 直接加上 Q 与 K 的关系嵌入的 矩阵乘法结果张量，即满足论文的公式
            attn_weights += torch.matmul(q_t, relations_k_t).transpose(1, 2)
            assert attn_weights.shape == (batch_size, self.num_heads, seq_b_len, seq_a_len)

        return attn_weights

    def _attn(
        self,
        attn_weights: torch.Tensor,
        value: torch.Tensor,
        relations_v: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Calculate attention output. 对 value 进行注意力计算，其中利用到了 V 的关系嵌入
            Input shapes:
                - attn_weights:     (batch_size, self.num_heads, seq_b_len, seq_a_len)
                - value:            (batch_size, seq_a_len, v_embed_dim)
                - relations_v:      (batch_size, seq_b_len, seq_a_len, head_dim), optional

            Output shapes:
                - attn:             (batch_size, seq_b_len, self.embed_dim)
        """

        batch_size, num_heads, seq_b_len, seq_a_len = attn_weights.shape
        assert num_heads == self.num_heads

        # 对 value 进行输入的全连接映射，然后把张量 _reshape 为方便多头注意力计算的形状
        v = self._reshape(self.v_in_proj(value))
        assert v.shape == (batch_size, self.num_heads, seq_a_len, self.head_dim)

        # 注意力权重 attn_weights 对 value 进行矩阵乘法，获得注意力计算结果
        attn = torch.matmul(attn_weights, v).transpose(1, 2)
        assert attn.shape == (batch_size, seq_b_len, self.num_heads, self.head_dim)

        # 如果存在 V 的关系嵌入 relations_v，则进行交互计算
        # relations_v 的张量形状为 (batch_size, seq_b_len, seq_a_len, head_dim)
        if relations_v is not None:
            attn_weights_t = attn_weights.transpose(1, 2)
            assert attn_weights_t.shape == (batch_size, seq_b_len, self.num_heads, seq_a_len)

            assert relations_v.shape == (batch_size, seq_b_len, seq_a_len, self.head_dim)

            # 直接加上注意力权重 attn_weights 对 V 的关系嵌入的 矩阵乘法结果张量，即满足论文的公式
            attn += torch.matmul(attn_weights_t, relations_v)
            assert attn.shape == (batch_size, seq_b_len, self.num_heads, self.head_dim)

        # 把最末两个维度合并起来，即 self.num_heads * self.head_dim == self.embed_dim
        attn = attn.reshape(batch_size, seq_b_len, self.embed_dim)

        # 最后进行输出全连接映射
        attn = self.out_proj(attn)
        assert attn.shape == (batch_size, seq_b_len, self.embed_dim)

        return attn

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        relations_k: Optional[torch.Tensor],  # K 的关系嵌入
        relations_v: Optional[torch.Tensor],  # V 的关系嵌入
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for relation-aware multi-headed attention.

        Input shapes:
            - query:            (batch_size, seq_b_len, embed_dim)
            - key:              (batch_size, seq_a_len, k_embed_dim)
            - value:            (batch_size, seq_a_len, v_embed_dim)
            - relations_k:      (batch_size, seq_b_len, seq_a_len, head_dim), optional
            - relations_v:      (batch_size, seq_b_len, seq_a_len, head_dim), optional
            - attention_mask:   (batch_size, seq_b_len, seq_a_len), optional
            - key_padding_mask: (batch_size, seq_a_len), optional

        Output shapes:
            - attn:             (batch_size, seq_b_len, embed_dim)
            - attn_weights:     (batch_size, seq_b_len, seq_a_len)
        """

        batch_size, seq_b_len, embed_dim = query.shape
        assert embed_dim == self.embed_dim

        _batch_size, seq_a_len, k_embed_dim = key.shape
        assert _batch_size == batch_size
        assert k_embed_dim == self.k_embed_dim

        _batch_size, _seq_a_len, v_embed_dim = value.shape
        assert _batch_size == batch_size
        assert _seq_a_len == seq_a_len
        assert v_embed_dim == self.v_embed_dim

        # 获得注意力权重张量 attn_weights
        # 如果 K 的关系嵌入 relations_k 存在，则 Q 不仅会和 K 计算注意力，还会和 K 的关系嵌入计算注意力，两个结果张量会加起来
        attn_weights = self._attn_weights(query, key, relations_k)

        # 设置注意力掩码
        attn_weights = self._mask_attention(attn_weights, attention_mask)
        assert attn_weights.shape == (batch_size, self.num_heads, seq_b_len, seq_a_len)

        # 把填充符号 <pad> 对应的位置掩盖掉
        attn_weights = self._mask_key_paddings(attn_weights, key_padding_mask)
        assert attn_weights.shape == (batch_size, self.num_heads, seq_b_len, seq_a_len)

        # 对注意力权重的末维度进行 softmax 归一化，然后进行 dropout (默认配置中 attention_dropout == 0.1)
        attn_weights = F.softmax(attn_weights, dim=3)
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        assert attn_weights.shape == (batch_size, self.num_heads, seq_b_len, seq_a_len)

        # 利用前面计算得到的注意力权重 attn_weights 对 value 进行注意力计算，其中利用到了 V 的关系嵌入
        attn = self._attn(attn_weights, value, relations_v)
        assert attn.shape == (batch_size, seq_b_len, self.embed_dim)

        # average attention weights over heads
        # 把注意力权重的多头信息 加起来取算术平均 (dim=1 为 self.num_heads 维度)
        attn_weights = attn_weights.sum(dim=1, keepdim=False) / self.num_heads
        assert attn_weights.shape == (batch_size, seq_b_len, seq_a_len)

        # 返回最终的注意力计算结果，以及注意力权重分数
        return attn, attn_weights


def _residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    dropout: Callable[[torch.Tensor], torch.Tensor],
    norm: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """进行残差连接 return norm(residual + dropout(x))"""
    shape = x.shape
    dtype = x.dtype
    device = x.device
    assert residual.shape == shape
    # assert residual.dtype == dtype
    assert residual.device == device
    x = residual + dropout(x)
    x = norm(x)
    assert x.shape == shape
    # assert x.dtype == dtype
    assert x.device == device
    return x


class DressedRelationAwareMultiheadAttention(nn.Module):
    """Relation-Aware Multi-Headed Attention (RAMHA) with residual connection and layer norm."""

    def __init__(
        self,
        embed_dim: int,
        k_embed_dim: int,
        v_embed_dim: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super(DressedRelationAwareMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

        # 利用了关系信息的多头注意力层
        self.self_attn = RelationAwareMultiheadAttention(
            embed_dim=embed_dim,
            k_embed_dim=k_embed_dim,
            v_embed_dim=v_embed_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        relations_k: Optional[torch.Tensor],
        relations_v: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for relation-aware multi-headed attention
        with residual connection and layer norms.

        Input shapes:
            - query:            (batch_size, seq_b_len, embed_dim)
            - key:              (batch_size, seq_a_len, k_embed_dim)
            - value:            (batch_size, seq_a_len, v_embed_dim)
            - relations_k:      (batch_size, seq_b_len, seq_a_len, head_dim), optional
            - relations_v:      (batch_size, seq_b_len, seq_a_len, head_dim), optional
            - attention_mask:   (batch_size, seq_b_len, seq_a_len), optional
            - key_padding_mask: (batch_size, seq_a_len), optional

        Output shapes:
            - y:                (batch_size, seq_b_len, embed_dim)
        """

        batch_size, seq_len, embed_dim = query.shape
        assert embed_dim == self.embed_dim

        z, _ = self.self_attn(
            query=query,
            key=key,
            value=value,
            relations_k=relations_k,
            relations_v=relations_v,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )

        # 进行残差连接，下面等价于 y = self.norm(residual + F.dropout(y))
        y = _residual(
            z,
            query,
            lambda z_: F.dropout(z_, p=self.dropout, training=self.training),
            self.norm,
        )

        return y


class TransformerMLP(nn.Module):
    """Transformer MLP Layer."""

    def __init__(
        self, embed_dim: int, ffn_dim: int, dropout: float, relu_dropout: float
    ) -> None:
        super(TransformerMLP, self).__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Forward pass for the transformer MLP layer.

        Input shapes:
            - y:                (batch_size, seq_len, embed_dim)

        Output shapes:
            - y:                (batch_size, seq_len, embed_dim)
        """

        batch_size, seq_len, embed_dim = y.shape
        assert embed_dim == self.embed_dim

        residual = y
        y = F.relu(self.fc1(y))
        assert y.shape == (batch_size, seq_len, self.ffn_dim)
        y = F.dropout(y, p=self.relu_dropout, training=self.training)
        y = self.fc2(y)
        # 上述 y = self.fc2(F.dropout(F.relu(self.fc1(y))))

        # 进行残差连接，下面等价于 y = self.norm(residual + F.dropout(y))
        y = _residual(
            y,
            residual,
            lambda y_: F.dropout(y_, p=self.dropout, training=self.training),
            self.norm,
        )

        return y


class RATLayer(nn.Module):
    """Relation-Aware Transformer Layer Block. 用于构建 DuoRAT 编码器"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        attention_dropout: float,
        relu_dropout: float,
    ) -> None:
        super(RATLayer, self).__init__()

        # 利用了关系信息的多头注意力层 ("Dressed" 意为：加上了残差连接 和 layer norm)
        self.self_attn = DressedRelationAwareMultiheadAttention(
            embed_dim=embed_dim,  # Q、K、V 相同，为自注意力
            k_embed_dim=embed_dim,
            v_embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        # Transformer 全连接层 (含两个 Linear 层、两个 dropout 层、一个 layer norm 层 以及 一次残差连接)
        self.mlp = TransformerMLP(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            relu_dropout=relu_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        relations_k: Optional[torch.Tensor],
        relations_v: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for the transformer layer.

        Input shapes:
            - x:                (batch_size, seq_len, embed_dim)
            - relations_k:      (batch_size, seq_len, seq_len, head_dim), optional
            - relations_v:      (batch_size, seq_len, seq_len, head_dim), optional
            - attention_mask:   (batch_size, seq_len, seq_len), optional
            - key_padding_mask: (batch_size, seq_len), optional

        Output shapes:
            - x:                (batch_size, seq_len, embed_dim)
        """
        return self.mlp(
            self.self_attn(
                query=x,
                key=x,
                value=x,
                relations_k=relations_k,
                relations_v=relations_v,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
            )
        )


class RATLayerWithMemory(nn.Module):
    """Relation-Aware Transformer Layer Block with Memory. 用于构建 DuoRAT 解码器"""

    def __init__(
        self,
        embed_dim: int,
        mem_embed_dim: int,
        dropout: float,
        num_heads: int,
        attention_dropout: float,
        relu_dropout: float,
        ffn_dim: int,
    ) -> None:
        super(RATLayerWithMemory, self).__init__()

        # 同 RATLayer 的首层：利用了关系信息的多头注意力层 ("Dressed" 意为：加上了残差连接 和 layer norm)
        self.self_attn = DressedRelationAwareMultiheadAttention(
            embed_dim=embed_dim,  # Q、K、V 相同，为自注意力
            k_embed_dim=embed_dim,
            v_embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        # 与 Memory 的互注意力层，也是利用了关系信息的多头注意力层
        self.memory_attn = DressedRelationAwareMultiheadAttention(
            embed_dim=embed_dim,
            k_embed_dim=mem_embed_dim,  # K、V 为 memory embedding
            v_embed_dim=mem_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        # 同 RATLayer 的末层：Transformer 全连接层 (含两个 Linear 层、两个 dropout 层、一个 layer norm 层 以及 一次残差连接)
        self.mlp = TransformerMLP(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            relu_dropout=relu_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        relations_k: Optional[torch.Tensor],
        memory_relations_k: Optional[torch.Tensor],
        relations_v: Optional[torch.Tensor],
        memory_relations_v: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        memory_attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for the transformer layer with memory.

        Input shapes:
            - x:                       (batch_size, seq_len, embed_dim)
            - memory:                  (batch_size, mem_len, mem_embed_dim)
            - relations_k:             (batch_size, seq_len, seq_len, head_dim), optional
            - memory_relations_k:      (batch_size, seq_len, mem_len, head_dim), optional
            - relations_v:             (batch_size, seq_len, seq_len, head_dim), optional
            - memory_relations_v:      (batch_size, seq_len, mem_len, head_dim), optional
            - attention_mask:          (batch_size, seq_len, seq_len), optional
            - memory_attention_mask:   (batch_size, seq_len, mem_len), optional
            - key_padding_mask:        (batch_size, seq_len), optional
            - memory_key_padding_mask: (batch_size, mem_len), optional

        Output shapes:
            - x:                       (batch_size, seq_len, embed_dim)
        """

        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            relations_k=relations_k,
            relations_v=relations_v,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        x = self.memory_attn(
            query=x,
            key=memory,
            value=memory,
            relations_k=memory_relations_k,
            relations_v=memory_relations_v,
            attention_mask=memory_attention_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        x = self.mlp(x)

        return x


class RAT(nn.Module):
    """RAT-SQL 的模型架构，在 DuoRAT 中没有使用"""
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_embed_x: int,
        pad_x_index: int,
        num_embed_r_k: int,
        pad_r_k_index: int,
        num_embed_r_v: int,
        pad_r_v_index: int,
        dropout: float,
        attention_dropout: float,
        relu_dropout: float,
    ):
        super(RAT, self).__init__()

        self.embed_dim = embed_dim
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim
        self.head_dim = head_dim

        self.num_embed_x = num_embed_x
        self.pad_x_index = pad_x_index
        self.embed_x = nn.Embedding(num_embed_x, embed_dim, padding_idx=pad_x_index)
        self.embed_x_scale = embed_dim ** 0.5

        self.num_embed_r_k = num_embed_r_k
        self.pad_r_k_index = pad_r_k_index
        self.embed_r_k = nn.Embedding(
            num_embed_r_k, head_dim, padding_idx=pad_r_k_index
        )
        self.num_embed_r_v = num_embed_r_v
        self.pad_r_v_index = pad_r_v_index
        self.embed_r_v = nn.Embedding(
            num_embed_r_v, head_dim, padding_idx=pad_r_v_index
        )
        self.embed_r_scale = head_dim ** 0.5

        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                RATLayer(
                    embed_dim=embed_dim,
                    dropout=dropout,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    ffn_dim=ffn_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj = nn.Linear(embed_dim, num_embed_x)

    def _forward(
        self,
        x_tokens: torch.Tensor,
        relation_tokens_k: torch.Tensor,
        relation_tokens_v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        (batch_size, seq_len) = x_tokens.shape
        assert x_tokens.dtype == torch.long
        assert relation_tokens_k.shape == (batch_size, seq_len, seq_len)
        # assert relation_tokens_k.dtype == torch.long
        assert relation_tokens_v.shape == (batch_size, seq_len, seq_len)
        # assert relation_tokens_v.dtype == torch.long

        key_padding_mask = x_tokens.eq(self.pad_x_index)
        assert key_padding_mask.shape == (batch_size, seq_len)
        # assert key_padding_mask.dtype == torch.bool

        x = self.embed_x_scale * self.embed_x(x_tokens)
        assert x.shape == (batch_size, seq_len, self.embed_dim)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len, seq_len)
            # assert attention_mask.dtype == x.dtype

        relations_k = self.embed_r_scale * self.embed_r_k(relation_tokens_k)
        assert relations_k.shape == (batch_size, seq_len, seq_len, self.head_dim)

        relations_v = self.embed_r_scale * self.embed_r_v(relation_tokens_v)
        assert relations_v.shape == (batch_size, seq_len, seq_len, self.head_dim)

        for layer in self.layers:
            x = layer(
                x=x,
                relations_k=relations_k,
                relations_v=relations_v,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
            )

        assert x.shape == (batch_size, seq_len, self.embed_dim)

        return x

    def forward(
        self,
        x_tokens: torch.Tensor,
        relation_tokens_k: torch.Tensor,
        relation_tokens_v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for the relation-aware transformer.

        Input shapes:
            - x_tokens:          (batch_size, seq_len)
            - relation_tokens_k: (batch_size, seq_len, seq_len)
            - relation_tokens_v: (batch_size, seq_len, seq_len)
            - attention_mask:    (batch_size, seq_len, seq_len), optional

        Output shapes:
            - x_token_logits:    (batch_size, seq_len, num_embed_x)
        """

        (batch_size, seq_len) = x_tokens.shape
        x = self._forward(
            x_tokens=x_tokens,
            relation_tokens_k=relation_tokens_k,
            relation_tokens_v=relation_tokens_v,
            attention_mask=attention_mask,
        )
        x_token_logits = self.proj(x)
        assert x_token_logits.shape == (batch_size, seq_len, self.num_embed_x)

        return x_token_logits


class RATWithMemory(nn.Module):
    """使用了 Memory 机制的 RAT-SQL 模型架构，在 DuoRAT 中没有使用"""
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        mem_embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_embed_x: int,
        pad_x_index: int,
        num_embed_r_k: int,
        pad_r_k_index: int,
        num_mem_embed_r_k: int,
        pad_mem_r_k_index: int,
        num_embed_r_v: int,
        pad_r_v_index: int,
        num_mem_embed_r_v: int,
        pad_mem_r_v_index: int,
        dropout: float,
        attention_dropout: float,
        relu_dropout: float,
    ):
        super(RATWithMemory, self).__init__()

        self.embed_dim = embed_dim
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim
        self.head_dim = head_dim
        self.mem_embed_dim = mem_embed_dim

        self.num_embed_x = num_embed_x
        self.pad_x_index = pad_x_index
        self.embed_x = nn.Embedding(num_embed_x, embed_dim, padding_idx=pad_x_index)
        self.embed_x_scale = embed_dim ** 0.5

        self.num_embed_r_k = num_embed_r_k
        self.pad_r_k_index = pad_r_k_index
        self.embed_r_k = nn.Embedding(
            num_embed_r_k, head_dim, padding_idx=pad_r_k_index
        )
        self.num_mem_embed_r_k = num_mem_embed_r_k
        self.pad_mem_r_k_index = pad_mem_r_k_index
        self.mem_embed_r_k = nn.Embedding(
            num_mem_embed_r_k, head_dim, padding_idx=pad_mem_r_k_index
        )
        self.num_embed_r_v = num_embed_r_v
        self.pad_r_v_index = pad_r_v_index
        self.embed_r_v = nn.Embedding(
            num_embed_r_v, head_dim, padding_idx=pad_r_v_index
        )
        self.num_mem_embed_r_v = num_mem_embed_r_v
        self.pad_mem_r_v_index = pad_mem_r_v_index
        self.mem_embed_r_v = nn.Embedding(
            num_mem_embed_r_v, head_dim, padding_idx=pad_mem_r_v_index
        )
        self.embed_r_scale = head_dim ** 0.5

        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                RATLayerWithMemory(
                    embed_dim=embed_dim,
                    mem_embed_dim=mem_embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj = nn.Linear(embed_dim, num_embed_x)

    def _forward(
        self,
        x_tokens: torch.Tensor,
        memory: torch.Tensor,
        relation_tokens_k: torch.Tensor,
        memory_relation_tokens_k: torch.Tensor,
        relation_tokens_v: torch.Tensor,
        memory_relation_tokens_v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        memory_attention_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len = x_tokens.shape
        # assert x_tokens.dtype == torch.long
        _batch_size, mem_len, mem_embed_dim = memory.shape
        assert _batch_size == batch_size
        assert mem_embed_dim == self.mem_embed_dim
        assert relation_tokens_k.shape == (batch_size, seq_len, seq_len)
        # assert relation_tokens_k.dtype == torch.long
        assert memory_relation_tokens_k.shape == (batch_size, seq_len, mem_len)
        # assert memory_relation_tokens_k.dtype == torch.long
        assert relation_tokens_v.shape == (batch_size, seq_len, seq_len)
        # assert relation_tokens_v.dtype == torch.long
        assert memory_relation_tokens_v.shape == (batch_size, seq_len, mem_len)
        # assert memory_relation_tokens_v.dtype == torch.long

        key_padding_mask = x_tokens.eq(self.pad_x_index)
        assert key_padding_mask.shape == (batch_size, seq_len)
        # assert key_padding_mask.dtype == torch.bool

        if memory_key_padding_mask is not None:
            assert memory_key_padding_mask.shape == (batch_size, mem_len)
            # assert memory_key_padding_mask.dtype == torch.bool

        x = self.embed_x_scale * self.embed_x(x_tokens)
        assert x.shape == (batch_size, seq_len, self.embed_dim)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len, seq_len)
            # assert attention_mask.dtype == x.dtype

        if memory_key_padding_mask is not None:
            assert memory_attention_mask.shape == (batch_size, seq_len, mem_len)
            # assert memory_attention_mask.dtype == x.dtype

        relations_k = self.embed_r_scale * self.embed_r_k(relation_tokens_k)
        assert relations_k.shape == (batch_size, seq_len, seq_len, self.head_dim)

        memory_relations_k = self.embed_r_scale * self.mem_embed_r_k(
            memory_relation_tokens_k
        )
        assert memory_relations_k.shape == (batch_size, seq_len, mem_len, self.head_dim)

        relations_v = self.embed_r_scale * self.embed_r_v(relation_tokens_v)
        assert relations_v.shape == (batch_size, seq_len, seq_len, self.head_dim)

        memory_relations_v = self.embed_r_scale * self.mem_embed_r_v(
            memory_relation_tokens_v
        )
        assert memory_relations_v.shape == (batch_size, seq_len, mem_len, self.head_dim)

        for layer in self.layers:
            x = layer(
                x=x,
                memory=memory,
                relations_k=relations_k,
                memory_relations_k=memory_relations_k,
                relations_v=relations_v,
                memory_relations_v=memory_relations_v,
                attention_mask=attention_mask,
                memory_attention_mask=memory_attention_mask,
                key_padding_mask=key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        assert x.shape == (batch_size, seq_len, self.embed_dim)

        return x

    def forward(
        self,
        x_tokens: torch.Tensor,
        memory: torch.Tensor,
        relation_tokens_k: torch.Tensor,
        memory_relation_tokens_k: torch.Tensor,
        relation_tokens_v: torch.Tensor,
        memory_relation_tokens_v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        memory_attention_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for the relation-aware transformer with memory.

        Input shapes:
            - x_tokens:                 (batch_size, seq_len)
            - memory:                   (batch_size, mem_len, mem_embed_dim)
            - relation_tokens_k:        (batch_size, seq_len, seq_len)
            - memory_relation_tokens_k: (batch_size, seq_len, mem_len)
            - relation_tokens_v:        (batch_size, seq_len, seq_len)
            - memory_relation_tokens_v: (batch_size, seq_len, mem_len)
            - attention_mask:           (batch_size, seq_len, seq_len), optional
            - memory_attention_mask:    (batch_size, seq_len, mem_len), optional
            - memory_key_padding_mask:  (batch_size, mem_len), optional

        Output shapes:
            - x_token_logits:           (batch_size, seq_len, num_embed_x)
        """

        (batch_size, seq_len) = x_tokens.shape
        x = self._forward(
            x_tokens=x_tokens,
            memory=memory,
            relation_tokens_k=relation_tokens_k,
            memory_relation_tokens_k=memory_relation_tokens_k,
            relation_tokens_v=relation_tokens_v,
            memory_relation_tokens_v=memory_relation_tokens_v,
            attention_mask=attention_mask,
            memory_attention_mask=memory_attention_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        x_token_logits = self.proj(x)
        assert x_token_logits.shape == (batch_size, seq_len, self.num_embed_x)

        return x_token_logits
