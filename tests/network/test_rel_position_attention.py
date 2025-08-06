"""RelPositionMultiHeadedAttentionモジュールのテスト"""

import torch

from unofficial_slash.network.transformer.attention import (
    RelPositionMultiHeadedAttention,
)


def _setup_rel_attention_with_predictable_weights(head_size, hidden_size):
    """予測しやすい重みで相対位置アテンションを設定して返す"""
    attention = RelPositionMultiHeadedAttention(
        head_size=head_size, hidden_size=hidden_size, dropout_rate=0.0
    )

    with torch.no_grad():
        attention.linear_q.weight.data.fill_(0.1)
        attention.linear_k.weight.data.fill_(0.1)
        attention.linear_v.weight.data.fill_(3.0)
        attention.linear_out.weight.data = torch.eye(
            attention.linear_out.weight.size(0),
            attention.linear_out.weight.size(1),
        )

        attention.linear_q.bias.data.fill_(0.0)
        attention.linear_k.bias.data.fill_(0.0)
        attention.linear_v.bias.data.fill_(0.0)
        attention.linear_out.bias.data.fill_(0.0)

        attention.linear_pos.weight.data.fill_(2.0)

        attention.bias_k.data.fill_(0.0)
        attention.bias_p.data.fill_(0.0)

    attention.eval()
    return attention


def test_mask_effectiveness_with_predictable_weights():
    """マスク効果テスト"""
    attention = _setup_rel_attention_with_predictable_weights(
        head_size=2, hidden_size=4
    )

    batch_size = 1
    seq_len = 3
    hidden_size = 4
    pos_len = 2 * seq_len - 1  # 5

    # 3番目に毒を仕込む
    query = torch.ones(batch_size, seq_len, hidden_size)
    key = torch.ones(batch_size, seq_len, hidden_size)
    value = torch.ones(batch_size, seq_len, hidden_size) * 10.0
    value[0, 2, :] = 999.0
    pos_emb = torch.ones(batch_size, pos_len, hidden_size)

    mask_all_valid = torch.tensor([[[1, 1, 1]]])  # 全て有効
    mask_poison_blocked = torch.tensor(
        [[[1, 1, 0]]]
    )  # 3番目をマスク（0=マスク、1=有効）

    with torch.no_grad():
        output_with_poison = attention.forward(
            query, key, value, pos_emb, mask_all_valid
        )
        output_poison_blocked = attention.forward(
            query, key, value, pos_emb, mask_poison_blocked
        )

    # マスクで毒の値の影響が遮断されることを確認
    assert not torch.allclose(output_with_poison, output_poison_blocked)
    assert not torch.any(torch.isclose(output_poison_blocked, torch.tensor(999.0)))


@torch.no_grad()
def test_rel_position_computation_verification():
    """相対位置アテンションの計算確認テスト"""
    attention = _setup_rel_attention_with_predictable_weights(
        head_size=1, hidden_size=2
    )

    batch_size = 1
    seq_len = 2
    hidden_size = 2

    query = torch.ones(batch_size, seq_len, hidden_size)
    key = torch.ones(batch_size, seq_len, hidden_size)
    value = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    pos_emb = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    # Q,K,V変換の計算値確認
    q, k, v = attention.forward_qkv(query, key, value)

    expected_q_val = 0.2  # 0.1*1 + 0.1*1 = 0.2
    expected_k_val = 0.2  # 0.1*1 + 0.1*1 = 0.2
    expected_v = torch.tensor(
        [
            [9.0, 9.0],  # 3*1 + 3*2 = 9.0
            [21.0, 21.0],  # 3*3 + 3*4 = 21.0
        ]
    ).reshape(1, 1, 2, 2)

    assert torch.allclose(q, torch.full_like(q, expected_q_val))
    assert torch.allclose(k, torch.full_like(k, expected_k_val))
    assert torch.allclose(v, expected_v)

    # 相対位置エンコーディングの効果確認
    output = attention.forward(query, key, value, pos_emb, mask=None)
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert torch.all(torch.isfinite(output))

    # 異なる位置エンベディングで出力が変わることを確認
    pos_emb_different = torch.tensor([[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]])
    output_different = attention.forward(
        query, key, value, pos_emb_different, mask=None
    )
    assert not torch.allclose(output, output_different)
