"""MultiHeadedAttentionモジュールのテスト"""

import torch

from unofficial_slash.network.transformer.attention import MultiHeadedAttention


def _setup_attention_with_predictable_weights(head_size, hidden_size):
    """予測しやすい重みでアテンションを設定して返す"""
    attention = MultiHeadedAttention(
        head_size=head_size, hidden_size=hidden_size, dropout_rate=0.0
    )

    with torch.no_grad():
        attention.linear_q.weight.data.fill_(1.0)
        attention.linear_k.weight.data.fill_(2.0)
        attention.linear_v.weight.data.fill_(3.0)
        attention.linear_out.weight.data = torch.eye(
            attention.linear_out.weight.size(0),
            attention.linear_out.weight.size(1),
        )

        attention.linear_q.bias.data.fill_(0.0)
        attention.linear_k.bias.data.fill_(0.0)
        attention.linear_v.bias.data.fill_(0.0)
        attention.linear_out.bias.data.fill_(0.0)

    attention.eval()
    return attention


def test_mask_effectiveness_with_predictable_weights():
    """マスク効果テスト"""
    attention = _setup_attention_with_predictable_weights(head_size=2, hidden_size=4)

    batch_size = 1
    seq_len = 3
    hidden_size = 4

    # 3番目に毒を仕込む
    query = torch.ones(batch_size, seq_len, hidden_size)
    key = torch.ones(batch_size, seq_len, hidden_size)
    value = torch.ones(batch_size, seq_len, hidden_size) * 10.0
    value[0, 2, :] = 999.0

    mask_all_valid = torch.tensor([[[1, 1, 1]]])  # 全て有効
    mask_poison_blocked = torch.tensor(
        [[[1, 1, 0]]]
    )  # 3番目をマスク（0=マスク、1=有効）

    with torch.no_grad():
        output_with_poison = attention.forward(query, key, value, mask_all_valid)
        output_poison_blocked = attention.forward(
            query, key, value, mask_poison_blocked
        )

    # マスクで毒の値の影響が遮断されることを確認
    assert not torch.allclose(output_with_poison, output_poison_blocked)
    assert not torch.any(torch.isclose(output_poison_blocked, torch.tensor(999.0)))


@torch.no_grad()
def test_qkv_computation_verification():
    """Q,K,V変換の計算確認テスト"""
    attention = _setup_attention_with_predictable_weights(head_size=1, hidden_size=2)

    batch_size = 1
    seq_len = 2
    hidden_size = 2

    # 単純な入力で計算を追跡しやすく
    query = torch.ones(batch_size, seq_len, hidden_size)
    key = torch.ones(batch_size, seq_len, hidden_size)
    value = torch.ones(batch_size, seq_len, hidden_size)

    # Q,K,V変換の計算値確認
    q, k, v = attention.forward_qkv(query, key, value)

    expected_q_val = 2.0  # 1*1 + 1*1 = 2
    expected_k_val = 4.0  # 2*1 + 2*1 = 4
    expected_v_val = 6.0  # 3*1 + 3*1 = 6

    assert torch.allclose(q, torch.full_like(q, expected_q_val))
    assert torch.allclose(k, torch.full_like(k, expected_k_val))
    assert torch.allclose(v, torch.full_like(v, expected_v_val))

    # 最終出力の計算値確認
    output = attention.forward(query, key, value, mask=None)
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert torch.all(torch.isfinite(output))

    # 予測される最終出力値
    # アテンションで全位置が等重みになり、V=6.0が各位置に出力される
    expected_output = torch.tensor([[[6.0, 6.0], [6.0, 6.0]]])
    assert torch.allclose(output, expected_output)
