# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このリポジトリは「SLASH: Self-Supervised Speech Pitch Estimation Leveraging DSP-derived Absolute Pitch」論文をなるべく忠実に実装することが目的です。SLASH は自己教師あり学習（SSL）と従来のデジタル信号処理（DSP）を組み合わせて、音声の基本周波数（F0）推定を行う手法です。

## TODO

- 右チャンネルの音声を使うように
- ピッチのフレーム数がおかしいかも、pad のとこと dataset のとこと比較のとこと

## 損失関数設計

### 主要損失

1. **Pitch Consistency Loss (L_cons)**: 相対ピッチ学習

   ```
   L_cons = (1/T) Σ h(log2(p_t) - log2(p_shift_t) + d/12)
   ```

2. **Pitch Guide Loss (L_g)**: 絶対ピッチ事前分布

   ```
   L_g = (1/T) Σ max(1 - Σ P_t,f × G_t,f - m, 0)
   ```

3. **Pitch Guide Shift Loss (L_g-shift)**: シフトされた絶対ピッチ学習

   ```
   L_g-shift = (1/T) Σ max(1 - Σ P_shift_t,f × G_t,f-Δf - m, 0)
   ```

4. **Pseudo Spectrogram Loss (L_pseudo)**: F0 勾配最適化

   ```
   L_pseudo = ||ψ(S*) - ψ(S)||₁ ⊙ (v × 1_K)
   ```

5. **Reconstruction Loss (L_recon)**: GED による aperiodicity 最適化
   ```
   L_recon = ||ψ(Š¹) – ψ(S)||₁ – α ||ψ(Š¹) – ψ(Š²)||₁
   ```

### 補助損失

### ノイズロバスト損失（Section 2.6 準拠）

6. **L_aug**: 拡張データでの基本損失（Huber loss between p and p_aug）
   ```
   L_aug = ||p - p_aug||₁
   ```
7. **L_g-aug**: 拡張データでの Pitch Guide 損失（P_aug を使用した L_g と同等）
   ```
   L_g-aug = (1/T) Σ max(1 - Σ P_aug_t,f × G_t,f - m, 0)
   ```
8. **L_ap**: Aperiodicity 一貫性損失（`||log(A_aug) - 
og(A)||₁`）

## 実装上の重要ポイント

### DSP パイプライン

1. **CQT 設定**: frame_shift=5ms, f_min=32.70Hz, 205bins, 24bins/octave, filter_scale=0.5
2. **Pitch Guide**: fine structure spectrum → SHS → 正規化
3. **Pseudo Spectrogram**: 三角波振動子 → 周期性励起スペクトログラム
4. **Differentiable WORLD 合成**: 時間領域での周期・非周期成分合成 → 最小位相応答処理 → 音声再構成

### 学習設定

- Optimizer: AdamW (lr=0.0002)
- Dynamic batching (平均バッチサイズ 17)
- 学習ステップ: 100,000
- 損失重み: L_cons, L_pseudo=10.0, L_recon=5.0, L_guide, L_g-shift, その他=1.0

## 設定管理（Pydantic + YAML）

**現在の設定システム**: Pydantic BaseModel + YAML 設定ファイル

SLASH 用の設定構造については以下を参照：

- **設定クラス定義**: `src/unofficial_slash/config.py`
- **実際の設定例**: `tests/input_data/base_config.yaml`

## 書類・文献整理

### paper/ディレクトリ（関連論文集）

SLASH 実装に必要な論文群。メイン論文と関連研究をテキスト形式で収録。

- `SLASH: Self-Supervised Speech Pitch Estimation Leveraging DSP-derived Absolute Pitch.txt` - **メイン論文**
- `SPICE: Self-supervised Pitch Estimation.txt` - SSL 相対ピッチ学習の基礎
- `NANSY++: UNIFIED VOICE SYNTHESIS WITH NEURAL ANALYSIS AND SYNTHESIS.txt` - Pitch Encoder アーキテクチャ
- `DDSP: DIFFERENTIABLE DIGITAL SIGNAL PROCESSING.txt` - 微分可能 DSP の概念
- `PESTO: PITCH ESTIMATION WITH SELF-SUPERVISED TRANSPOSITION-EQUIVARIANT OBJECTIVE.txt` - SSL 比較手法
- `A Spectral Energy Distance for Parallel Speech Synthesis.txt` - GED 損失の原典
- `DIFFERENTIABLE WORLD SYNTHESIZER-BASED NEURAL VOCODER WITH APPLICATION TO END-TO-END AUDIO STYLE TRANSFER.txt` - DDSP Synthesizer 参考
- `ESPNET-TTS: UNIFIED, REPRODUCIBLE, AND INTEGRATABLE OPEN SOURCE END-TO-END TEXT-TO-SPEECH TOOLKIT.txt` - Dynamic batching 参考
- `Singing Voice Separation and Vocal F0 Estimation based on Mutual Combination of Robust Principal Component Analysis and Subharmonic Summation.txt` - SHS アルゴリズム

### docs/ディレクトリ（調査・分析資料）

SLASH 実装のための詳細調査結果とプロジェクト分析。

#### 調査項目系（実装上の課題整理）

- `調査項目_概要.md` - 全調査項目の総合概要と分類
- `調査項目_DSP信号処理.md` - CQT、SHS、最小位相応答等の実装詳細調査
- `調査項目_ニューラルネットワーク学習.md` - NANSY++構造、Dynamic batching 等の学習関連調査
- `調査項目_データセット前処理.md` - MIR-1K 形式、ノイズ付加手法等のデータ処理調査
- `調査項目_微分可能DSP損失関数.md` - ピッチシフト境界処理、GED 安定化等の損失関数調査

#### 調査結果系（優先度別実装指針）

- `調査結果_高優先.md` - 実装開始前必須の 4 項目（CQT 選択、NANSY++構造等）
- `調査結果_中優先.md` - Phase2-3 で必要な実装詳細調査結果
- `調査結果_低優先.md` - 最適化・発展的実装のための調査結果

#### 参考文献系（理論的背景）

- `参考文献1.md` - SLASH 基盤論文の詳細リストと実装必要性解説
- `参考文献2.md` - 補完的な関連研究論文リスト
- `参考文献総評.md` - 全参考文献の総合評価とレビュー

### データセット

- **LibriTTS-R**: 学習用。大規模音声データセット（585 時間）
- **MIR-1K**: 評価専用。

#### MIR-1K データセット詳細

**データセット場所**: `./train_dataset/mir1k/`

**基本情報**:

- **ファイル数**: 1,000 クリップ（110 楽曲から抽出）
- **総時間**: 133 分
- **クリップ長**: 4-13 秒
- **サンプリングレート**: 16kHz (論文では 24kHz にリサンプリング)
- **フォーマット**: ステレオ WAV（左チャンネル：楽器、右チャンネル：ボーカル）
- **言語**: 中国語ポップス
- **歌手**: 8 名の女性、11 名の男性（アマチュア）

**ディレクトリ構造**:

```
train_dataset/mir1k/MIR-1K/
├── Wavfile/                   # 1,000個のステレオ音声ファイル (.wav)
├── PitchLabel/                # 1,000個のピッチアノテーション (.pv)
├── UnvoicedFrameLabel/        # 1,000個の無声音フレームラベル
├── vocal-nonvocalLabel/       # 1,000個のV/UVセグメントラベル
├── Lyrics/                    # 歌詞テキストファイル (.txt)
├── UndividedWavfile/          # 元の楽曲全体ファイル（110曲）
└── LyricsWav/                 # 歌詞朗読音声ファイル
```

**評価で使用するファイル**:

1. **音声ファイル** (`Wavfile/*.wav`):

   - ステレオ 16bit WAV、左右分離済み
   - 左チャンネル: 楽器のみ（伴奏）
   - 右チャンネル: ボーカルのみ（歌声）
   - ファイル名形式: `{SingerId}_{SongId}_{ClipId}.wav`

2. **ピッチラベル** (`PitchLabel/*.pv`):
   - フレーム単位の F0 アノテーション（セミトーン）
   - フレーム長: 40ms、シフト: 20ms
   - 0Hz = 無声フレーム、>0 = 有声フレーム（セミトーン値）

**評価用サブセット**: `./train_dataset/mir1k/eval_250/`

- 論文評価で使用する 250 クリップセット（実際は 110 クリップ利用可能）
- 同一ディレクトリ構造で対応ファイルを格納
- 再現性確保のため seed=42 で固定選択

**データ処理時の注意**:

- **音声**: 16kHz → 24kHz リサンプリング必要
- **チャンネル分離**: `librosa.load(stereo=True)` で L/R 取得
  - `audio[0]`: 楽器トラック（評価時は使用しない）
  - `audio[1]`: ボーカルトラック（SLASH 評価対象）
- **ピッチラベル**: セミトーン → Hz 変換式: `f = 440 * 2^((semitone-69)/12)`
- **フレーム対応**: WAV(24kHz) ↔ ピッチラベル(50fps) の時間軸同期

## 主なコンポーネント

### 設定管理 (`src/unofficial_slash/config.py`)

### 学習システム (`scripts/train.py`)

- TensorBoard/W&B 統合
- torch.amp 対応
- スナップショット機能

### データ処理 (`src/unofficial_slash/dataset.py`)

- 遅延読み込み、dataclass 構造
- train/test/eval/valid
- CQT 変換、MIR-1K ピッチラベル読み込み

### ネットワーク (`src/unofficial_slash/network/predictor.py`)

### 損失計算 (`src/unofficial_slash/model.py`)

### DSP モジュール (`src/unofficial_slash/dsp/`)

- CQT Analyzer: 定 Q 変換による特徴抽出
- Pitch Guide Generator: SHS による事前分布計算
- Pseudo Spectrogram Generator: 微分可能スペクトログラム生成
- DDSP Synthesizer: 音声再合成
- V/UV Detector: 有声/無声判定

### 推論・生成システム

- `src/unofficial_slash/generator.py`: 推論ジェネレーター
- `scripts/generate.py`: 推論実行スクリプト

## 使用方法

<!-- TODO: 古いかも -->

### SLASH 学習実行 (現在のコマンドを SLASH 用設定で使用)

```bash
# LibriTTS-Rでの学習 (現在のconfig.yamlをSLASH用に変更後)
uv run -m scripts.train configs/slash_libritts.yaml output/slash_train

# MIR-1Kでの評価
uv run -m scripts.train configs/slash_mir1k_eval.yaml output/slash_eval
```

### SLASH 推論実行 (F0 推定結果出力)

```bash
# 現在のgenerate.pyをSLASH用に変更後使用
uv run -m scripts.generate --model_dir output/slash_train --output_dir output/f0_results [--use_gpu]
```

### データセットチェック

```bash
uv run -m scripts.check_dataset configs/slash_config.yaml [--trials 10]
```

### テスト実行

```bash
uv run pytest tests/ -sv
```

### 開発コマンド

```bash
# 環境セットアップ
uv sync

# 静的解析とフォーマット
uv run pyright && uv run ruff check --fix && uv run ruff format
```

## 技術仕様

### 設定ファイル

- **形式**: YAML
- **管理**: Pydantic による型安全な設定

### 主な依存関係

- **Python**: 3.12+
- **PyTorch**: 2.7.1+
- **NumPy**: 2.2.5+
- **Pydantic**: 2.11.7+
- **librosa**: 0.11.0+（音声処理）
- その他詳細は`pyproject.toml`を参照

### パッケージ管理

- **uv**による高速パッケージ管理
- **pyproject.toml**ベースの依存関係管理

## Docker 設計思想

このプロジェクトの Dockerfile は、実行環境の提供に特化した設計を採用しています：

- **環境のみ提供**: Dockerfile は依存関係とライブラリのインストールのみを行い、学習コードや推論コードは含みません
- **Git Clone 前提**: 実際の利用時は、コンテナ内で Git clone を実行してコードを取得することを想定しています
- **最新依存関係**: 参照プロジェクト（yukarin_sosoa、yukarin_sosfd、accent_estimator）に準拠し、最新の CUDA/PyTorch ベースイメージを使用
- **音声処理対応**: libsoundfile1-dev、libasound2-dev 等の音声処理ライブラリの整備方法をコメント等で案内
- **uv 使用**: pyproject.toml ベースの依存関係管理に uv を使用し、高速なパッケージインストールを実現

## フォーク時の拡張例

このフレームワークを拡張する際の参考：

1. **新しいネットワークアーキテクチャ**: `network/`ディレクトリに追加
2. **カスタム損失関数**: `model.py`の拡張
3. **異なるデータ形式**: データローダーの拡張

---

@docs/設計.md
@docs/コーディング規約.md
