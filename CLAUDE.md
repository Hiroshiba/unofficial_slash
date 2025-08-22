# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このリポジトリは「SLASH: Self-Supervised Speech Pitch Estimation Leveraging DSP-derived Absolute Pitch」論文をなるべく忠実に実装することが目的です。SLASH は自己教師あり学習（SSL）と従来のデジタル信号処理（DSP）を組み合わせて、音声の基本周波数（F0）推定を行う手法です。

### 優先度の高い残存タスク

- cqt_hop_lengthとpseudo_spec_hop_lengthをなくしてframe_rateにする

- 部分部分でtorch compileしていく
  - 全体でコンパイルするのはCQTがあるため不可能

- 許容するフレーム数の量と判定ロジックを共通化したい
  - 便利関数を使っていたり使っていなかったりする、可能なら使う方に統一したいのと、２種以上のtensorを受け取れるようにしたい
  - 最大で2フレームまで許容する方に倒したい
  - TODOコメントとして許容できるフレーム数の誤差を減らしたいと書き加える

### 主要な特徴
- 相対的なピッチ差学習（ピッチシフト利用）
- DSP 由来の絶対ピッチ情報を活用
- 微分可能 DSP（DDSP）による F0 最適化
- 非周期性（aperiodicity）と有声/無声（V/UV）の同時推定
- ノイズロバスト性の向上

## アーキテクチャ構成

### コアファイル構造
- `train.py`: メインの学習ループ、損失計算、勾配更新
- `network.py` または `network/`: ニューラルネットワーク構造
  - Pitch Encoder: CQT から F0 確率分布と BAP を出力
- `dataset.py`: データローダー（LibriTTS-R, MIR-1K 対応）
- `config.py`: argparse による設定管理

### モジュール設計
1. **Pitch Encoder** (DNN): NANSY++ ベースの改変版
   - 入力: CQT (T×176 bins)
   - 出力: F0 確率分布 P (T×1024) + Band Aperiodicity B (T×8)

2. **DSP モジュール群**:
   - CQT Analyzer: 定 Q 変換による特徴抽出
   - Pitch Guide Generator: SHS による事前分布計算
   - Spec Env. Estimator: スペクトル包絡推定
   - Pseudo Spec. Generator: 微分可能スペクトログラム生成
   - Differentiable WORLD Synthesizer: 音声再合成
   - V/UV Detector: 有声/無声判定

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
### ノイズロバスト損失（Section 2.6準拠）
6. **L_aug**: 拡張データでの基本損失（Huber loss between p and p_aug）
   ```
   L_aug = ||p - p_aug||₁
   ```
7. **L_g-aug**: 拡張データでのPitch Guide損失（P_augを使用したL_gと同等）
   ```
   L_g-aug = (1/T) Σ max(1 - Σ P_aug_t,f × G_t,f - m, 0)
   ```
8. **L_ap**: Aperiodicity一貫性損失（`||log(A_aug) - 
   og(A)||₁`）
79    
**実装状況**: SLASH論文準拠の全8種類損失を完全実装済み


## 実装上の重要ポイント

### DSP パイプライン
1. **CQT 設定**: frame_shift=5ms, f_min=32.70Hz, 205bins, 24bins/octave, filter_scale=0.5
2. **Pitch Guide**: fine structure spectrum → SHS → 正規化
3. **Pseudo Spectrogram**: 三角波振動子 → 周期性励起スペクトログラム
4. **Differentiable WORLD 合成**: 時間領域での周期・非周期成分合成→最小位相応答処理→音声再構成（SLASH論文Footnote 3準拠）。

### 学習設定
- Optimizer: AdamW (lr=0.0002)
- Dynamic batching (平均バッチサイズ17)
- 学習ステップ: 100,000
- 損失重み: L_cons, L_pseudo=10.0, L_recon=5.0, L_guide, L_g-shift, その他=1.0

### 曖昧な実装部分（要 FIXME コメント）
- SHS アルゴリズムの詳細パラメータ
- GED 損失の安定化手法
- Dynamic batching の実装詳細


## 設定管理（Pydantic + YAML）

**現在の設定システム**: Pydantic BaseModel + YAML 設定ファイル

SLASH用の設定構造については以下を参照：
- **設定クラス定義**: `src/unofficial_slash/config.py`
- **実際の設定例**: `tests/input_data/base_config.yaml`

現在の汎用ML設定をSLASH専用（CQT、F0推定、SLASH損失関数）に破壊的変更予定。

## 参考論文（実装必須）

### 基礎理論
- **SPICE**: SSL による相対ピッチ学習の基礎
- **NANSY++**: Pitch Encoder アーキテクチャ
- **DDSP**: 微分可能信号処理の概念
- **WORLD**: 音声分析合成システム（F0, spectral envelope, aperiodicity）

### 信号処理
- **CQT**: Constant-Q Transform の計算
- **SHS**: Subharmonic Summation による pitch guide
- **GED**: Generalized Energy Distance for aperiodicity optimization

## 書類・文献整理

このセクションでは、プロジェクト内の論文・調査資料の構成を整理します。

### paper/ディレクトリ（関連論文集）
SLASH実装に必要な論文群。メイン論文と関連研究をテキスト形式で収録。

- `SLASH: Self-Supervised Speech Pitch Estimation Leveraging DSP-derived Absolute Pitch.txt` - **メイン論文**
- `SPICE: Self-supervised Pitch Estimation.txt` - SSL相対ピッチ学習の基礎
- `NANSY++: UNIFIED VOICE SYNTHESIS WITH NEURAL ANALYSIS AND SYNTHESIS.txt` - Pitch Encoderアーキテクチャ  
- `DDSP: DIFFERENTIABLE DIGITAL SIGNAL PROCESSING.txt` - 微分可能DSPの概念
- `PESTO: PITCH ESTIMATION WITH SELF-SUPERVISED TRANSPOSITION-EQUIVARIANT OBJECTIVE.txt` - SSL比較手法
- `A Spectral Energy Distance for Parallel Speech Synthesis.txt` - GED損失の原典
- `DIFFERENTIABLE WORLD SYNTHESIZER-BASED NEURAL VOCODER WITH APPLICATION TO END-TO-END AUDIO STYLE TRANSFER.txt` - DDSP Synthesizer参考
- `ESPNET-TTS: UNIFIED, REPRODUCIBLE, AND INTEGRATABLE OPEN SOURCE END-TO-END TEXT-TO-SPEECH TOOLKIT.txt` - Dynamic batching参考
- `Singing Voice Separation and Vocal F0 Estimation based on Mutual Combination of Robust Principal Component Analysis and Subharmonic Summation.txt` - SHSアルゴリズム

### docs/ディレクトリ（調査・分析資料）
SLASH実装のための詳細調査結果とプロジェクト分析。

#### 調査項目系（実装上の課題整理）
- `調査項目_概要.md` - 全調査項目の総合概要と分類
- `調査項目_DSP信号処理.md` - CQT、SHS、最小位相応答等の実装詳細調査
- `調査項目_ニューラルネットワーク学習.md` - NANSY++構造、Dynamic batching等の学習関連調査  
- `調査項目_データセット前処理.md` - MIR-1K形式、ノイズ付加手法等のデータ処理調査
- `調査項目_微分可能DSP損失関数.md` - ピッチシフト境界処理、GED安定化等の損失関数調査

#### 調査結果系（優先度別実装指針）  
- `調査結果_高優先.md` - 実装開始前必須の4項目（CQT選択、NANSY++構造等）
- `調査結果_中優先.md` - Phase2-3で必要な実装詳細調査結果
- `調査結果_低優先.md` - 最適化・発展的実装のための調査結果

#### 参考文献系（理論的背景）
- `参考文献1.md` - SLASH基盤論文の詳細リストと実装必要性解説
- `参考文献2.md` - 補完的な関連研究論文リスト  
- `参考文献総評.md` - 全参考文献の総合評価とレビュー

### データセット
- **LibriTTS-R**: 大規模音声データセット（585時間）**← 学習用**
- **MIR-1K**: 歌声データセット **← 評価専用**

#### MIR-1K データセット詳細（評価専用）

**データセット場所**: `./train_dataset/mir1k/` ※評価専用（学習には使用しない）

**基本情報**:
- **ファイル数**: 1,000クリップ（110楽曲から抽出）
- **総時間**: 133分
- **クリップ長**: 4-13秒
- **サンプリングレート**: 16kHz (論文では24kHzにリサンプリング)
- **フォーマット**: ステレオWAV（左チャンネル：楽器、右チャンネル：ボーカル）
- **言語**: 中国語ポップス
- **歌手**: 8名の女性、11名の男性（アマチュア）

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
   - ステレオ16bit WAV、左右分離済み
   - 左チャンネル: 楽器のみ（伴奏）
   - 右チャンネル: ボーカルのみ（歌声）
   - ファイル名形式: `{SingerId}_{SongId}_{ClipId}.wav`

2. **ピッチラベル** (`PitchLabel/*.pv`):
   - フレーム単位のF0アノテーション（セミトーン）
   - フレーム長: 40ms、シフト: 20ms
   - 0Hz = 無声フレーム、>0 = 有声フレーム（セミトーン値）

3. **無声音ラベル** (`UnvoicedFrameLabel/*`):
   - 5種類の音素分類:
     - Unvoiced stop (無声閉鎖音)
     - Unvoiced fricative/affricate (無声摩擦音/破擦音)
     - /h/ (無声咽頭摩擦音)
     - Inhaling sound (吸気音)
     - Others (その他：有声音、楽器音)

4. **V/UVラベル** (`vocal-nonvocalLabel/*`):
   - ボーカルセグメント vs 非ボーカルセグメント
   - 論文のV/UV検出評価に使用

**評価用サブセット**: `./train_dataset/mir1k/eval_250/`
- 論文評価で使用する250クリップセット（実際は110クリップ利用可能）
- 同一ディレクトリ構造で対応ファイルを格納
- 再現性確保のためseed=42で固定選択

**データ処理時の注意**:
- **音声**: 16kHz → 24kHzリサンプリング必要
- **チャンネル分離**: `librosa.load(stereo=True)` でL/R取得
  - `audio[0]`: 楽器トラック（評価時は使用しない）
  - `audio[1]`: ボーカルトラック（SLASH評価対象）
- **ピッチラベル**: セミトーン → Hz変換式: `f = 440 * 2^((semitone-69)/12)`
- **フレーム対応**: WAV(24kHz) ↔ ピッチラベル(50fps) の時間軸同期

## 開発・評価コマンド

### データセットダウンロード

#### MIR-1K（評価専用データセット）
```bash
# フルデータセット（1,000クリップ）をダウンロード
python scripts/download_mir1k_test.py

# 評価用サブセット（250クリップ相当）のみ作成
python scripts/download_mir1k_test.py --partial

# 既存ファイルをスキップして実行
python scripts/download_mir1k_test.py --skip-existing
```
- ダウンロード先: `./train_dataset/mir1k/`
- データ容量: 約760MB
- 自動展開・検証機能付き

#### LibriTTS-R（学習専用データセット）
```bash
# 学習用全セット（デフォルト、約84GB）- SLASH論文で使用
python scripts/download_libritts_train.py

# 16並列で高速ダウンロード（推奨）
python scripts/download_libritts_train.py --parallel 16

# テスト用小さなセット（約4GB）
python scripts/download_libritts_train.py --test-only --parallel 8

# 特定サブセットのみダウンロード
python scripts/download_libritts_train.py --subsets dev_clean --parallel 4

# 既存ファイルをスキップして再開
python scripts/download_libritts_train.py --skip-existing --parallel 16
```
- ダウンロード先: `./train_dataset/libritts-r/`
- データ容量: 約84GB（学習用3セット）、約87GB（全セット）
- 用途: SLASH モデルの学習のみ
- **特徴**: 16並列ダウンロード対応、複数ミラーサーバー自動切り替え、レジューム機能

### パスリスト作成（学習前必須）

#### SLASH学習用パスリスト生成
```bash
# LibriTTS-R dev_cleanサブセットでのパスリスト作成
uv run scripts/create_pathlist.py --dataset libritts-r --subset dev_clean

# LibriTTS-R test_cleanサブセットでのパスリスト作成
uv run scripts/create_pathlist.py --dataset libritts-r --subset test_clean

# MIR-1K フルセットでのパスリスト作成  
uv run scripts/create_pathlist.py --dataset mir1k

# MIR-1K 評価サブセットでのパスリスト作成
uv run scripts/create_pathlist.py --dataset mir1k --subset eval_250
```

**対応データセット**:
- **LibriTTS-R**: dev_clean, test_clean, train_clean_100, train_clean_360, train_other_500等
- **MIR-1K**: full（1,000クリップ）, eval_250（250クリップ相当のサブセット）

**生成ファイル**:
- `{dataset}_{subset}_audio_pathlist.txt`: 音声ファイルパス一覧
- `{dataset}_{subset}_pitch_label_pathlist.txt`: ピッチラベルファイルパス一覧（MIR-1K時のみ）

**特徴**:
- ファイル整合性チェック機能付き（音声とピッチラベルの対応確認）
- 相対パス形式での出力（train_dataset/からの相対パス）
- 自動的な出力ディレクトリ作成（train_dataset/）
- データセット構造の自動認識・検証機能

### 学習実行（LibriTTS-R使用）
```bash
python train.py --dataset libritts-r --data_root ./train_dataset/libritts-r --max_steps 100000
```

### 評価実行（MIR-1K使用）
```bash
python evaluate.py --model_path checkpoints/best.pth --test_data mir-1k --data_root ./train_dataset/mir1k
```

### 評価指標
- RPA (Raw Pitch Accuracy): 50/100 cents 以内の精度
- RCA (Raw Chroma Accuracy): オクターブエラー許容精度  
- log-F0 RMSE: 対数 F0 の RMSE
- V/UV Error Rate: 有声/無声分類エラー率

## 🚨 **現状の技術課題・未解決問題**

- ⚠️ **TorchScript無効化問題**: nnAudio/CQTとtorch.jit.scriptの互換性問題
  - 一時的にTorchScript化を無効化（train.py:155-157）
  - 推論速度・デプロイ時の最適化に影響する可能性
  - nnAudioの代替CQT実装またはTorchScript対応版への移行検討が必要

### 実装時の重要な注意点

1. **論文式番号の明記**: 各実装箇所に対応する論文の式番号をコメントで記載
2. **FIXME コメント**: 曖昧な部分は必ず `# FIXME:` で明記
3. **単体テスト**: 各モジュールの単体テスト作成
4. **可視化機能**: デバッグ用の中間結果可視化機能実装
5. **実験再現性**: Random seed 固定とログ詳細化

## 注意事項

- **コード内コメントは全て日本語で記述すること**
- 全ての曖昧な実装部分には `# FIXME:` コメントを必ず追加
- DSP モジュールは numpy/scipy ベース、学習部分は PyTorch
- 実験の再現性確保のため random seed 固定
- メモリ効率を考慮した動的バッチング実装
- GPU/CPU 両対応の実装
- 論文の式番号と実装の対応を明記（コメント内）


## 主なコンポーネント (**現在**: 汎用MLフレームワーク → **SLASH専用に破壊的変更予定**)

### 設定管理 (`src/unofficial_slash/config.py`) 
**現在**: 汎用ML設定 → **SLASH特化設定に変更予定**
- `DataFileConfig`: pathlist管理 → SLASH音声データ用
- `NetworkConfig`: 汎用NN設定 → Pitch Encoder + CQT設定
- `ModelConfig`: 汎用モデル設定 → SLASH損失関数設定
- `DatasetConfig`, `TrainConfig`, `ProjectConfig`: 継続使用

### 学習システム (`scripts/train.py`)
**継続使用**: PyTorchエポックベース学習ループ
- TensorBoard/W&B統合
- torch.amp対応
- スナップショット機能

### データ処理 (`src/unofficial_slash/dataset.py`)
**部分変更予定**: pathlistシステムをSLASH用に適応
- 遅延読み込み、dataclass構造: 継続使用
- train/test/eval/valid: 継続使用
- **変更予定**: 話者マッピング削除、音声+ピッチラベル対応
- **新規実装**: CQT変換、MIR-1Kピッチラベル読み込み

### ネットワーク (`src/unofficial_slash/network/predictor.py`)
**✅ 完了**: SLASH Pitch Encoder実装完了
- **実装済み**: NANSY++ベース Pitch Encoder
  - 入力: CQT (T×176)
  - 出力: F0確率分布 P (T×1024) + Band Aperiodicity B (T×8)
- **新規ファイル**: `src/unofficial_slash/network/nansy.py`
  - FrequencyResBlock: 周波数軸Residual Block
  - NansyPitchEncoder: CNN+GRUアーキテクチャ

### 損失計算 (`src/unofficial_slash/model.py`)
**破壊的変更予定**: 汎用損失 → SLASH特化損失関数
- **現在**: cross_entropy + mse_loss
- **変更後**: L_cons + L_guide + L_pseudo + L_recon (GED)

### DSPモジュール (`src/unofficial_slash/dsp/`) 
**新規実装予定**:
- CQT Analyzer: 定 Q 変換による特徴抽出
- Pitch Guide Generator: SHS による事前分布計算
- Pseudo Spectrogram Generator: 微分可能スペクトログラム生成
- DDSP Synthesizer: 音声再合成
- V/UV Detector: 有声/無声判定

### 推論・生成システム
**継続使用**: 現在のフレームワーク構造
- `src/unofficial_slash/generator.py`: 推論ジェネレーター  
- `scripts/generate.py`: 推論実行スクリプト
- **変更予定**: F0推定結果出力に特化

### テストシステム
**継続使用**: 現在のテストシステム
- 自動テストデータ生成  
- **変更予定**: SLASH用テストケース追加

## 使用方法 (**継続使用**: 現在のコマンド構造)

### SLASH学習実行 (現在のコマンドをSLASH用設定で使用)
```bash
# LibriTTS-Rでの学習 (現在のconfig.yamlをSLASH用に変更後)
uv run -m scripts.train configs/slash_libritts.yaml output/slash_train

# MIR-1Kでの評価
uv run -m scripts.train configs/slash_mir1k_eval.yaml output/slash_eval
```

### SLASH推論実行 (F0推定結果出力)
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
- **管理**: Pydanticによる型安全な設定

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

## Docker設計思想

このプロジェクトのDockerfileは、実行環境の提供に特化した設計を採用しています：

- **環境のみ提供**: Dockerfileは依存関係とライブラリのインストールのみを行い、学習コードや推論コードは含みません
- **Git Clone前提**: 実際の利用時は、コンテナ内でGit cloneを実行してコードを取得することを想定しています
- **最新依存関係**: 参照プロジェクト（yukarin_sosoa、yukarin_sosfd、accent_estimator）に準拠し、最新のCUDA/PyTorchベースイメージを使用
- **音声処理対応**: libsoundfile1-dev、libasound2-dev等の音声処理ライブラリの整備方法をコメント等で案内
- **uv使用**: pyproject.tomlベースの依存関係管理にuvを使用し、高速なパッケージインストールを実現

## フォーク時の拡張例

このフレームワークを拡張する際の参考：

1. **新しいネットワークアーキテクチャ**: `network/`ディレクトリに追加
2. **カスタム損失関数**: `model.py`の拡張
3. **異なるデータ形式**: データローダーの拡張

---

@docs/設計.md
@docs/コーディング規約.md
