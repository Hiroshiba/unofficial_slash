# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このリポジトリは「SLASH: Self-Supervised Speech Pitch Estimation Leveraging DSP-derived Absolute Pitch」論文の実装です。SLASH は自己教師あり学習（SSL）と従来のデジタル信号処理（DSP）を組み合わせて、音声の基本周波数（F0）推定を行う手法です。

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
   - DDSP Synthesizer: 音声再合成
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

3. **Pseudo Spectrogram Loss (L_pseudo)**: F0 勾配最適化
   ```
   L_pseudo = ||ψ(S*) - ψ(S)||₁ ⊙ (v × 1_K)
   ```

4. **Reconstruction Loss (L_recon)**: GED による aperiodicity 最適化
   ```
   L_recon = ||ψ(Š¹) – ψ(S)||₁ – α ||ψ(Š¹) – ψ(Š²)||₁
   ```

### 補助損失
- L_aug, L_g-aug, L_ap: ノイズロバスト性向上

## 実装上の重要ポイント

### DSP パイプライン
1. **CQT 設定**: frame_shift=5ms, f_min=32.70Hz, 205bins, 24bins/octave, filter_scale=0.5
2. **Pitch Guide**: fine structure spectrum → SHS → 正規化
3. **Pseudo Spectrogram**: 三角波振動子 → 周期性励起スペクトログラム
4. **DDSP 合成**: 時間領域での最小位相応答による periodic/aperiodic 成分生成

### 学習設定
- Optimizer: AdamW (lr=0.0002)
- Dynamic batching (平均バッチサイズ17)
- 学習ステップ: 100,000
- 損失重み: L_cons, L_pseudo=10.0, L_recon=5.0, その他=1.0

### 曖昧な実装部分（要 FIXME コメント）
- Triangle wave oscillator の具体的実装
- SHS アルゴリズムの詳細パラメータ
- GED 損失の安定化手法
- Dynamic batching の実装詳細
- 時間領域波形生成の最小位相応答計算

## 設定管理（argparse）

必須パラメータの初期値:
```python
# Data
--data_root: データセットルートパス
--dataset: "libritts-r"（学習用） or "mir-1k"（評価用）
--sample_rate: 24000

# Model  
--f_bins: 1024  # F0 probability bins
--bap_dim: 8    # Band aperiodicity dimension
--cqt_bins: 205 # CQT frequency bins
--cqt_hop: 120  # 5ms at 24kHz

# Training
--batch_size: 17
--learning_rate: 0.0002
--max_steps: 100000
--shift_range: 14  # ±14 bins for pitch shift

# Loss weights
--w_cons: 10.0
--w_pseudo: 10.0  
--w_recon: 5.0
--w_guide: 1.0

# DSP parameters
--hinge_margin: 0.5    # m in pitch guide loss
--ged_alpha: 0.1       # α in GED loss
--vuv_threshold: 0.5   # θ for V/UV detection
--epsilon: 0.001       # ε for pseudo spec generation
```

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

## 実装優先順位

1. **Phase 1**: 基本構造構築
   - データローダー（dataset.py）
   - Pitch Encoder（network.py）
   - CQT Analyzer, DSP モジュール骨格

2. **Phase 2**: 相対ピッチ学習
   - L_cons 損失実装
   - ピッチシフト処理
   - 基本学習ループ（train.py）

3. **Phase 3**: 絶対ピッチ統合
   - Pitch Guide Generator (SHS)
   - L_g 損失実装
   - Pseudo Spectrogram Generator
   - L_pseudo 損失実装

4. **Phase 4**: 完全統合
   - DDSP Synthesizer
   - L_recon (GED) 損失
   - V/UV Detector
   - ノイズロバスト化

## 詳細実装計画

### 現在の状況
- ✅ 基本的なプロジェクト構造の理解完了
- ✅ 論文解読と参考文献の整理完了
- ✅ アーキテクチャ設計の概要決定
- ✅ **requirements.txt作成・依存関係管理完了**
  - torch, torchaudio, numpy, scipy, librosa
  - aiohttp, aiofiles, tqdm（データセットダウンロード用）
  - 全依存関係のインストール・動作確認済み
- ✅ **MIR-1Kデータセットダウンロード・セットアップ完了**
  - 1,000クリップ + アノテーション完全取得
  - 評価用250クリップサブセット準備完了
  - データ構造解析・検証済み
- ✅ **LibriTTS-Rダウンロードスクリプト作成・動作確認完了**
  - 16並列ダウンロード対応、複数ミラーサーバー活用
  - dev_cleanサブセット（5,736ファイル）ダウンロード動作確認済み
  - レジューム機能、自動展開・検証機能付き
  - デフォルトで学習用全セット（約84GB）をダウンロード
- ⬜ 実装Phase 1開始（基本構造構築）

### Phase 1: 基本構造構築（推定期間: 3-5日）

#### 1.1 プロジェクト骨格の作成
**実装対象**: 基本ディレクトリ構造、依存関係管理
**参考資料**: なし（標準的な Python プロジェクト構造）
**実装手順**:
1. ✅ `requirements.txt` 作成完了（torch, torchaudio, numpy, scipy, librosa + データセットダウンロード用ライブラリ）
2. ✅ `scripts/` ディレクトリ作成済み（ダウンロードスクリプト配置済み）
3. ⬜ `src/` ディレクトリ作成
4. ⬜ `tests/`, `configs/` ディレクトリ作成
5. ⬜ `__init__.py` ファイル配置

**曖昧な部分**: 
- librosa vs torchaudio での CQT 実装選択 (FIXME: 性能比較必要)

#### 1.2 設定管理システム（config.py）
**実装対象**: argparse ベースの設定管理
**参考資料**: 論文 Table 1 (Model details)
**実装手順**:
1. `Config` クラス作成
2. argparse による全パラメータ定義
3. 設定ファイル（YAML/JSON）読み込み機能
4. 実験管理用の設定保存機能

**曖昧な部分**:
- Dynamic batching のバッチサイズ決定ロジック (FIXME: ESPnet-TTS[27]参照)

#### 1.3 データローダー（dataset.py）
**実装対象**: LibriTTS-R（学習用）, MIR-1K（評価用）データセット対応
**参考資料**: 
- LibriTTS-R 論文 (データフォーマット) ← 学習用
- MIR-1K 論文 (アノテーション形式) ← 評価用
**実装手順**:
1. `BaseDataset` 抽象クラス作成
2. `LibriTTSDataset` クラス実装（学習専用）
3. `MIR1KDataset` クラス実装（評価専用）  
4. 音声ファイル読み込み・リサンプリング処理
5. データ拡張（ノイズ付加、音量変更）実装（学習時のみ）

**曖昧な部分**:
- ✅ **MIR-1K アノテーションの読み込み形式確認済み**:
  - PitchLabel/*.pv: セミトーン値、0=無声
  - UnvoicedFrameLabel/*: 5分類の音素ラベル
  - vocal-nonvocalLabel/*: V/UVセグメントラベル
  - フレーム: 40ms窓、20msシフト
- ノイズ付加の具体的手法（SNR -6dB の実装）(FIXME: 論文3.1節参照)

#### 1.4 CQT Analyzer 基本実装
**実装対象**: Constant-Q Transform による特徴抽出
**参考資料**: 
- Brown 1991: CQT 原論文
- librosa.cqt() ドキュメント
**実装手順**:
1. CQT パラメータ設定（f_min=32.70Hz, 205bins, 24bins/octave）
2. フレームシフト 5ms の実装
3. フィルタースケール 0.5 の適用
4. 中央 176 bins の抽出処理

**曖昧な部分**:
- filter_scale=0.5 の具体的な効果と実装方法 (FIXME: librosa実装確認)

### Phase 2: 相対ピッチ学習（推定期間: 4-6日）

#### 2.1 Pitch Encoder アーキテクチャ
**実装対象**: NANSY++ ベースの Pitch Encoder
**参考資料**: 
- NANSY++ 論文 Section 3.2 (Pitch Encoder)
- 論文 docs/参考文献1.md の詳細
**実装手順**:
1. `PitchEncoder` クラス作成
2. CQT (T×176) → F0確率分布 P (T×1024) 変換
3. Band Aperiodicity B (T×8) 出力追加
4. 重み付き平均による F0 値 p 計算
5. 線形補間による Aperiodicity A (T×K) 計算

**曖昧な部分**:
- NANSY++ の具体的なネットワーク構造（層数、チャンネル数）(FIXME: 原論文詳細確認)
- 対数周波数スケール（20Hz-2kHz）への正確なマッピング (FIXME: 式化必要)

#### 2.2 ピッチシフト処理
**実装対象**: CQT 空間でのピッチシフト
**参考資料**: 
- SPICE 論文 Section 2.1
- PESTO 論文 Section 3.1
**実装手順**:
1. CQT の周波数軸シフト処理（±14 bins）
2. シフト量 d (semitones) の記録
3. バッチ内でのランダムシフト適用
4. シフト後 CQT (C_shift) の生成

**曖昧な部分**:
- 端の周波数ビンでの境界処理方法 (FIXME: ゼロパディングか循環シフトか)

#### 2.3 Pitch Consistency Loss (L_cons)
**実装対象**: 相対ピッチ差損失
**参考資料**: 論文 Equation (1)
**実装手順**:
1. Huber loss 実装
2. 対数ピッチ差計算（log2(p_t) - log2(p_shift_t)）
3. semitone → octave 変換（/12）
4. フレーム平均計算

**曖昧な部分**:
- Huber loss のデルタパラメータ値 (FIXME: 論文に明記なし、実験で調整)

#### 2.4 基本学習ループ
**実装対象**: train.py の基本構造
**参考資料**: 一般的な PyTorch 学習ループ
**実装手順**:
1. モデル、オプティマイザ、データローダ初期化
2. 学習ループ（100,000 steps）
3. 損失計算・逆伝播・更新
4. ログ出力・チェックポイント保存
5. 検証データでの評価

**曖昧な部分**:
- Dynamic batching の具体的実装 (FIXME: 平均バッチサイズ17の制御方法)

### Phase 3: 絶対ピッチ統合（推定期間: 5-7日）

#### 3.1 Pitch Guide Generator (SHS)
**実装対象**: Subharmonic Summation による事前分布
**参考資料**: 
- Ikemiya et al. 2016 (SHS 原論文)
- 論文 Equation (2) (Fine structure spectrum)
**実装手順**:
1. Fine structure spectrum ψ(S) 計算
2. Lag-window法によるスペクトル包絡計算
3. SHS アルゴリズム実装
4. 正規化処理（各フレーム最大値=1）

**曖昧な部分**: 
- SHS のサブハーモニック次数と重み設定 (FIXME: 原論文の実装詳細確認)
- Lag-window法の具体的パラメータ (FIXME: WORLD実装参照)

#### 3.2 Pitch Guide Loss (L_g)
**実装対象**: 絶対ピッチ事前分布損失
**参考資料**: 論文 Equation (3)
**実装手順**:
1. Pitch probability P と Pitch guide G の内積計算
2. Hinge loss with margin m=0.5
3. シフト版 L_g-shift の実装
4. フレーム平均化

**曖昧な部分**: なし（論文に明確に記載）

#### 3.3 Pseudo Spectrogram Generator
**実装対象**: 微分可能スペクトログラム生成
**参考資料**: 
- 論文 Section 2.4 (Equations 4-6)
- DDSP 論文 (微分可能DSPの概念)
**実装手順**:
1. 位相計算 Φ_t,k = (f_s/2p_tK)k
2. 三角波振動子 X_t,k の生成
3. Pseudo periodic excitation E*_p 計算
4. ノイズ項追加（ε=0.001）

**曖昧な部分**:
- 三角波振動子の正確な実装式 (FIXME: 論文の条件式が不明確)
- ガウシアンノイズ Z の具体的な適用方法 (FIXME: 実装で確認)

#### 3.4 Pseudo Spectrogram Loss (L_pseudo)  
**実装対象**: F0 勾配最適化損失
**参考資料**: 論文 Equation (6)
**実装手順**:
1. Fine structure spectrum ψ の計算
2. V/UV マスク v の適用
3. L1 norm 計算
4. 計算グラフの F0 以外のdetach処理

**曖昧な部分**: なし（論文に明確に記載）

### Phase 4: 完全統合（推定期間: 6-8日）

#### 4.1 DDSP Synthesizer
**実装対象**: 微分可能音声合成器
**参考資料**: 
- DDSP 論文 Section 3
- Differentiable WORLD Synthesizer 論文
**実装手順**:
1. 周期励起信号 e_p 生成（複数サイン波合成）
2. 非周期励起信号 e_ap 処理
3. 最小位相応答による時間領域合成
4. STFT による周波数領域変換

**曖昧な部分**:
- 最小位相応答の具体的実装方法 (FIXME: scipy.signal.minimum_phase使用検討)
- ハーモニクス次数の決定方法 (FIXME: 論文に明記なし)

#### 4.2 Reconstruction Loss (L_recon) - GED
**実装対象**: Generalized Energy Distance 損失
**参考資料**: 
- Gritsenko et al. 2020 (GED 原論文)
- 論文 Equation (8)
**実装手順**:
1. 2つの生成スペクトログラム S~1, S~2 作成
2. Fine structure spectrum 計算
3. L1 norm による距離計算
4. Repulsive term (α=0.1) の実装

**曖昧な部分**:
- S~1, S~2 の具体的な生成方法の違い (FIXME: ランダムシード？パラメータ摂動？)

#### 4.3 V/UV Detector
**実装対象**: 有声/無声判定器
**参考資料**: 論文 Equation (9)
**実装手順**:
1. 周期成分マグニチュード M_p 計算
2. 非周期成分マグニチュード M_ap 計算
3. 比率計算 v' = M_p/(M_p + M_ap)
4. 閾値処理 (θ=0.5)

**曖昧な部分**: なし（論文に明確に記載）

#### 4.4 ノイズロバスト化
**実装対象**: データ拡張と追加損失
**参考資料**: 論文 Section 2.6
**実装手順**:
1. ノイズ付加処理（最大SNR -6dB）
2. 音量変更処理（±6dB）
3. 拡張損失 L_aug, L_g-aug, L_ap 実装
4. 全損失の重み付き合成

**曖昧な部分**:
- ノイズの種類（白色ノイズ以外も考慮？）(FIXME: 論文では白色ノイズのみ言及)

### 実装検証計画

#### Phase 1 検証
- [ ] CQT 出力形状確認（T×176）
- [x] **MIR-1Kデータローダーの基本仕様確認**（ファイル形式、アノテーション構造）
- [ ] データローダーの実装・動作確認
- [ ] 設定管理の完全性確認

#### Phase 2 検証  
- [ ] ピッチシフト処理の正確性確認
- [ ] L_cons 損失の数値計算確認
- [ ] 基本学習ループの動作確認

#### Phase 3 検証
- [ ] Pitch Guide の可視化確認
- [ ] SHS アルゴリズムの動作確認
- [ ] Pseudo Spectrogram の生成確認

#### Phase 4 検証
- [ ] DDSP Synthesizer の音声品質確認
- [ ] 全損失項の統合動作確認
- [ ] MIR-1K での評価指標計算

### 実装時の重要な注意点

1. **論文式番号の明記**: 各実装箇所に対応する論文の式番号をコメントで記載
2. **FIXME コメント**: 曖昧な部分は必ず `# FIXME:` で明記
3. **単体テスト**: 各モジュールの単体テスト作成
4. **可視化機能**: デバッグ用の中間結果可視化機能実装
5. **実験再現性**: Random seed 固定とログ詳細化

## 注意事項

- 全ての曖昧な実装部分には `# FIXME:` コメントを必ず追加
- DSP モジュールは numpy/scipy ベース、学習部分は PyTorch
- 実験の再現性確保のため random seed 固定
- メモリ効率を考慮した動的バッチング実装
- GPU/CPU 両対応の実装
- 論文の式番号と実装の対応を明記（コメント内）