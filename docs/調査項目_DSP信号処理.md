# 調査項目: DSP・信号処理関連

## 1. CQT (Constant-Q Transform) 実装の選択

### 質問内容
SLASH論文で使用されるCQTの実装において、librosa.cqt() vs torchaudio.transforms.CQT のどちらを選択すべきか？性能差や互換性の観点から最適解を知りたい。

### 背景
- 論文では CQT パラメータ: f_min=32.70Hz, 205bins, 24bins/octave, filter_scale=0.5
- フレームシフト 5ms、中央176bins使用
- PyTorch環境での効率的な実装が必要

### 調査ポイント
1. 計算速度の比較（CPU/GPU）
2. メモリ使用量の違い
3. 勾配計算の対応状況
4. パラメータ設定の互換性
5. バッチ処理対応の違い

### 参考資料
- librosa.cqt() ドキュメント
- torchaudio.transforms.CQT ドキュメント
- Brown 1991: "Calculation of a constant Q spectral transform"

---

## 2. Filter Scale 0.5 の効果と実装

### 質問内容
CQTパラメータの filter_scale=0.5 設定の具体的な効果と、librosa/torchaudio での正確な実装方法は？

### 背景
- 論文では「To accommodate rapid pitch changes, the filter scale was set to 0.5」
- 急速なピッチ変化に対応するための設定らしいが、具体的な効果が不明

### 調査ポイント
1. filter_scale の数学的定義
2. 0.5 設定による時間/周波数分解能への影響
3. ピッチ変化追従性への効果
4. librosa/torchaudio での実装差異

### 参考資料
- librosa CQT 実装ソースコード
- 音声信号処理におけるCQTの応用論文

---

## 3. Subharmonic Summation (SHS) アルゴリズム詳細

### 質問内容
Pitch Guide 生成で使用するSHSアルゴリズムの具体的な実装パラメータ（サブハーモニック次数、重み設定）は？

### 背景
- 論文 Equation (2): fine structure spectrum から SHS で pitch guide 生成
- Ikemiya et al. 2016 が引用されているが、具体的パラメータ不明

### 調査ポイント
1. SHS の数学的定義と計算手順
2. サブハーモニック次数の選択基準
3. 各次数の重み係数設定
4. 正規化手法の詳細
5. 計算効率化の手法

### 参考資料
- Ikemiya et al. 2016: "Singing voice separation and vocal F0 estimation based on mutual combination of robust principal component analysis and subharmonic summation"
- WORLD実装でのSHS使用例

---

## 4. Lag-window法によるスペクトル包絡計算

### 質問内容
Fine structure spectrum 計算（Equation 2）で使用されるLag-window法の具体的なパラメータ設定は？

### 背景
- ψ(S) = log(S) - W(log(S)) の W(·) がlag-window法
- スペクトル包絡除去による細構造抽出が目的

### 調査ポイント
1. Lag-window法の窓関数選択
2. 窓長の決定方法
3. WORLD実装での標準パラメータ
4. 音声/歌声での最適設定

### 参考資料
- Tohkura, Itakura & Hashimoto 1978: "Spectral smoothing technique in PARCOR speech analysis-synthesis"
- WORLD ボコーダー実装

---

## 5. 三角波振動子の実装式

### 質問内容
Pseudo Spectrogram Generation（Equation 4周辺）で使用される三角波振動子 X_t,k の正確な実装式は？

### 背景
- 論文の条件式: X_t,k = -1 if Φ_t,k < 0.5, 4|Φ_t,k - ⌊Φ_t,k⌋ - 0.5| - 1 otherwise
- 位相 Φ_t,k から三角波生成だが、境界条件が不明確

### 調査ポイント
1. 位相の周期性処理（0-1 正規化？）
2. 床関数適用のタイミング
3. 連続性の保証方法
4. 微分可能性の確認

### 参考資料
- DDSP論文での振動子実装例
- 数値音楽合成における三角波生成手法

---

## 6. 最小位相応答による時間領域合成

### 質問内容
DDSP Synthesizer での最小位相応答を用いた時間領域波形生成の具体的実装方法は？

### 背景
- 論文 footnote 3: "periodic and aperiodic component of w˜ is generated from ep and eap in time-domain using minimum-phase response"
- 周波数領域から時間領域への変換で最小位相が使用される

### 調査ポイント
1. scipy.signal.minimum_phase() の使用可否
2. STFT → 最小位相ISTFT の実装
3. 位相情報の処理方法
4. リアルタイム処理での効率化

### 参考資料
- DDSP論文 Section 3: Differentiable Components
- Differentiable WORLD Synthesizer 論文
- scipy.signal.minimum_phase ドキュメント