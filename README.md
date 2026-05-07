# kaggle_ariel

`ariel_v1.py` から `ariel_v5.py` までをまとめた、Kaggle Ariel Data Challenge 2025 向けの推論スクリプト集です。  
全バージョンとも、AIRS-CH0 / FGS1 の生信号を前処理し、1D のトランジット深さを推定し、恒星メタ情報と組み合わせて最終的な `submission.csv` を生成します。

## このリポジトリの位置づけ

- 学習済みモデルを使った推論用コードです。
- 学習コードや重み生成コードは含まれていません。
- 実行環境は Kaggle のオフライン Notebook / Script を前提にしています。
- コード中のパスは Kaggle 入力ディレクトリに強く依存しています。

## 各バージョンの概要

| ファイル | 位置づけ | 主な特徴 |
| --- | --- | --- |
| `ariel_v1.py` | 初版ベースライン | 生データ前処理、1Dトランジット深さ推定、FGS/AIRS の学習済み MLP 推論、`submission.csv` 生成。Notebook 断片に近い構成。 |
| `ariel_v2.py` | ベースライン改善版 | `flat-field` 補正を追加。AIRS の固定スムージングとブレンドを導入。`__main__` 付きで実行しやすく整理。 |
| `ariel_v3.py` | 自動平滑化導入版 | AIRS に対する適応的スムージング、winsorize、FGS と 1D 深さの軽い自己キャリブレーションを追加。 |
| `ariel_v4.py` | 安定化・整合性強化版 | AIRS 予測の中央値方向への縮約、AIRS/FGS 比の整合、`sigma` の収縮を追加して提出の安定性を強化。 |
| `ariel_v5.py` | 最終発展版 | OOT ベースの AIRS 重み付け、MC Dropout による test-time ensemble、不確かさ反映、任意の Tikhonov 平滑化を追加。 |

現時点で使うなら、基本的には `ariel_v5.py` が最も機能が多く、`ariel_v3.py` または `ariel_v4.py` が比較的追いやすい中間版です。

## 共通パイプライン

全体の処理はおおむね次の流れです。

1. `adc_info.csv`、`*_star_info.csv`、各 planet の parquet 信号を読む。
2. ADC gain / offset 補正、非線形補正、dark 補正を適用する。
3. センサーごとに ROI を切り出す。
   - AIRS-CH0: 波長方向を `39:321` に制限し、主に行 `10:22` を使用
   - FGS1: 中央 `10:22, 10:22` を使用
4. CDS 化と binning で時系列を圧縮する。
5. `TransitModel` で惑星ごとの 1D トランジット深さを推定する。
6. `transit_depth`, `Rs`, `i` を特徴量にして、学習済み PyTorch MLP で以下を予測する。
   - FGS1 の平均トランジット深さ `mu`（1次元）
   - AIRS のスペクトル `mu`（282次元）
7. `sigma` を推定し、`sample_submission.csv` の列構造に合わせて `submission.csv` を保存する。

補足:

- `flat-field` 補正は `v2` 以降で導入されています。
- AIRS の後処理は `v2` 以降で強化され、`v3` 以降は自動平滑化、`v4` 以降は縮約・整合、`v5` では不確かさ連動の処理まで入っています。

## バージョンごとの差分

### v1

- ベースとなる推論パイプラインです。
- `SignalProcessor`、`TransitModel`、FGS/AIRS 用 MLP、`SubmissionGenerator` という構造はこの時点で揃っています。
- 一方で Notebook 的な書き方が残っており、後続版ほど実行フローは整理されていません。

### v2

- `flat-field` 補正が入り、センサー校正が一段強化されています。
- AIRS に対して Savitzky-Golay ベースの固定スムージング＋線形ブレンドを追加しています。
- `__main__` でまとまっているため、`v1` より扱いやすい構成です。

### v3

- AIRS 予測に対して、窓幅とブレンド率を候補集合から選ぶ適応的スムージングを導入しています。
- 行単位の winsorize で外れ値を抑えています。
- 1D 深さと FGS 予測のスケール差を確認する軽い自己キャリブレーションが追加されています。

### v4

- `sigma` を全体中央値へ軽く寄せる収縮を導入しています。
- AIRS スペクトルを全惑星中央値方向へ縮約する処理が追加されています。
- `mean(AIRS)/FGS` が極端になりすぎないよう、惑星ごとの比率整合を行います。
- `v3` より「予測の暴れを抑える」方向の改善が多い版です。

### v5

- AIRS の列重みを全区間ではなく OOT 区間の分散から作るように改善しています。
- MC Dropout を使って FGS/AIRS の推論を複数回行い、平均を `mu`、分散を追加の不確かさとして利用します。
- AIRS の自動スムージングが 2D の不確かさ情報を使えるようになっています。
- 任意で Tikhonov の 1次差分正則化スムージングを使えます。

## 実行前提

コードは Kaggle の以下のような入力配置を前提にしています。

- `/kaggle/input/ariel-data-challenge-2025`
- `/kaggle/input/ariel-2024-pqdm`
- `/kaggle/input/fgs1/pytorch/default/1/best_model.pth`
- `/kaggle/input/airs/pytorch/default/1/best_model_airs.pth`

主な依存ライブラリ:

- `numpy`
- `pandas`
- `torch`
- `scipy`
- `astropy`
- `tqdm`
- `pqdm`

補足:

- `ariel_v1.py` は追加で `scikit-learn` と `matplotlib` を import しています。
- 各ファイル先頭の `!pip install ...` は Jupyter / Kaggle のセルマジックです。  
  そのまま `python ariel_v5.py` のように実行すると失敗します。

## 実行方法

### Kaggle 上で使う場合

1. 必要な dataset / model checkpoint を Notebook にアタッチする。
2. 使いたい版を開く。通常は `ariel_v5.py` を起点にすれば十分です。
3. Notebook セルとして実行するか、Script として流す。
4. 実行後にカレントディレクトリへ `submission.csv` が出力されます。

### ローカルで使う場合

1. `pqdm` を含む依存関係を事前にインストールする。
2. 先頭の `!pip install ...` を削除または通常の Python 実行に置き換える。
3. `ROOT_PATH` やモデル重みのパスをローカル環境向けに修正する。
4. 必要なら `MODE` と `Config.DATASET` を `test` 以外へ変更する。

## このリポジトリを読むときの見方

- 実験の流れを追うなら `v1 -> v2 -> v3 -> v4 -> v5` の順で読むと自然です。
- 現状の完成形を知りたいなら `ariel_v5.py` だけ先に読むのが最短です。
- AIRS 後処理の変遷を見たいなら `v2` 以降の `AIRS_*` パラメータと `airs_adaptive_smooth` 周辺を見ると差分が分かりやすいです。

## 注意点

- どの版も実行のたびに `submission.csv` を上書きします。
- データパス、重みパス、列構造は Kaggle のコンペ前提でハードコードされています。
- リポジトリ内には評価スクリプト、訓練スクリプト、checkpoint 本体は含まれていません。
