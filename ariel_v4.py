# Kaggle (offline) 環境向け: pqdm を事前配布ホイールからインストール
!pip install --no-index --find-links=/kaggle/input/ariel-2024-pqdm pqdm > /dev/null

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from pqdm.threads import pqdm
from astropy.stats import sigma_clip
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT_PATH = "/kaggle/input/ariel-data-challenge-2025"
MODE = "test"

# =========================
# Config & Seed
# =========================
class Config:
    DATA_PATH = '/kaggle/input/ariel-data-challenge-2025'
    DATASET = "test"

    # 1本目寄りの初期値（必要に応じて微調整）
    SCALE = 0.952
    SIGMA = 0.00055

    CUT_INF = 39
    CUT_SUP = 321

    SENSOR_CONFIG = {
        "AIRS-CH0": {
            "raw_shape": [11250, 32, 356],
            "calibrated_shape": [1, 32, CUT_SUP - CUT_INF],
            "linear_corr_shape": (6, 32, 356),
            "dt_pattern": (0.1, 4.5),
            "binning": 30
        },
        "FGS1": {
            "raw_shape": [135000, 32, 32],
            "calibrated_shape": [1, 32, 32],
            "linear_corr_shape": (6, 32, 32),
            "dt_pattern": (0.1, 0.1),
            "binning": 30 * 12
        }
    }

    MODEL_PHASE_DETECTION_SLICE = slice(30, 140)
    MODEL_OPTIMIZATION_DELTA = 11
    MODEL_POLYNOMIAL_DEGREE = 3

    # === AIRS smoothing: 固定法の既定値（ADAPTIVE=False のときに使用） ===
    AIRS_SMOOTH_WINDOW = 13  # odd
    AIRS_SMOOTH_POLY = 2
    AIRS_BLEND_MAIN = 0.65
    AIRS_BLEND_SMOOTH = 0.35

    # === AIRS 自動適応スムージングのハイパラ ===
    AIRS_ADAPTIVE = True
    AIRS_SMOOTH_GRID = [9, 11, 13, 15]     # 窓候補（奇数に正規化）
    AIRS_BLEND_GRID  = [0.5, 0.65, 0.8]    # s = β*sm + (1-β)*raw の β 値
    AIRS_ADAPT_ALPHA = 0.55                # 0:忠実度重視, 1:なめらかさ重視
    AIRS_WINSOR_PCT  = (0.2, 99.8)         # 惑星行ごとウィンザー化（外れ抑制）

    # === AIRS 予測スペクトルの Empirical-Bayes 縮約（群中央値へ） ===
    AIRS_SHRINK_TO_MEDIAN = True
    AIRS_SHRINK_ALPHA = 0.18               # 縮約の基準強度（S/Nでスケーリング）
    AIRS_SHRINK_MAX = 0.35                 # 縮約上限

    # === AIRS-FGS 一貫性の穏やかなスケーリング（惑星別） ===
    AIRS_FGS_RATIO_ALIGN = True
    AIRS_FGS_RATIO_BOUNDS = (0.65, 1.45)   # mean(AIRS)/FGS をこの範囲にクリップ的に整合
    AIRS_FGS_RATIO_SOFTNESS = 0.35         # 1.0なら完全整合、0なら無視（中庸に）

    # === σ の Empirical-Bayes 収縮 ===
    SIGMA_SHRINK = True
    SIGMA_SHRINK_STRENGTH = 0.25          # 0〜1: 大きいほど全体メディアンに寄せる

    # FGSと1D深さスケールの軽い自己キャリブ（情報ログ用・任意）
    CALIBRATE_WITH_FGS = True              # s_corr を計算してログ表示（提出値は変えない）
    CALIBRATE_RECOMPUTE = False            # Trueなら再特徴量→MLP再推論（重いので既定False）

    # 実行まわり
    N_JOBS = 3
    SEED = 2025


def _set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# 位相検出（共通ヘルパ・安定版）
# =========================
def _phase_detector_signal(signal, cfg):
    sl = cfg.MODEL_PHASE_DETECTION_SLICE
    min_idx = int(np.argmin(signal[sl])) + sl.start
    s1 = signal[:min_idx]; s2 = signal[min_idx:]
    if s1.size < 3 or s2.size < 3:
        return 0, len(signal) - 1
    g1 = np.gradient(s1); g2 = np.gradient(s2)
    g1_max = np.max(np.abs(g1)) if g1.size else 0.0
    g2_max = np.max(np.abs(g2)) if g2.size else 0.0
    if g1_max != 0: g1 = g1 / g1_max
    if g2_max != 0: g2 = g2 / g2_max
    phase1 = int(np.argmin(g1)); phase2 = int(np.argmax(g2)) + min_idx
    return phase1, phase2

# =========================
# σ 推定（#4のクリップ幅＋仕上げ係数 1.04）＋ 収縮
# =========================
def estimate_sigma_fgs(preprocessed_data, cfg):
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12
    for single in preprocessed_data:
        air_white = savgol_filter(single[:, 1:].mean(axis=1), 20, 2)
        p1, p2 = _phase_detector_signal(air_white, cfg)
        p1 = max(delta, p1)
        p2 = min(len(air_white) - delta - 1, p2)

        fgs = single[:, 0]
        oot = (fgs[: p1 - delta] if p1 - delta > 0 else np.empty(0, fgs.dtype))
        if p2 + delta < fgs.size:
            oot = np.concatenate([oot, fgs[p2 + delta :]])
        inn = fgs[p1 + delta : max(p1 + delta, p2 - delta)]

        if oot.size == 0 or inn.size == 0:
            sig_rel.append(np.nan); continue

        n_oot, n_in = len(oot), len(inn)
        var_oot = np.nanvar(oot, ddof=1)
        var_in  = np.nanvar(inn, ddof=1)
        oot_mean = float(np.nanmean(oot)) if np.isfinite(np.nanmean(oot)) else float(np.nanmean(fgs))
        sigma_rel = np.sqrt(var_oot / max(n_oot,1) + var_in / max(n_in,1)) / max(oot_mean, eps)
        sig_rel.append(sigma_rel)

    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0

    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)

    k = np.clip(k, 0.85, 1.30)
    sigma_fgs = k * cfg.SIGMA
    sigma_fgs *= 1.04

    # 収縮（Empirical-Bayes風に全体メディアンへ）
    if cfg.SIGMA_SHRINK:
        m = np.nanmedian(sigma_fgs[np.isfinite(sigma_fgs)])
        w = float(cfg.SIGMA_SHRINK_STRENGTH)
        sigma_fgs = (1-w)*sigma_fgs + w*m

    return sigma_fgs


def estimate_sigma_air(preprocessed_data, cfg):
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12

    for single in preprocessed_data:
        white = np.nanmean(single[:, 1:], axis=1)
        white_s = savgol_filter(white, 20, 2)

        p1, p2 = _phase_detector_signal(white_s, cfg)
        p1 = max(delta, p1)
        p2 = min(len(white) - delta - 1, p2)

        oot_left = white[: p1 - delta] if p1 - delta > 0 else np.empty(0, white.dtype)
        oot_right = white[p2 + delta :] if (p2 + delta) < white.size else np.empty(0, white.dtype)
        oot = np.concatenate([oot_left, oot_right]) if (oot_left.size + oot_right.size) else oot_left
        inn = white[p1 + delta : max(p1 + delta, p2 - delta)]

        if oot.size == 0 or inn.size == 0:
            sig_rel.append(np.nan); continue

        n_oot, n_in = len(oot), len(inn)
        var_oot = np.nanvar(oot, ddof=1)
        var_in  = np.nanvar(inn, ddof=1)
        oot_mean = float(np.nanmean(oot)) if np.isfinite(np.nanmean(oot)) else float(np.nanmean(white))
        sigma_rel = np.sqrt(var_oot / max(n_oot,1) + var_in / max(n_in,1)) / max(oot_mean, eps)
        sig_rel.append(sigma_rel)

    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0

    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)

    k = np.clip(k, 0.92, 1.22)
    sigma_air = k * cfg.SIGMA
    sigma_air *= 1.04

    # 収縮
    if cfg.SIGMA_SHRINK:
        m = np.nanmedian(sigma_air[np.isfinite(sigma_air)])
        w = float(cfg.SIGMA_SHRINK_STRENGTH)
        sigma_air = (1-w)*sigma_air + w*m

    return sigma_air

# =========================
# 前処理パイプライン（flat補正あり）
# =========================
class SignalProcessor:
    def __init__(self, config):
        self.cfg = config
        self.adc_info = pd.read_csv(f"{self.cfg.DATA_PATH}/adc_info.csv")
        self.planet_ids = pd.read_csv(
            f'{self.cfg.DATA_PATH}/{self.cfg.DATASET}_star_info.csv',
            index_col='planet_id'
        ).index.astype(int)

    def _apply_linear_corr(self, linear_corr, signal):
        # Horner 法
        coeffs = np.flip(linear_corr, axis=0)
        x = signal.astype(np.float64, copy=False)
        out = np.empty_like(x, dtype=np.float64)
        out[...] = coeffs[0]
        for k in range(1, coeffs.shape[0]):
            np.multiply(out, x, out=out)
            out += coeffs[k]
        return out.astype(signal.dtype, copy=False)

    def _calibrate_single_signal(self, planet_id, sensor):
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]

        signal = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_signal_0.parquet").to_numpy()
        dark   = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/dark.parquet").to_numpy()
        dead   = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/dead.parquet").to_numpy()
        flat   = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/flat.parquet").to_numpy()
        linear_corr = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/linear_corr.parquet").values.astype(np.float64).reshape(sensor_cfg["linear_corr_shape"])

        signal = signal.reshape(sensor_cfg["raw_shape"])
        gain = self.adc_info[f"{sensor}_adc_gain"].iloc[0]
        offset = self.adc_info[f"{sensor}_adc_offset"].iloc[0]
        signal = signal / gain + offset

        # AIRS: 波長ROI、FGS: 中央ROI
        if sensor == "AIRS-CH0":
            signal = signal[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            linear_corr = linear_corr[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dark = dark[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dead = dead[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            flat = flat[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]

        if sensor == "FGS1":
            y0, y1, x0, x1 = 10, 22, 10, 22
            signal = signal[:, y0:y1, x0:x1]
            dark   = dark[y0:y1, x0:x1]
            dead   = dead[y0:y1, x0:x1]
            flat   = flat[y0:y1, x0:x1]
            linear_corr = linear_corr[:, y0:y1, x0:x1]

        # 非線形補正（Horner）
        np.maximum(signal, 0, out=signal)
        if sensor == "AIRS-CH0":
            sl = (slice(None), slice(10, 22), slice(None))
            signal[sl] = self._apply_linear_corr(linear_corr[:, 10:22, :], signal[sl])
        else:
            signal = self._apply_linear_corr(linear_corr, signal)

        # ダークスケール
        base_dt, inc = sensor_cfg["dt_pattern"]
        signal[::2]  -= dark * base_dt
        signal[1::2] -= dark * (base_dt + inc)

        # flat-field 補正（dead/NaN/0 を NaN に置換して除算）
        if sensor == "FGS1":
            flat_roi = flat.astype(signal.dtype, copy=False).copy()
            bad = (dead) | ~np.isfinite(flat_roi) | (flat_roi == 0)
            flat_roi[bad] = np.nan
            signal /= flat_roi
        elif sensor == "AIRS-CH0":
            y0, y1 = 10, 22
            flat_roi = flat[y0:y1, :].astype(signal.dtype, copy=False).copy()
            bad = (dead[y0:y1, :]) | ~np.isfinite(flat_roi) | (flat_roi == 0)
            flat_roi[bad] = np.nan
            signal[:, y0:y1, :] /= flat_roi
        else:
            flat2 = flat.astype(signal.dtype, copy=False).copy()
            bad2 = (dead) | ~np.isfinite(flat2) | (flat2 == 0)
            flat2[bad2] = np.nan
            signal /= flat2

        return signal

    def _preprocess_calibrated_signal(self, calibrated_signal, sensor):
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]
        binning = sensor_cfg["binning"]

        if sensor == "AIRS-CH0":
            signal_roi = calibrated_signal[:, 10:22, :]
        elif sensor == "FGS1":
            signal_roi = calibrated_signal[:, 10:22, 10:22]
            signal_roi = signal_roi.reshape(signal_roi.shape[0], -1)

        mean_signal = np.nanmean(signal_roi, axis=1)
        cds_signal = mean_signal[1::2] - mean_signal[0::2]

        n_bins = cds_signal.shape[0] // binning
        binned = np.array([
            cds_signal[j*binning : (j+1)*binning].mean(axis=0)
            for j in range(n_bins)
        ])

        if sensor == "AIRS-CH0":
            q_lo = np.nanpercentile(binned, 5.0, axis=1, keepdims=True)
            q_hi = np.nanpercentile(binned, 95.0, axis=1, keepdims=True)
            np.clip(binned, q_lo, q_hi, out=binned)

        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))

        if sensor == "AIRS-CH0":
            var = np.nanvar(binned, axis=0, ddof=1)
            med = np.nanmedian(var)
            safe_var = np.where(~np.isfinite(var) | (var <= 0), med if (np.isfinite(med) and med > 0) else 1.0, var)
            w = 1.0 / safe_var

            lo, hi = np.nanpercentile(w, 5.0), np.nanpercentile(w, 95.0)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                w = np.clip(w, lo, hi)

            M = binned.shape[1]
            s = np.nansum(w)
            if np.isfinite(s) and s > 0:
                w = w * (M / s)
            else:
                w = np.ones_like(w)

            binned *= w[None, :]

        return binned

    def _process_planet_sensor(self, args):
        planet_id, sensor = args['planet_id'], args['sensor']
        calibrated = self._calibrate_single_signal(planet_id, sensor)
        preprocessed = self._preprocess_calibrated_signal(calibrated, sensor)
        return preprocessed

    def process_all_data(self):
        args_fgs1 = [dict(planet_id=int(pid), sensor="FGS1") for pid in self.planet_ids]
        preprocessed_fgs1 = pqdm(args_fgs1, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS)

        args_airs_ch0 = [dict(planet_id=int(pid), sensor="AIRS-CH0") for pid in self.planet_ids]
        preprocessed_airs_ch0 = pqdm(args_airs_ch0, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS)

        preprocessed_signal = np.concatenate(
            [np.stack(preprocessed_fgs1), np.stack(preprocessed_airs_ch0)], axis=2
        )
        return preprocessed_signal

# =========================
# 1D トランジット深さ（NM＋閉形式のハイブリッド＋フォールバック）
# =========================
class TransitModel:
    def __init__(self, config):
        self.cfg = config

    def _phase_detector(self, signal):
        sl = self.cfg.MODEL_PHASE_DETECTION_SLICE
        min_index = int(np.argmin(signal[sl])) + sl.start
        s1 = signal[:min_index]; s2 = signal[min_index:]
        g1 = np.gradient(s1); g2 = np.gradient(s2)
        g1mx = np.max(np.abs(g1)) if g1.size else 0.0
        g2mx = np.max(np.abs(g2)) if g2.size else 0.0
        if g1mx != 0: g1 = g1 / g1mx
        if g2mx != 0: g2 = g2 / g2mx
        return int(np.argmin(g1)), int(np.argmax(g2)) + min_index

    @staticmethod
    def _polyfit_oot(x, y, deg):
        c = np.polyfit(x, y, deg=deg); p = np.poly1d(c)
        r = y - p(x)
        mad = np.median(np.abs(r - np.median(r))) + 1e-12
        mask = np.abs(r) < 4.0 * 1.4826 * mad
        if mask.sum() >= deg + 1:
            c = np.polyfit(x[mask], y[mask], deg=deg)
        return np.poly1d(c)

    @staticmethod
    def _closed_form_s(y_in, poly_in):
        denom = float(np.sum(y_in * y_in))
        if denom <= 0 or not np.isfinite(denom): return 0.0
        numer = float(np.sum(y_in * poly_in))
        s = (numer / denom) - 1.0
        return float(np.clip(s, -0.03, 0.03))  # 少し緩め

    def _objective(self, s, signal, p1, p2):
        d = self.cfg.MODEL_OPTIMIZATION_DELTA
        power = self.cfg.MODEL_POLYNOMIAL_DEGREE
        if p1 - d <= 0 or p2 + d >= len(signal) or p2 - d - (p1 + d) < 5:
            d = 2
        y = np.concatenate([signal[:p1-d], signal[p1+d:p2-d]*(1+s), signal[p2+d:]])
        x = np.arange(len(y))
        poly = np.poly1d(np.polyfit(x, y, deg=power))
        return float(np.mean(np.abs(poly(x) - y)))

    def predict(self, single_preprocessed_signal):
        sig = single_preprocessed_signal[:, 1:].mean(axis=1)
        sig = savgol_filter(sig, 23, 2)  # 奇数窓

        p1, p2 = self._phase_detector(sig)
        d = self.cfg.MODEL_OPTIMIZATION_DELTA
        p1 = max(d, p1); p2 = min(len(sig) - d - 1, p2)

        # Nelder–Mead
        try:
            s_nm = minimize(lambda s: self._objective(s[0], sig, p1, p2),
                            x0=[0.0001], method="Nelder-Mead").x[0]
            j_nm = self._objective(s_nm, sig, p1, p2)
        except Exception:
            s_nm, j_nm = 0.0, np.inf

        # 閉形式（OOT poly × IN整合）
        left_end = max(0, p1 - d); right_start = min(len(sig), p2 + d)
        x_oot = np.r_[0:left_end, right_start:len(sig)].astype(float)
        x_in  = np.r_[p1 + d : max(p1 + d, p2 - d)]
        if x_oot.size >= 4 and x_in.size >= 5:
            try:
                p = self._polyfit_oot(x_oot, sig[x_oot.astype(int)], deg=self.cfg.MODEL_POLYNOMIAL_DEGREE)
                s_cf = self._closed_form_s(sig[x_in], p(x_in))
                j_cf = self._objective(s_cf, sig, p1, p2)
            except Exception:
                s_cf, j_cf = s_nm, j_nm
        else:
            s_cf, j_cf = s_nm, j_nm

        # 採用
        s = s_nm if j_nm <= j_cf else s_cf
        if not np.isfinite(s):  # 最終フォールバック
            s = 0.0
        return s

    def predict_all(self, preprocessed_signals):
        arr = [self.predict(x) for x in tqdm(preprocessed_signals)]
        return np.array(arr) * self.cfg.SCALE

# =========================
# MLP (FGS 1次元 / AIRS 282次元)
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, dim, p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return self.relu(out + identity)

class ResNetMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=1, num_blocks=3, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, p=dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x

class ResidualBlock2(nn.Module):
    def __init__(self, dim, p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return self.relu(out + identity)

class ResNetMLP2(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=282, num_blocks=3, dropout_rate=0.3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock2(hidden_dim, p=dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x

# =========================
# AIRS 平滑ブレンド（安全版＆自動適応）＋ 追加処理
# =========================
def _savgol_safe(arr, win, poly, axis=1):
    L = arr.shape[axis]
    win = int(win)
    win = win if win % 2 == 1 else win + 1  # odd
    win = min(win, L if L % 2 == 1 else L - 1)
    if win <= poly or win < 3 or L < 3:
        return arr
    return savgol_filter(arr, window_length=win, polyorder=int(poly), axis=axis)

def _winsorize_per_row(arr, p_low=0.2, p_high=99.8):
    ql = np.nanpercentile(arr, p_low, axis=1, keepdims=True)
    qh = np.nanpercentile(arr, p_high, axis=1, keepdims=True)
    return np.clip(arr, ql, qh)

def _roughness(arr):
    d2 = np.diff(arr, n=2, axis=1)
    return np.nanmean(d2*d2, axis=1)

def _fidelity(smooth, raw, sig=None):
    if sig is not None:
        w = 1.0 / np.maximum(sig.reshape(-1,1), 1e-12)
    else:
        w = 1.0
    res = (smooth - raw)
    return np.nanmean((res*res)*w, axis=1)

def airs_adaptive_smooth(pred2, cfg, sigma_air_vec=None):
    raw = pred2.copy()
    if cfg.AIRS_WINSOR_PCT:
        raw = _winsorize_per_row(raw, *cfg.AIRS_WINSOR_PCT)
    best = raw.copy()
    best_J = np.full(raw.shape[0], np.inf, dtype=float)

    for w in cfg.AIRS_SMOOTH_GRID:
        sm = _savgol_safe(raw, w, cfg.AIRS_SMOOTH_POLY, axis=1)
        for beta in cfg.AIRS_BLEND_GRID:
            s = beta*sm + (1-beta)*raw
            J = cfg.AIRS_ADAPT_ALPHA*_roughness(s) + (1-cfg.AIRS_ADAPT_ALPHA)*_fidelity(s, raw, sigma_air_vec)
            take = J < best_J
            if np.any(take):
                best[take] = s[take]
                best_J[take] = J[take]
    return best

def airs_shrink_to_global_median(pred2, cfg, sigma_air_vec=None):
    """全惑星の波長ごとの中央値へ、惑星ごとに S/N に応じた係数で縮約."""
    Y = np.asarray(pred2, float)
    mu_k = np.nanmedian(Y, axis=0, keepdims=True)  # (1, 282)
    if sigma_air_vec is None:
        lam = cfg.AIRS_SHRINK_ALPHA
        lam = np.clip(lam, 0.0, cfg.AIRS_SHRINK_MAX)
        return (1-lam)*Y + lam*mu_k
    med_sig = float(np.nanmedian(sigma_air_vec))
    # S/N 低いほど強く縮約（σ大 → λ大）
    lam_vec = (np.asarray(sigma_air_vec).reshape(-1,1) / max(med_sig, 1e-12)) * float(cfg.AIRS_SHRINK_ALPHA)
    lam_vec = np.clip(lam_vec, 0.0, float(cfg.AIRS_SHRINK_MAX))
    return (1-lam_vec)*Y + lam_vec*mu_k

def airs_align_mean_to_fgs(pred2, fgs_mu, cfg):
    """惑星ごとに mean(AIRS)/FGS を緩く範囲内へ近づける."""
    Y = np.asarray(pred2, float)
    fgs = np.clip(np.asarray(fgs_mu, float).reshape(-1), 1e-12, None)
    m = np.nanmean(Y, axis=1)
    ratio = m / fgs
    lo, hi = cfg.AIRS_FGS_RATIO_BOUNDS
    soft = float(cfg.AIRS_FGS_RATIO_SOFTNESS)
    out = Y.copy()
    # 下側
    idx = ratio < lo
    if np.any(idx):
        scale = (lo / np.maximum(ratio[idx], 1e-12))**soft
        out[idx] = out[idx] * scale.reshape(-1,1)
    # 上側
    idx = ratio > hi
    if np.any(idx):
        scale = (hi / ratio[idx])**soft
        out[idx] = out[idx] * scale.reshape(-1,1)
    return out

# =========================
# 提出生成（安全化）
# =========================
class SubmissionGenerator:
    def __init__(self, config):
        self.cfg = config
        self.sample_submission = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/sample_submission.csv", index_col="planet_id")

    def create(self, predictions1, predictions2, predictions, sigma_fgs=None, sigma_air=None):
        planet_ids = self.sample_submission.index
        n_mu = self.sample_submission.shape[1] // 2  # 283 (FGS1 + AIRS 282)

        # μ 初期化（1D深さベース：後で上書き）
        preds = np.asarray(predictions, dtype=float).reshape(-1)
        preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
        preds = np.clip(preds, 0.0, None)
        mu = np.tile(preds.reshape(-1, 1), (1, n_mu))

        # σ 初期化
        sigmas = np.full_like(mu, self.cfg.SIGMA, dtype=float)

        if sigma_fgs is not None:
            sigma_fgs = np.asarray(sigma_fgs, dtype=float).reshape(-1)
            sigma_fgs = np.nan_to_num(sigma_fgs, nan=self.cfg.SIGMA, posinf=self.cfg.SIGMA, neginf=self.cfg.SIGMA)
            sigmas[:, 0] = np.clip(sigma_fgs, 1e-6, 0.1)

        if sigma_air is not None:
            sigma_air = np.asarray(sigma_air, dtype=float).reshape(-1, 1)
            sigma_air = np.nan_to_num(sigma_air, nan=self.cfg.SIGMA, posinf=self.cfg.SIGMA, neginf=self.cfg.SIGMA)
            sigmas[:, 1:] = np.clip(sigma_air, 1e-6, 0.1)

        submission_df = pd.DataFrame(
            np.concatenate([mu, sigmas], axis=1),
            columns=self.sample_submission.columns,
            index=planet_ids
        )

        # μ を最終上書き（FGS=predictions1, AIRS=predictions2）
        preds1 = np.nan_to_num(np.asarray(predictions1, float), nan=0.0, posinf=1.0, neginf=0.0)
        preds2 = np.nan_to_num(np.asarray(predictions2, float), nan=0.0, posinf=1.0, neginf=0.0)

        submission_df.iloc[:, 0]     = np.clip(preds1.reshape(-1), 0.0, None)
        submission_df.iloc[:, 1:283] = np.clip(preds2, 0.0, None)

        # μ/σ の最終ガード
        n_mu = submission_df.shape[1] // 2
        submission_df.iloc[:, :n_mu] = np.clip(
            np.nan_to_num(submission_df.iloc[:, :n_mu].values, nan=0.0, posinf=1.0, neginf=0.0),
            0.0, None
        )
        submission_df.iloc[:, n_mu:] = np.nan_to_num(
            submission_df.iloc[:, n_mu:].values,
            nan=self.cfg.SIGMA, posinf=self.cfg.SIGMA, neginf=self.cfg.SIGMA
        )

        # インデックス順を sample_submission に合わせる
        submission_df = submission_df.reindex(self.sample_submission.index)

        submission_df.to_csv("submission.csv")
        print("[Submit] saved -> submission.csv")
        return submission_df

# =========================
# 軽い自己キャリブレーション（任意）
# =========================
def calibrate_scale_with_fgs(one_d_depth_scaled, fgs_mu):
    # s*one_d ≈ fgs_mu を最小二乗で推定（5–95%分位で粗外れ除外）
    x = np.clip(np.asarray(one_d_depth_scaled, float).reshape(-1), 0, None)
    y = np.clip(np.asarray(fgs_mu, float).reshape(-1), 0, None)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[m], y[m]
    if x.size < 10: return 1.0
    ratio = y / np.maximum(x, 1e-12)
    lo, hi = np.percentile(ratio, [5, 95])
    m2 = (ratio >= lo) & (ratio <= hi)
    x, y = x[m2], y[m2]
    denom = np.dot(x, x)
    if denom <= 0: return 1.0
    s = float(np.dot(x, y) / denom)
    return np.clip(s, 0.95, 1.05)  # 動かし過ぎない

# =========================
# Main
# =========================
if __name__ == "__main__":
    _set_seed(Config.SEED)
    try:
        Config.N_JOBS = max(1, min(os.cpu_count() or 1, Config.N_JOBS))
    except Exception:
        pass

    # 前処理
    signal_processor = SignalProcessor(Config)
    preprocessed_data = signal_processor.process_all_data()

    # 1D 深さ（TransitModel → 特徴量用）
    model_1d = TransitModel(Config)
    with torch.inference_mode():
        predictions = model_1d.predict_all(preprocessed_data)  # shape (N,)
    print(f"[1D] mean={predictions.mean():.6g} std={predictions.std():.6g}")

    # メタ特徴（transit_depth, Rs, i）
    StarInfo = pd.read_csv(ROOT_PATH + f"/{MODE}_star_info.csv")
    StarInfo["planet_id"] = StarInfo["planet_id"].astype(int)
    PlanetIds = StarInfo["planet_id"].tolist()
    StarInfo = StarInfo.set_index("planet_id")

    input_df = StarInfo.copy()
    pred_series = pd.Series(predictions, index=input_df.index)
    input_df.insert(0, "transit_depth", (pred_series * 10000).to_numpy())
    features = ["transit_depth", "Rs", "i"]
    X = input_df[features].values.astype("float32")
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # FGS (1次元 μ)
    model_fgs = ResNetMLP(num_blocks=80, dropout_rate=0.2)
    model_fgs.load_state_dict(torch.load("/kaggle/input/fgs1/pytorch/default/1/best_model.pth", map_location="cpu"))
    model_fgs.eval()

    # AIRS (282次元 μ)
    model_airs = ResNetMLP2(num_blocks=80, dropout_rate=0.3)
    model_airs.load_state_dict(torch.load("/kaggle/input/airs/pytorch/default/1/best_model_airs.pth", map_location="cpu"))
    model_airs.eval()

    with torch.inference_mode():
        predictions1 = model_fgs(X_tensor).cpu().numpy() / 10000.0      # FGS μ (N,1)
        predictions2 = model_airs(X_tensor).cpu().numpy() / 10000.0     # AIRS μ (N,282)

    # --- 軽い自己キャリブ（情報ログ用途） ---
    if Config.CALIBRATE_WITH_FGS:
        s_corr = calibrate_scale_with_fgs(predictions, predictions1.reshape(-1))
        print(f"[Calibrate] s_corr (one_d -> FGS) = {s_corr:.6f}")
        if Config.CALIBRATE_RECOMPUTE and abs(s_corr - 1.0) > 1e-3:
            predictions *= s_corr
            input_df["transit_depth"] = (predictions * 10000).astype(np.float32)
            X = input_df[features].values.astype("float32")
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.inference_mode():
                predictions1 = model_fgs(X_tensor).cpu().numpy() / 10000.0
                predictions2 = model_airs(X_tensor).cpu().numpy() / 10000.0

    # σ 推定（AIRS自動法の目的にも使う）
    sigma_fgs_vec = estimate_sigma_fgs(preprocessed_data, Config)
    sigma_air_vec = estimate_sigma_air(preprocessed_data, Config)

    # AIRS: 固定 or 自動スムージング＆ブレンド
    if Config.AIRS_ADAPTIVE:
        predictions2 = airs_adaptive_smooth(predictions2, Config, sigma_air_vec=sigma_air_vec)
    else:
        predictions2_smooth = _savgol_safe(predictions2, Config.AIRS_SMOOTH_WINDOW, Config.AIRS_SMOOTH_POLY, axis=1)
        w_main, w_smooth = float(Config.AIRS_BLEND_MAIN), float(Config.AIRS_BLEND_SMOOTH)
        s = max(1e-12, w_main + w_smooth)
        predictions2 = (w_main/s) * predictions2 + (w_smooth/s) * predictions2_smooth

    # === 追加強化: EB縮約 → FGS整合 ===
    if Config.AIRS_SHRINK_TO_MEDIAN:
        predictions2 = airs_shrink_to_global_median(predictions2, Config, sigma_air_vec=sigma_air_vec)

    if Config.AIRS_FGS_RATIO_ALIGN:
        predictions2 = airs_align_mean_to_fgs(predictions2, predictions1.reshape(-1), Config)

    # 提出
    submission = SubmissionGenerator(Config).create(
        predictions1=predictions1,
        predictions2=predictions2,
        predictions=predictions,
        sigma_fgs=sigma_fgs_vec,
        sigma_air=sigma_air_vec
    )
    print(submission.head())
    # 出力: submission.csv