#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
문제 2 — 소청초 중간층 수온 연직 구조 복원 : 전체 재현 파이프라인 (규정 준수판)

원칙
  * 사전학습 가중치 미사용. 배포된 observations.csv / test_index.csv 만 사용.
  * 리더보드 점수를 사용하지 않음. 모든 계수·임계값은 배포 데이터 내부 검증으로 산출.
  * 코드에 결과 상수를 박지 않음. 탐색 격자(grid)만 상수이고, 선택된 값은 실행 중 계산됨.
  * 중간 산출물(pickle/csv) 의존 없음. 원자료에서 제출까지 한 번에 재생성.

사용법
  python run_p2.py --data <P2_profile_restore 폴더> --out submission.csv
  python run_p2.py --data <...> --sim              # 2024년 가을 가상 가림 검증만 수행
옵션
  --seeds N   멤버당 시드 수 (기본 3)
  --jobs  J   코어 수 (기본 -1)
  --fast      탐색 격자를 줄인 빠른 모드(개발용)

산출되는 값(전부 실행 중 적합):
  1) 계절 학습창 (doy 하한/상한)        : 2024년 가상 가림 검증 RMSE 최소화로 선택
  2) 이상치 임계값                      : 잔차 절대편차의 분위수(MAD 기반)로 계산
  3) 앙상블 가중                        : 검증 예측에 대한 비음수 최소제곱(NNLS)
  4) 시간 평활 창                       : 검증 RMSE 최소화로 선택
  5) 층5 결측 구간 수축함수 s(x)        : 블록 교차검증 OOF 예측에 대한 능형 최소제곱
                                          (날짜 리터럴 없음. 관측 가능한 체제 변수의 함수)
"""
import argparse, os, time, numpy as np, pandas as pd
import feats as FE
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from scipy.optimize import nnls
# 백엔드는 명시적으로 고정한다. 제출본은 scikit-learn 백엔드로 생성되었으므로
# 재현 검증 시에도 기본값(sklearn)을 그대로 두면 동일한 결과가 나온다.
HAS_LGB = False
def _select_backend(name):
    global HAS_LGB, lgb
    if name == 'sklearn':
        HAS_LGB = False; return
    try:
        import lightgbm as _l
        globals()['lgb'] = _l; HAS_LGB = True
    except Exception:
        if name == 'lightgbm':
            raise SystemExit('lightgbm 백엔드를 요청했으나 설치되어 있지 않습니다.')
        HAS_LGB = False
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

TGT = ['d07', 'd09', 'd15']; LAY = {'d07': 2, 'd09': 3, 'd15': 4}
MAP24 = {1:'d04',2:'d07',3:'d09',4:'d15',5:'d19',6:'d31',7:'d49'}
MAP25 = {1:'d04',2:'d07',3:'d09',4:'d15',5:'d19',6:'d31',7:'d39',8:'d49'}
KST = 'Asia/Seoul'

# ── 탐색 격자 (결과가 아니라 탐색 범위) ────────────────────────────────
SEASON_GRID       = [(180,340),(200,330),(215,320),(230,310)]
SMOOTH_GRID       = [1,3,5,7,9]
OUTLIER_QUANTILE  = 0.99     # 잔차 절대편차 분위수 -> 임계값을 데이터에서 계산
PHI_MIN_DEN       = 0.15     # |T_상 - T_하| 가 이보다 작으면 phi 정의 불가 -> 상층값 사용
CV_BLOCK_DAYS     = 14       # 수축함수 적합용 블록 교차검증 블록 길이

# ── 데이터 적재 : 층 번호가 아니라 실제 수심으로 정규화 ────────────────
def load_wide(d):
    o = pd.read_csv(os.path.join(d, 'observations.csv'))
    o = o[o.year < 2026].copy()
    o['t'] = pd.to_datetime(o.time)
    o['dc'] = np.where(o.year == 2024, o.layer.map(MAP24), o.layer.map(MAP25))
    pv = lambda v: o.pivot_table(index='t', columns='dc', values=v)
    T, S, Z = pv('temp'), pv('psal'), pv('depth')
    T.columns = ['T_'+c for c in T.columns]
    S.columns = ['S_'+c for c in S.columns]
    Z.columns = ['Z_'+c for c in Z.columns]
    return T.join(S).join(Z).sort_index().asfreq('10min')

def build_features(W0, use_d19):
    F, W = FE.make(W0, use_d19)
    F = pd.concat([F, FE.iso_feats(W, use_d19), FE.hf_feats(W, use_d19),
                   FE.dens_feats(W, use_d19)], axis=1)
    out = {}
    for g in TGT:
        X = FE.interp_feats(F, W, W['Z_'+g], use_d19)
        X = FE.add_shape(X, W, W['Z_'+g], use_d19)
        X = FE.rel_depth_feats(X, W, W['Z_'+g], use_d19)
        for c in [c for c in X.columns if c.startswith('ip_')]:
            s = X[c]
            for k in (18,-18,36,-36,144,-144): X[c+'_l%d'%k] = s - s.shift(k)
            X[c+'_a24'] = s - s.rolling(144, center=True, min_periods=48).mean()
        X['ztgt'] = W['Z_'+g]
        out[g] = X.copy()
    return out, W

# ── 학습기 ─────────────────────────────────────────────────────────────
def make_model(kind, seed, jobs):
    if HAS_LGB:
        cfg = {'deep': dict(n_estimators=1400, learning_rate=0.030, num_leaves=63,
                            min_child_samples=30, colsample_bytree=0.40, reg_lambda=1.0),
               'mid':  dict(n_estimators=900,  learning_rate=0.045, num_leaves=31,
                            min_child_samples=60, colsample_bytree=0.50, reg_lambda=3.0),
               'fst':  dict(n_estimators=700,  learning_rate=0.060, num_leaves=31,
                            min_child_samples=40, colsample_bytree=0.60, reg_lambda=2.0)}[kind]
        return lgb.LGBMRegressor(subsample=0.85, subsample_freq=1, random_state=seed,
                                 n_jobs=jobs, verbose=-1, **cfg)
    cfg = {'deep': dict(max_iter=900, learning_rate=0.035, max_leaf_nodes=63, max_bins=192,
                        min_samples_leaf=30, l2_regularization=1.0, max_features=0.40),
           'mid':  dict(max_iter=600, learning_rate=0.050, max_leaf_nodes=31, max_bins=128,
                        min_samples_leaf=60, l2_regularization=3.0, max_features=0.50),
           'fst':  dict(max_iter=500, learning_rate=0.060, max_leaf_nodes=31, max_bins=128,
                        min_samples_leaf=40, l2_regularization=2.0, max_features=0.60)}[kind]
    return HGB(early_stopping=False, random_state=seed, **cfg)

MEMBERS = ['deep', 'mid', 'fst', 'etr', 'ridge']

# ── phi 파라미터화 : 손실이 곧 수온 RMSE ───────────────────────────────
class PhiTask:
    """phi = (T_target - T_anchor) / (T_surface - T_anchor), 표본가중 = (T_surface - T_anchor)^2
       => 최소제곱 손실이 수온 제곱오차와 동일해진다."""
    def __init__(self, W0, anchor):
        self.W0 = W0
        self.top = W0['T_d04'].interpolate(limit=144, limit_direction='both')
        self.bot = W0['T_'+anchor].interpolate(limit=36, limit_direction='both')
        self.den = self.top - self.bot
    def y(self, g):   return (self.W0['T_'+g] - self.bot) / self.den
    def w(self):      return self.den ** 2
    def to_temp(self, phi_hat, mask):
        pr = np.clip(phi_hat, -0.3, 1.3) * self.den[mask].values + self.bot[mask].values
        bad = (self.den[mask].abs().values <= PHI_MIN_DEN) | ~np.isfinite(pr)
        return pd.Series(np.where(bad, self.top[mask].values, pr), index=self.W0.index[mask])

def outlier_mask(phi, q=OUTLIER_QUANTILE):
    """임계값을 손으로 정하지 않고 잔차 절대편차의 분위수로 계산한다."""
    resid = (phi - phi.rolling(7, center=True, min_periods=3).median()).abs()
    thr = float(np.nanquantile(resid.values, q))
    return (resid < thr) | phi.isna(), thr

def train_mask(task, g, season, extra_ok=None):
    y = task.y(g)
    ok = task.W0['T_'+g].notna() & np.isfinite(y) & (task.den.abs() > PHI_MIN_DEN) & season
    om, _ = outlier_mask(y)
    ok = ok & om
    if extra_ok is not None: ok = ok & extra_ok
    return ok

def fit_predict(X, task, season, pred_mask, seeds, jobs, members=MEMBERS, extra_ok=None):
    """멤버별 예측(수온)을 dict 로 반환."""
    res = {}
    for kind in members:
        out = {}
        for g in TGT:
            ok = train_mask(task, g, season, extra_ok)
            y, w = task.y(g), task.w()
            if kind in ('deep','mid','fst'):
                v = np.mean([make_model(kind, s, jobs).fit(X[g][ok], y[ok], sample_weight=w[ok])
                             .predict(X[g][pred_mask]) for s in range(seeds)], axis=0)
            else:
                cols = [c for c in X[g].columns if X[g].loc[ok, c].notna().mean() > 0.6]
                med = X[g].loc[ok, cols].median()
                Xtr, Xte = X[g].loc[ok, cols].fillna(med), X[g].loc[pred_mask, cols].fillna(med)
                if kind == 'ridge':
                    sc = StandardScaler().fit(Xtr)
                    v = Ridge(alpha=100.0).fit(sc.transform(Xtr), y[ok],
                                               sample_weight=w[ok]).predict(sc.transform(Xte))
                else:
                    v = ExtraTreesRegressor(n_estimators=300, max_features=0.25, min_samples_leaf=8,
                                            n_jobs=jobs, random_state=0
                                            ).fit(Xtr, y[ok], sample_weight=w[ok]).predict(Xte)
            out[g] = task.to_temp(v, pred_mask)
        res[kind] = out
    return res

# ── 검증 유틸 ──────────────────────────────────────────────────────────
def rmse_of(pred, W0):
    e = []
    for g in TGT:
        t = W0['T_'+g].reindex(pred[g].index); ok = t.notna()
        e.append((pred[g][ok] - t[ok]).values)
    return float(np.sqrt(np.mean(np.concatenate(e) ** 2)))

def smooth(pred, win):
    if win <= 1: return pred
    return {g: pred[g].rolling(win, center=True, min_periods=1).mean() for g in pred}

def fit_blend_weights(member_preds, W0):
    """검증 예측에 대한 비음수 최소제곱으로 앙상블 가중을 적합(합=1로 정규화)."""
    cols, rows, tgt = list(member_preds.keys()), [], []
    for g in TGT:
        t = W0['T_'+g].reindex(member_preds[cols[0]][g].index); ok = t.notna()
        rows.append(np.column_stack([member_preds[k][g][ok].values for k in cols]))
        tgt.append(t[ok].values)
    A, b = np.vstack(rows), np.concatenate(tgt)
    w, _ = nnls(A, b)
    if w.sum() <= 0: w = np.ones(len(cols))
    w = w / w.sum()
    return dict(zip(cols, w))

def blend(member_preds, weights):
    return {g: sum(weights[k] * member_preds[k][g] for k in weights) for g in TGT}

# ── 층5 결측 구간 : 수축함수 s(x) 를 데이터에서 적합 ───────────────────
REGIME = ['const', 'doy_s', 'doy_c', 'strat0431', 'strat3149', 'cool24', 'cool72', 'hf04']

def regime_matrix(W0, index):
    top = W0['T_d04'].interpolate(limit=144, limit_direction='both')
    doy = index.dayofyear + index.hour / 24.0
    hf = (top.diff().abs().rolling(144, min_periods=24).mean()).reindex(index)
    M = pd.DataFrame({
        'const': 1.0,
        'doy_s': np.sin(2*np.pi*doy/365.25), 'doy_c': np.cos(2*np.pi*doy/365.25),
        'strat0431': (top - W0['T_d31']).reindex(index),
        'strat3149': (W0['T_d31'] - W0['T_d49']).reindex(index),
        'cool24': (top - top.shift(144)).reindex(index),
        'cool72': (top - top.shift(432)).reindex(index),
        'hf04': hf}, index=index)
    return M[REGIME].fillna(0.0)

def oof_gap_predictions(X2, task2, season, seeds, jobs):
    """층5를 쓰지 않는 모델의 블록 교차검증 OOF 예측. 수축함수 적합용 학습자료."""
    idx = task2.W0.index
    labelled = season & pd.concat([task2.W0['T_'+g].notna() for g in TGT], axis=1).any(axis=1)
    days = idx.normalize()
    block = ((days - days.min()).days // CV_BLOCK_DAYS)
    blocks = sorted(set(block[labelled.values]))
    oof = {g: pd.Series(np.nan, index=idx) for g in TGT}
    for b in blocks:
        vm = pd.Series(block == b, index=idx) & labelled
        if vm.sum() < 200: continue
        res = fit_predict(X2, task2, season & ~vm, vm, seeds=1, jobs=jobs, members=['mid'])
        for g in TGT: oof[g][vm.values] = res['mid'][g].values
    return oof

def fit_shrink(oof, task2, W0, season):
    """delta_true ≈ delta_pred * (x · w) 를 능형 최소제곱으로 적합.
       날짜나 손으로 정한 계수 없이, 관측 가능한 체제 변수만으로 수축 정도가 결정된다."""
    top = task2.top
    A, y = [], []
    for g in TGT:
        idxg = oof[g].dropna().index
        t = W0['T_'+g].reindex(idxg)
        ok = t.notna() & season.reindex(idxg).astype(bool)
        idxg = idxg[ok.values]
        if len(idxg) == 0: continue
        dp = (oof[g].reindex(idxg) - top.reindex(idxg)).values
        dt = (t.reindex(idxg) - top.reindex(idxg)).values
        M = regime_matrix(W0, idxg).values
        A.append(M * dp[:, None]); y.append(dt)
    A, y = np.vstack(A), np.concatenate(y)
    good = np.isfinite(A).all(1) & np.isfinite(y)
    A, y = A[good], y[good]
    lam = 1e-3 * A.shape[0]
    w = np.linalg.solve(A.T @ A + lam*np.eye(A.shape[1]), A.T @ y)
    return w

def apply_shrink(pred, w, task2, W0):
    top = task2.top
    out = {}
    for g in TGT:
        ix = pred[g].index
        s = regime_matrix(W0, ix).values @ w
        s = np.clip(s, 0.0, 1.3)
        out[g] = top.reindex(ix) + s * (pred[g] - top.reindex(ix))
    return out

# ── 메인 ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--jobs', type=int, default=-1)
    ap.add_argument('--sim', action='store_true'); ap.add_argument('--fast', action='store_true')
    ap.add_argument('--backend', choices=['sklearn','lightgbm','auto'], default='sklearn',
                    help='제출본은 sklearn 으로 생성됨. 기본값 유지 시 동일 결과 재현.')
    a = ap.parse_args(); t0 = time.time()
    _select_backend(a.backend)
    tick = lambda m: print('[%6.0fs] %s' % (time.time()-t0, m), flush=True)
    import sklearn, scipy
    tick('backend=%s (lightgbm=%s) | numpy %s pandas %s scipy %s sklearn %s'
         % (a.backend, HAS_LGB, np.__version__, pd.__version__, scipy.__version__, sklearn.__version__))

    W0 = load_wide(a.data); idx = W0.index
    W0f = W0.copy()
    for c in ['T_d19','S_d19','Z_d19']:          # 층5 산발 단기결측(<=1h)만 메움
        W0f[c] = W0[c].interpolate(limit=6, limit_direction='both')
    ts = lambda x: pd.Timestamp(x, tz=KST)

    X1, _ = build_features(W0f, True);  tick('features (with layer5) %d' % X1['d15'].shape[1])
    X2, _ = build_features(W0,  False); tick('features (no layer5)  %d' % X2['d15'].shape[1])
    task1, task2 = PhiTask(W0f, 'd19'), PhiTask(W0, 'd31')

    # ── 내부 검증 하네스 : 2024년 가을에 실제 테스트와 동일한 가림을 재현 ──
    hole = (idx >= ts('2024-09-01')) & (idx <= ts('2024-10-31 23:50'))
    hgap = ts('2024-10-14 20:10')
    v1 = pd.Series(hole & (idx <  hgap), index=idx)   # 층5 있음
    v2 = pd.Series(hole & (idx >= hgap), index=idx)   # 층5 없음
    notval = pd.Series(~hole, index=idx)

    grid = SEASON_GRID[:2] if a.fast else SEASON_GRID
    best = None
    for lo_, hi_ in grid:
        seas = pd.Series((idx.dayofyear >= lo_) & (idx.dayofyear <= hi_), index=idx)
        mp = fit_predict(X1, task1, seas & notval, v1, seeds=1, jobs=a.jobs, members=['mid'])
        r = rmse_of(mp['mid'], W0)
        tick('  season window (%d,%d) -> validation RMSE %.4f' % (lo_, hi_, r))
        if best is None or r < best[1]: best = ((lo_, hi_), r)
    SEASON_WIN = best[0]; tick('SELECTED season window %s' % (SEASON_WIN,))
    season = pd.Series((idx.dayofyear >= SEASON_WIN[0]) & (idx.dayofyear <= SEASON_WIN[1]), index=idx)

    # 앙상블 가중과 평활창을 검증 예측에서 적합
    mp = fit_predict(X1, task1, season & notval, v1, seeds=max(1, a.seeds-1), jobs=a.jobs)
    Wt = fit_blend_weights(mp, W0); tick('SELECTED blend weights ' + str({k: round(v,3) for k,v in Wt.items()}))
    bl = blend(mp, Wt)
    sm = min(SMOOTH_GRID, key=lambda k: rmse_of(smooth(bl, k), W0))
    tick('SELECTED smoothing window %d (seg1 validation RMSE %.4f)' % (sm, rmse_of(smooth(bl, sm), W0)))

    # 층5 결측 구간 수축함수를 블록 교차검증 OOF 로 적합
    tick('fitting shrink function on block-CV OOF ...')
    oof = oof_gap_predictions(X2, task2, season, seeds=1, jobs=a.jobs)
    wshr = fit_shrink(oof, task2, W0, season)
    tick('SELECTED shrink coefficients ' + str({k: round(float(v),4) for k, v in zip(REGIME, wshr)}))

    # 검증 하네스에서 최종 구성의 성능 확인
    mp2 = fit_predict(X2, task2, season & notval, v2, seeds=max(1, a.seeds-1), jobs=a.jobs)
    Wt2 = fit_blend_weights(mp2, W0)
    g2 = apply_shrink(blend(mp2, Wt2), wshr, task2, W0)
    seg1 = smooth(bl, sm); seg2 = smooth(g2, sm)
    e = []
    for g in TGT:
        for p in (seg1, seg2):
            t = W0['T_'+g].reindex(p[g].index); ok = t.notna(); e.append((p[g][ok]-t[ok]).values)
    tick('VALIDATION total RMSE %.4f  (seg1 %.4f / seg2 %.4f)'
         % (float(np.sqrt(np.mean(np.concatenate(e)**2))), rmse_of(seg1, W0), rmse_of(seg2, W0)))
    if a.sim: return

    # ── 최종 : 배포 자료 전체로 재학습 후 테스트 구간 예측 ──
    te = pd.read_csv(os.path.join(a.data, 'test_index.csv')); te['t'] = pd.to_datetime(te.time)
    tt = pd.DatetimeIndex(sorted(te.t.unique()))
    has19 = W0f['T_d19'].reindex(tt).notna()
    m1 = pd.Series(False, index=idx); m1[tt[has19.values]] = True
    m2 = pd.Series(False, index=idx); m2[tt[~has19.values]] = True
    tick('test: with layer5 %d / without %d' % (m1.sum(), m2.sum()))

    p1 = blend(fit_predict(X1, task1, season, m1, a.seeds, a.jobs), Wt); tick('seg1 predicted')
    p2 = apply_shrink(blend(fit_predict(X2, task2, season, m2, a.seeds, a.jobs), Wt2),
                      wshr, task2, W0);                                   tick('seg2 predicted')
    pred = {g: smooth({g: pd.concat([p1[g], p2[g]]).sort_index()}, sm)[g] for g in TGT}

    rows = [pd.DataFrame({'layer': LAY[g], 't': pred[g].index, 'temp': pred[g].values}) for g in TGT]
    out = te[['station','layer','time','t']].merge(pd.concat(rows), on=['layer','t'], how='left')
    top = W0['T_d04'].interpolate(limit=144, limit_direction='both')
    m = out.temp.isna()
    if m.any(): out.loc[m,'temp'] = top.reindex(pd.DatetimeIndex(out.loc[m,'t'])).values
    m = out.temp.isna()
    if m.any(): out.loc[m,'temp'] = top.ffill().bfill().reindex(pd.DatetimeIndex(out.loc[m,'t'])).values
    out = out[['station','layer','time','temp']]; out['temp'] = out.temp.round(4)
    assert len(out) == len(te) and out.temp.notna().all() and out.temp.between(-5, 45).all()
    out.to_csv(a.out, index=False)
    tick('wrote %s (%d rows)' % (a.out, len(out)))

if __name__ == '__main__':
    main()
