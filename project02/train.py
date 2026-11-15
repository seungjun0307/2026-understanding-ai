import time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# 스트레이트 판정용 10개 윈도우 (Ace=1 로우 스트레이트 및 Ace=14 하이 스트레이트 포함)
_STRAIGHT_WINDOWS = [set(range(s, s + 5)) for s in range(1, 10)] + [{1, 10, 11, 12, 13}]
_ROYAL_SET = {1, 10, 11, 12, 13}
_RANK_FEAT_CACHE = {}


def _get_rank_features(ranks_tuple, k):
    """정렬된 숫자 튜플로부터 숫자 기반 도메인 피처 10개를 계산 및 캐싱합니다."""
    key = (k, ranks_tuple)
    if key in _RANK_FEAT_CACHE:
        return _RANK_FEAT_CACHE[key]

    rank_counts = {}
    for r in ranks_tuple:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    freq_sorted = sorted(rank_counts.values(), reverse=True)

    max_freq = freq_sorted[0]
    second_freq = freq_sorted[1] if len(freq_sorted) > 1 else 0
    num_unique_ranks = len(freq_sorted)
    num_pairs = sum(1 for c in freq_sorted if c >= 2)

    if max_freq == 4:
        current_score = 7
    elif max_freq == 3:
        current_score = 3
    elif max_freq == 2 and second_freq == 2:
        current_score = 2
    elif max_freq == 2:
        current_score = 1
    else:
        current_score = 0

    span_low = ranks_tuple[-1] - ranks_tuple[0]
    ranks_high = sorted([14 if r == 1 else r for r in ranks_tuple])
    span_high = ranks_high[-1] - ranks_high[0]
    min_span = min(span_low, span_high)

    straight_windows = 0
    rank_set = set(ranks_tuple)
    if num_unique_ranks == k:
        for w in _STRAIGHT_WINDOWS:
            if rank_set.issubset(w):
                straight_windows += 1

    in_royal = 1 if (num_unique_ranks == k and rank_set.issubset(_ROYAL_SET)) else 0

    res = (
        max_freq,
        second_freq,
        num_unique_ranks,
        num_pairs,
        current_score,
        span_low,
        span_high,
        min_span,
        straight_windows,
        in_royal,
    )
    _RANK_FEAT_CACHE[key] = res
    return res


def extract_features_batch(suits_arr, ranks_arr):
    """
    배열(N, k)에서 중복 rank 조합을 묶어 poker.py의 extract_features_from_hand와
    동일한 피처 행렬(N, k + 13)을 추출합니다.
    """
    _, k = ranks_arr.shape
    sorted_ranks = np.sort(ranks_arr, axis=1)

    unique_ranks, inv_idx = np.unique(sorted_ranks, axis=0, return_inverse=True)
    rank_stats = np.zeros((len(unique_ranks), 10), dtype=np.float32)
    for idx, r_row in enumerate(unique_ranks):
        r_tuple = tuple(int(x) for x in r_row)
        rank_stats[idx] = _get_rank_features(r_tuple, k)

    mapped_stats = rank_stats[inv_idx]

    s1 = np.sum(suits_arr == 1, axis=1)
    s2 = np.sum(suits_arr == 2, axis=1)
    s3 = np.sum(suits_arr == 3, axis=1)
    s4 = np.sum(suits_arr == 4, axis=1)
    max_suit = np.maximum(np.maximum(s1, s2), np.maximum(s3, s4)).astype(np.float32)
    is_all_same_suit = (max_suit == k).astype(np.float32)

    straight_windows = mapped_stats[:, 8]
    in_royal = mapped_stats[:, 9]

    is_sf_draw = ((is_all_same_suit == 1) & (straight_windows > 0)).astype(np.float32)
    is_royal_draw = ((is_all_same_suit == 1) & (in_royal == 1)).astype(np.float32)

    out = np.column_stack([
        sorted_ranks.astype(np.float32),
        mapped_stats[:, 0:5],
        max_suit,
        is_all_same_suit,
        mapped_stats[:, 5:9],
        is_sf_draw,
        is_royal_draw,
    ])
    return out.astype(np.float32)


def optimize_joint_thresholds(model3, model4, val_rows):
    """
    검증용 자연 데이터셋에서 poker.py의 하드 룰과 결합된
    (th3, th4) 2차원 베팅 커트라인을 상대 점수(1~9점)별로 탐색합니다.
    """
    X3_val = extract_features_batch(val_rows[:, [0, 2, 4]], val_rows[:, [1, 3, 5]])
    X4_val = extract_features_batch(val_rows[:, [0, 2, 4, 6]], val_rows[:, [1, 3, 5, 7]])
    y_val = val_rows[:, -1].astype(np.int64)

    probs3_val = model3.predict_proba(X3_val)
    probs4_val = model4.predict_proba(X4_val)
    classes3 = model3.classes_
    classes4 = model4.classes_

    cur3, suit3, str3 = X3_val[:, 7], X3_val[:, 9], X3_val[:, 13]
    cur4, suit4, str4 = X4_val[:, 8], X4_val[:, 10], X4_val[:, 14]

    t3_grid = np.linspace(-2.0, 0.2, 45)
    t4_grid = np.linspace(-2.3, 0.2, 51)

    best_th3 = {}
    best_th4 = {}

    for oppo in range(1, 10):
        p_win3 = probs3_val[:, classes3 > oppo].sum(axis=1)
        p_lose3 = probs3_val[:, classes3 < oppo].sum(axis=1)
        ev3 = 3.0 * (p_win3 - p_lose3)

        p_win4 = probs4_val[:, classes4 > oppo].sum(axis=1)
        p_lose4 = probs4_val[:, classes4 < oppo].sum(axis=1)
        ev4 = 3.0 * (p_win4 - p_lose4)

        payoff5 = np.where(y_val > oppo, 3.0, np.where(y_val < oppo, -3.0, 0.0))

        force_call3 = (cur3 >= oppo)
        force_fold3 = (
            ((cur3 == 0) & (suit3 == 0) & (str3 == 0)) |
            ((cur3 == 1) & (oppo >= 8))
        ) & (~force_call3)

        force_call4 = (cur4 >= oppo)
        force_fold4 = (
            ((cur4 == 0) & (suit4 == 0) & (str4 == 0)) |
            ((cur4 == 1) & (oppo >= 4)) |
            ((cur4 == 2) & (oppo >= 3)) |
            ((cur4 == 3) & (oppo >= 8))
        ) & (~force_call4)

        call4_mat = force_call4[None, :] | ((~force_fold4)[None, :] & (ev4[None, :] >= t4_grid[:, None]))
        reward4_mat = np.where(call4_mat, payoff5[None, :], -2.0)

        call3_mat = force_call3[None, :] | ((~force_fold3)[None, :] & (ev3[None, :] >= t3_grid[:, None]))

        # 임계값 조합별 평균 손익을 float64 정밀도로 계산
        scores_mat = -1.0 + (call3_mat.astype(np.float64) @ (reward4_mat.astype(np.float64) + 1.0).T) / len(y_val)
        max_val = np.max(scores_mat)

        # 최고 검증 수익과 유사한 후보 중 손절 비용(-1.0, -2.0)에 가장 가까운 기준값 선택
        cand_indices = np.argwhere(np.isclose(scores_mat, max_val, atol=1e-4))
        best_i, best_j = 0, 0
        best_dist = 1e9
        for i_idx, j_idx in cand_indices:
            dist = (t3_grid[i_idx] - (-1.0)) ** 2 + (t4_grid[j_idx] - (-2.0)) ** 2
            if dist < best_dist:
                best_dist = dist
                best_i, best_j = i_idx, j_idx

        best_th3[oppo] = float(round(t3_grid[best_i], 4))
        best_th4[oppo] = float(round(t4_grid[best_j], 4))

    return best_th3, best_th4


def main():
    t_start = time.time()
    train_csv = 'poker-hand-training.csv'
    print(f"[1/4] Loading training dataset from '{train_csv}'...")
    df = pd.read_csv(train_csv)
    raw_all = df.to_numpy()
    print(f"      Loaded {len(raw_all):,} rows.")

    # 앞쪽 10행이 모두 8~9점(인위적으로 삽입된 로열/스트레이트 플러시)인 경우에만 가중치를 낮추고,
    # 나머지 자연 분포 데이터를 학습(85%) / 검증(15%)으로 분할합니다.
    has_prepended_rare = (len(raw_all) > 10) and np.all(raw_all[:10, -1] >= 8)
    if has_prepended_rare:
        rare_rows = raw_all[:10]
        natural_rows = raw_all[10:]
    else:
        rare_rows = raw_all[:0]
        natural_rows = raw_all

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(natural_rows))
    val_size = int(len(natural_rows) * 0.15)
    val_rows = natural_rows[perm[:val_size]]
    train_nat = natural_rows[perm[val_size:]]

    train_split = np.vstack([rare_rows, train_nat])
    split_weights = np.ones(len(train_split), dtype=np.float32)
    if has_prepended_rare:
        split_weights[:10] = 0.05

    print("[2/4] Extracting domain features & generating two card subsets per hand...")
    # 각 5장 패에서 두 종류의 부분 패를 추출하여 학습 데이터 구성
    s3_a, r3_a = train_split[:, [0, 2, 4]], train_split[:, [1, 3, 5]]
    s3_b, r3_b = train_split[:, [4, 6, 8]], train_split[:, [5, 7, 9]]
    X3_tr = extract_features_batch(np.vstack([s3_a, s3_b]), np.vstack([r3_a, r3_b]))
    y3_tr = np.concatenate([train_split[:, -1], train_split[:, -1]]).astype(np.int64)
    w3_tr = np.concatenate([split_weights, split_weights])

    s4_a, r4_a = train_split[:, [0, 2, 4, 6]], train_split[:, [1, 3, 5, 7]]
    s4_b, r4_b = train_split[:, [2, 4, 6, 8]], train_split[:, [3, 5, 7, 9]]
    X4_tr = extract_features_batch(np.vstack([s4_a, s4_b]), np.vstack([r4_a, r4_b]))
    y4_tr = np.concatenate([train_split[:, -1], train_split[:, -1]]).astype(np.int64)
    w4_tr = np.concatenate([split_weights, split_weights])

    print("[3/4] Training 3-card & 4-card HistGradientBoosting models and optimizing EV thresholds...")
    val_model3 = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=0.08, max_depth=6, min_samples_leaf=30, random_state=42
    )
    val_model3.fit(X3_tr, y3_tr, sample_weight=w3_tr)

    val_model4 = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=0.08, max_depth=6, min_samples_leaf=30, random_state=42
    )
    val_model4.fit(X4_tr, y4_tr, sample_weight=w4_tr)

    best_th3, best_th4 = optimize_joint_thresholds(val_model3, val_model4, val_rows)
    print(f"      Selected 3-card thresholds: {best_th3}")
    print(f"      Selected 4-card thresholds: {best_th4}")

    print("[4/4] Fitting final models on full dataset & saving to disk...")
    full_weights = np.ones(len(raw_all), dtype=np.float32)
    if has_prepended_rare:
        full_weights[:10] = 0.05

    s3_fa, r3_fa = raw_all[:, [0, 2, 4]], raw_all[:, [1, 3, 5]]
    s3_fb, r3_fb = raw_all[:, [4, 6, 8]], raw_all[:, [5, 7, 9]]
    X3_full = extract_features_batch(np.vstack([s3_fa, s3_fb]), np.vstack([r3_fa, r3_fb]))
    y3_full = np.concatenate([raw_all[:, -1], raw_all[:, -1]]).astype(np.int64)
    w3_full = np.concatenate([full_weights, full_weights])

    s4_fa, r4_fa = raw_all[:, [0, 2, 4, 6]], raw_all[:, [1, 3, 5, 7]]
    s4_fb, r4_fb = raw_all[:, [2, 4, 6, 8]], raw_all[:, [3, 5, 7, 9]]
    X4_full = extract_features_batch(np.vstack([s4_fa, s4_fb]), np.vstack([r4_fa, r4_fb]))
    y4_full = np.concatenate([raw_all[:, -1], raw_all[:, -1]]).astype(np.int64)
    w4_full = np.concatenate([full_weights, full_weights])

    model3 = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=0.08, max_depth=6, min_samples_leaf=30, random_state=42
    )
    model3.fit(X3_full, y3_full, sample_weight=w3_full)
    model3.best_thresholds_ = best_th3

    model4 = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=0.08, max_depth=6, min_samples_leaf=30, random_state=42
    )
    model4.fit(X4_full, y4_full, sample_weight=w4_full)
    model4.best_thresholds_ = best_th4

    with open('poker-model3.sav', 'wb') as f3:
        pickle.dump(model3, f3)
    with open('poker-model4.sav', 'wb') as f4:
        pickle.dump(model4, f4)

    print(
        f"Done in {time.time() - t_start:.2f}s! "
        f"(Iterations: model3={model3.n_iter_}, model4={model4.n_iter_}) "
        f"Saved 'poker-model3.sav' and 'poker-model4.sav'."
    )


if __name__ == '__main__':
    main()