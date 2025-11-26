import numpy as np

def stable_score(my_reward: float,
                 reward_random: float,
                 reward_benchmark: float,
                 min_gap_ratio: float = 0.05,
                 max_score: float = 2.0) -> float:
    """
    数学上更稳定的评分函数：
    - 以随机策略为 0 参考点
    - 基准策略附近 ~ 1
    - 超越基准时平滑上升，最多趋近 max_score (默认 2.0)
    """

    baseline = float(reward_random)

    # ---------- 1️⃣ 计算 '有效 gap'，防止分母过小 ----------
    if reward_benchmark > baseline:
        raw_gap = float(reward_benchmark - baseline)
        # 最小 gap：防止 benchmark ≈ random 时爆炸
        min_gap = max(min_gap_ratio * abs(reward_benchmark), 1.0)
        gap = max(raw_gap, min_gap)
    else:
        # 基准不如随机或几乎相等：只能退化成“相对随机收益”
        # 用 baseline 的量级来归一化，避免除以很小的数
        gap = max(abs(baseline), 1.0)

    # ---------- 2️⃣ 相对位置 r ----------
    r = (my_reward - baseline) / gap  # 可能 <0，也可能 >1

    # ---------- 3️⃣ 将 r 映射为 [0, max_score) 内的平滑得分 ----------
    if r <= 0:
        # 比随机还差，直接给 0，下限裁掉，避免负分和极端尾部
        return 0.0

    if r <= 1:
        # 在 [随机, 基准] 之间，保持线性：解释性最强
        # r=0 -> 0, r=1 -> 1
        return r

    # r > 1：超越基准，采用平滑饱和映射
    # 1 处连续且可导，向右渐进 max_score
    # beta 控制上升速度，1.0 意味着在 r≈2 时差不多接近上限
    beta = 1.0
    return 1.0 + (max_score - 1.0) * (1.0 - np.exp(-beta * (r - 1.0)))

def original_score(my_reward, reward_random, reward_benchmark):
    if reward_benchmark > reward_random and reward_benchmark > my_reward:
        return (my_reward - reward_random) / (reward_benchmark - reward_random)
    else:
        return my_reward / reward_benchmark

import numpy as np

def check_unstable(scores: np.ndarray) -> bool:
    """
    稳定性判定函数（替代 scale/mean > 0.5）

    返回 True  -> 不稳定
    返回 False -> 稳定
    """

    scores = np.asarray(scores, dtype=float)

    if len(scores) < 20:
        # 样本太少，直接认为稳定（防误判）
        return False

    mean = scores.mean()
    std  = scores.std()
    p10  = np.percentile(scores, 10)
    p50  = np.percentile(scores, 50)
    min_score = scores.min()

    # -------- ① 极端厚尾波动 --------
    if std > 0.5:
        print(f'极端厚尾波动: {std}')
        return True

    # -------- ② 明显下侧翻车风险 --------
    # 至少有 10% 的局面低于 0.5，说明存在结构性“爆亏局”
    if p10 < 0.5 or min_score < 0.1:
        print(f'明显下侧翻车风险: {p10}')
        return True

    # -------- ③ 低均值 + 高波动 = 赌博型策略 --------
    if mean < 0.95 and std > 0.25:
        print(f'低均值 + 高波动: {mean} {std}')
        return True

    # -------- ④ 中位数长期低于 0.9（大多数局都不行）--------
    if p50 < 0.90:
        print(f'中位数长期低于 0.9: {p50}')
        return True

    return False

if __name__ == '__main__':
    # main.py
    import numpy as np
    import matplotlib.pyplot as plt


    reward_random = 500
    reward_benchmark = 1000

    my_rewards = np.linspace(300, 1600, 300)

    original_scores = [
        original_score(x, reward_random, reward_benchmark)
        for x in my_rewards
    ]

    stable_scores = [
        stable_score(x, reward_random, reward_benchmark)
        for x in my_rewards
    ]

    plt.figure()
    plt.plot(my_rewards, original_scores, label="Original Score")
    plt.plot(my_rewards, stable_scores, label="Stable Score")
    plt.xlabel("My Reward")
    plt.ylabel("Score")
    plt.title("Original Score vs Stable Score")
    plt.legend()
    plt.grid(True)
    plt.show()
