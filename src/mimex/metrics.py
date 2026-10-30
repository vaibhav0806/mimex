import math


def calibrate(scores, labels, target_fpr=0.01):
    if not 0 <= target_fpr < 1:
        raise ValueError("Target FPR must be in [0, 1).")
    humans = sorted(
        (s for s, label in zip(scores, labels, strict=True) if label == 0), reverse=True
    )
    if not humans:
        raise ValueError("Calibration requires human examples.")
    allowed = math.floor(len(humans) * target_fpr)
    # Strictly above this order statistic also handles tied scores conservatively.
    return math.nextafter(humans[allowed], math.inf)


def wilson(k, n):
    if n == 0:
        return None
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0, center - width), min(1, center + width)]


def evaluate_scores(scores, labels, threshold):
    if not scores or len(scores) != len(labels):
        raise ValueError("Scores and labels must be nonempty and equally sized.")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("Scores must be finite and in [0, 1].")
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for score, label in zip(scores, labels, strict=True):
        if label not in (0, 1):
            raise ValueError("Labels must be 0 or 1.")
        key = (
            ("tp" if label else "fp")
            if score >= threshold
            else ("fn" if label else "tn")
        )
        counts[key] += 1
    human, ai = counts["tn"] + counts["fp"], counts["tp"] + counts["fn"]
    return {
        **counts,
        "examples": len(scores),
        "threshold": threshold,
        "accuracy": (counts["tp"] + counts["tn"]) / len(scores),
        "false_positive_rate": counts["fp"] / human if human else None,
        "ai_detection_rate": counts["tp"] / ai if ai else None,
        "ai_detection_95pct_wilson_interval": wilson(counts["tp"], ai),
        "fpr_95pct_wilson_interval": wilson(counts["fp"], human),
    }
