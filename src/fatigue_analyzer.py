"""多级疲劳评分 + 5 级分类（带迟滞）。"""
import config
from src.metrics_buffer import MetricsBuffer


class FatigueAnalyzer:
    def __init__(self):
        self._current_level = 0

    def analyze(self, snapshot: dict):
        """分析缓冲区快照，返回 FatigueAnalysis。"""
        # 子分数
        blink_rate = MetricsBuffer.compute_blink_rate(snapshot)
        avg_blink_dur = MetricsBuffer.compute_avg_blink_duration(snapshot)
        microsleep_count = MetricsBuffer.count_microsleeps(snapshot)
        yawn_count = MetricsBuffer.count_yawns(snapshot)
        head_nod = MetricsBuffer.compute_head_nod_intensity(snapshot)
        ear_drift = MetricsBuffer.compute_ear_drift(snapshot)
        perclos = MetricsBuffer.compute_perclos(snapshot)

        # 各因子得分
        sub = {}

        # 1. 眨眼频率 (基准 12/min，2.5x = 30/min 得满分)
        sub["blink_rate"] = self._score_blink_rate(blink_rate)
        # 2. 微睡眠次数 (0=0, 3+=满分)
        sub["microsleep"] = min(microsleep_count / 3.0, 1.0) * 100 * config.WEIGHT_MICROSLEEP
        # 3. 平均眨眼时长 (150ms=0, 400ms+=满分)
        sub["blink_duration"] = self._score_blink_duration(avg_blink_dur)
        # 4. 头部点头强度
        sub["head_nod"] = min(head_nod / 50.0, 1.0) * 100 * config.WEIGHT_HEAD_NOD
        # 5. 打哈欠频率
        sub["yawn"] = min(yawn_count / 3.0, 1.0) * 100 * config.WEIGHT_YAWN
        # 6. EAR 漂移 (20%下降=满分)
        sub["ear_drift"] = min(ear_drift / 0.2, 1.0) * 100 * config.WEIGHT_EAR_DRIFT
        # 7. PERCLOS 闭眼占比 (20% 以上逐步认为风险升高)
        sub["perclos"] = min(perclos / 0.2, 1.0) * 100 * config.WEIGHT_PERCLOS

        # 总分
        score = sum(sub.values())

        # 带迟滞的等级判定
        level = self._classify_with_hysteresis(score)

        # 提取显著因素
        factors = self._identify_factors(sub, snapshot)

        from src.metrics_buffer import FatigueAnalysis
        return FatigueAnalysis(level=level, score=score, sub_scores=sub, factors=factors)

    def _score_blink_rate(self, blink_rate):
        """将眨眼频率映射为得分。正常 12/min，2.5x=30/min 得满分。"""
        delta = max(0, blink_rate - 12)
        factor = min(delta / 18.0, 1.0)  # 18 额外眨眼 = 满分
        return factor * 100 * config.WEIGHT_BLINK_RATE

    def _score_blink_duration(self, avg_duration):
        """将平均眨眼时长映射为得分。150ms=0, 400ms+=满分。"""
        if avg_duration <= 0.15:
            return 0
        factor = min((avg_duration - 0.15) / 0.25, 1.0)
        return factor * 100 * config.WEIGHT_BLINK_DURATION

    def _classify_with_hysteresis(self, score):
        """带迟滞的等级判定。"""
        levels = config.FATIGUE_LEVELS

        # 向上突破：分数超过某级的 enter 阈值就进入该级
        new_level = self._current_level
        for lv in levels:
            if score >= lv["enter"]:
                new_level = lv["level"]

        # 向下突破：用新等级的 exit 阈值判断是否回退
        if new_level > 0 and score <= levels[new_level]["exit"]:
            for lv in reversed(levels):
                if score <= lv["exit"] and lv["level"] < new_level:
                    new_level = lv["level"]
                    break

        self._current_level = new_level
        return new_level

    @staticmethod
    def _identify_factors(sub_scores, snapshot):
        """识别当前最显著的疲劳因素（用于 LLM 生成个性化对话）。"""
        factors = []
        thresholds = {
            "blink_rate": (15, "眨眼频率升高"),
            "microsleep": (8, "微睡眠事件"),
            "blink_duration": (5, "眨眼时间变长"),
            "head_nod": (5, "头部晃动增加"),
            "yawn": (3, "频繁打哈欠"),
            "ear_drift": (2, "眼睑下垂"),
            "perclos": (3, "闭眼占比升高"),
        }
        for key, (thresh, label) in thresholds.items():
            if sub_scores.get(key, 0) > thresh:
                factors.append(label)
        return factors if factors else ["无明显疲劳信号"]
