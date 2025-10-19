import logging
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence

import pandas as pd

Decision = Literal["BUY", "SELL", "HOLD"]

LOGGER = logging.getLogger(__name__)

# 型別別名讓管線中的函式簽名更清楚
FeatureStep = Callable[[pd.DataFrame], pd.DataFrame]
Scorer = Callable[[pd.Series], Optional[Dict[str, Any]]]
Resolver = Callable[..., Decision]


def apply_feature_steps(
    df: pd.DataFrame, feature_steps: Iterable[FeatureStep]
) -> pd.DataFrame:
    """依序套用特徵工程步驟，維持函數式管線的可組合性。"""
    transformed = df.copy()
    for step in feature_steps:
        transformed = step(transformed)
    return transformed


def normalise_evaluation(
    raw_eval: Dict[str, Any], default_weight: float = 1.0
) -> Dict[str, Any]:
    """統一評分結果格式，確保解析器可以正確計算。"""
    evaluation = {
        "name": raw_eval.get("name", "unknown"),
        "decision": raw_eval.get("decision", "HOLD"),
        "confidence": float(raw_eval.get("confidence", 1.0)),
        "weight": float(raw_eval.get("weight", default_weight)),
        "metadata": raw_eval.get("metadata", {}),
    }
    return evaluation


def weighted_vote_resolver(
    evaluations: Sequence[Dict[str, Any]],
    *,
    default_decision: Decision = "HOLD",
) -> Decision:
    """利用加權投票整合不同評分器的結果。"""
    if not evaluations:
        return default_decision
    score_board: Dict[Decision, float] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
    for evaluation in evaluations:
        decision = evaluation.get("decision", default_decision)
        if decision not in score_board:
            LOGGER.warning("Unknown decision %s encountered", decision)
            continue
        score_board[decision] += evaluation.get("weight", 1.0) * evaluation.get(
            "confidence", 0.0
        )
    best_decision = max(score_board.items(), key=lambda item: item[1])[0]
    if score_board[best_decision] == 0:
        return default_decision
    return best_decision


def generate_signal_frame(
    df: pd.DataFrame,
    feature_steps: Iterable[FeatureStep],
    scorers: Sequence[Scorer],
    resolver: Resolver = weighted_vote_resolver,
    *,
    default_decision: Decision = "HOLD",
) -> pd.DataFrame:
    """核心訊號產生流程，統一各種策略的輸出格式。"""
    # 先完成特徵管線，確保後續評分器可以直接使用所需欄位
    enriched_df = apply_feature_steps(df, feature_steps)
    decisions: List[Decision] = []
    diagnostics: List[List[Dict[str, Any]]] = []

    for row_index, (_, row_series) in enumerate(enriched_df.iterrows(), start=1):
        # 收集所有評分器針對當前列的判斷結果
        evaluations: List[Dict[str, Any]] = []
        for scorer in scorers:
            raw_eval = scorer(row_series)
            if raw_eval:
                evaluations.append(normalise_evaluation(raw_eval))
        decision = resolver(evaluations, default_decision=default_decision)
        decisions.append(decision)
        diagnostics.append(evaluations)
        LOGGER.debug("row=%s decision=%s evals=%s", row_index, decision, evaluations)

    output = enriched_df.copy()
    output["signal"] = decisions
    output["evaluations"] = diagnostics
    return output


__all__ = [
    "Decision",
    "FeatureStep",
    "Scorer",
    "Resolver",
    "apply_feature_steps",
    "generate_signal_frame",
    "weighted_vote_resolver",
]
