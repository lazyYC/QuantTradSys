"""star_xgb 策略模組入口。"""
from .params import StarIndicatorParams, StarModelParams
from .features import StarFeatureCache, build_feature_frame
from .labels import build_label_frame
from .dataset import build_training_dataset, split_train_test
from .model import StarTrainingResult, train_star_model
from .runtime import StarRuntimeState, generate_realtime_signal

__all__ = [
    'StarIndicatorParams',
    'StarModelParams',
    'StarFeatureCache',
    'build_feature_frame',
    'build_label_frame',
    'build_training_dataset',
    'split_train_test',
    'StarTrainingResult',
    'train_star_model',
    'StarRuntimeState',
    'generate_realtime_signal',
]
