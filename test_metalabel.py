"""Test script to debug meta-labeling training flow."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from strategies.metalabel.adapter import MetaLabelStrategy
from strategies.metalabel.params import PrimarySignalParams, TripleBarrierParams, MetaModelParams
from strategies.metalabel.primary import generate_primary_signals, get_signal_bars
from strategies.metalabel.labels import build_labeled_dataset
from strategies.metalabel.features import build_meta_features
from strategies.metalabel.model import train_meta_model
from data_pipeline.fetcher import fetch_ohlcv_data
import tempfile

# Load sample data
print("Loading data...")
data = fetch_ohlcv_data('BTC/USDT', '1m', limit=50000)
print(f'Data shape: {data.shape}')

# Generate signals
print("Generating primary signals...")
params = PrimarySignalParams()
signal_df = generate_primary_signals(data, params)
signals_only = get_signal_bars(signal_df)
print(f'Total signals: {len(signals_only)}')

# Build labels
print("Building labels...")
barrier_params = TripleBarrierParams()
labeled = build_labeled_dataset(signal_df, signals_only, barrier_params)
print(f'Labeled samples: {len(labeled)}')
if not labeled.empty:
    print(f'Pos ratio: {labeled["meta_label"].mean():.3f}')
    print(f'Columns: {labeled.columns.tolist()}')

# Build features
print("Building meta-features...")
meta_params = MetaModelParams()
features = build_meta_features(signal_df, meta_params)
print(f'Features shape: {features.shape}')
print(f'Feature columns: {features.columns.tolist()}')

# Try training
if len(labeled) >= 50:
    print("Training meta-model...")
    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_meta_model(
            labeled_df=labeled,
            full_df=signal_df,
            params=meta_params,
            model_dir=tmpdir,
            seed=42,
        )
        print(f'Training complete!')
        print(f'Validation metrics: {result.validation_metrics}')
else:
    print(f"Not enough labeled samples ({len(labeled)}) to train")

print("All done!")
