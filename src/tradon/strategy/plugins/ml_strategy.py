"""ML-based strategy — optional extra (`pip install -e .[ml]`).

A classifier trained offline (see `tradon train-model` / tradon.ml.train)
predicts whether price will be higher `horizon` bars from now; this turns
that probability into a signal. Same no-orders, signal-only rule as every
other strategy — this doesn't retrain live or touch real money, it just
swaps the entry rule from hand-written indicator logic for a learned one.
Needs a model trained and saved first; there is no usable default.
"""
from __future__ import annotations

from tradon.data.base import Candle
from tradon.ml.features import extract_features
from tradon.ml.train import load_model
from tradon.strategy.base import Signal, SignalDirection, Strategy


class MLStrategy(Strategy):
    name = "ml"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        model_path = self.params.get("model_path")
        if not model_path:
            raise ValueError(
                "ml strategy needs params.model_path — train one first with "
                "`tradon train-model --symbol ... --model-path ...`"
            )
        try:
            self._model = load_model(model_path)
        except FileNotFoundError as exc:
            raise ValueError(f"No trained model at {model_path!r} — run `tradon train-model` first") from exc
        try:
            self._up_index = list(self._model.classes_).index(1)
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"{model_path!r} doesn't look like a model trained by `tradon train-model`") from exc

        self.long_threshold = float(self.params.get("long_threshold", 0.55))
        self.short_threshold = float(self.params.get("short_threshold", 0.45))

    def on_candles(self, candles: list[Candle]) -> Signal | None:
        features = extract_features(candles)
        if features is None:
            return None
        probability_up = float(self._model.predict_proba([features])[0][self._up_index])

        if probability_up >= self.long_threshold:
            direction = SignalDirection.LONG
        elif probability_up <= self.short_threshold:
            direction = SignalDirection.SHORT
        else:
            return None

        last = candles[-1]
        strength = abs(probability_up - 0.5) * 2  # 0.5 (the fence) -> 0, 0 or 1 (full certainty) -> 1
        return Signal(
            strategy=self.name,
            symbol=last.symbol,
            direction=direction,
            strength=strength,
            timestamp=last.timestamp,
            reason=f"ML model P(up)={probability_up:.2f} ({direction.value})",
        )
