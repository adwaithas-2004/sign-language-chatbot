"""The BiGRU sign classifier: building, saving and loading it, and turning a sign into words"""
import json
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np

from isl.features import FEATURE_VERSION, NUM_FEATURES, SEQUENCE_LENGTH

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "isl50_bigru.keras"
LABELS_PATH = MODELS_DIR / "labels.json"


def build_model(num_classes, sequence_length=SEQUENCE_LENGTH, num_features=NUM_FEATURES):
    layers = keras.layers
    model = keras.Sequential([
        keras.Input((sequence_length, num_features)),
        layers.TimeDistributed(layers.Dense(128, activation="relu")),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.GRU(128, return_sequences=True)),
        layers.Bidirectional(layers.GRU(64)),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def save_labels(path, keys, displays):
    Path(path).write_text(json.dumps({"feature_version": FEATURE_VERSION, "sequence_length": SEQUENCE_LENGTH,
                                      "num_features": NUM_FEATURES, "keys": list(keys),
                                      "display": list(displays)}, indent=2), encoding="utf-8")


@dataclass
class Recogniser:
    model: keras.Model
    keys: list
    display: list  # word for the interpreter, same order as keys

    def predict(self, sequence, top_k=3):
        """[(word, probability)] for one (32, F) sign, most likely first"""
        # predict_on_batch runs a compiled graph: ~3 ms, where calling the model eagerly took ~200+ ms
        probabilities = np.asarray(self.model.predict_on_batch(np.asarray(sequence, np.float32)[None]))[0]
        best = np.argsort(probabilities)[::-1][:top_k]
        return [(self.display[i], float(probabilities[i])) for i in best]


def load_recogniser(model_path=MODEL_PATH, labels_path=LABELS_PATH):
    model_path, labels_path = Path(model_path), Path(labels_path)
    if not model_path.exists() or not labels_path.exists():
        raise SystemExit(f"No trained model found ({model_path.name}). Run: python scripts/train.py "
                         "(or pull the trained model from GitHub)")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    expected = {"feature_version": FEATURE_VERSION, "sequence_length": SEQUENCE_LENGTH, "num_features": NUM_FEATURES}
    for name, value in expected.items():
        if labels.get(name) != value:
            raise SystemExit(f"{labels_path.name} was made with {name}={labels.get(name)}, but isl/features.py "
                             f"uses {value}. Retrain with: python scripts/train.py")
    # Only used for predictions, so the optimizer state isn't needed
    return Recogniser(keras.models.load_model(model_path, compile=False), labels["keys"], labels["display"])
