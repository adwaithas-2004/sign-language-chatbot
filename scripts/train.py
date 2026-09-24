"""Train and evaluate the INCLUDE-50 sign classifiers, then write the model and reports.

Run from the project folder after scripts/prepare_include50.py:  python scripts/train.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import keras  # noqa: E402

from isl import include_data  # noqa: E402
from isl.augment import augment, crop_view, random_view  # noqa: E402
from isl.features import NUM_FEATURES, raised_part, to_sequence  # noqa: E402
from isl.model import Recogniser, build_model, save_labels  # noqa: E402

LANDMARKS_DIR = include_data.DATA_DIR / "landmarks"
PAPER_TOP1 = 0.945  # best INCLUDE-50 result reported in the INCLUDE paper
SEED = 42
BATCH_SIZE = 32
PATIENCE = 25
# Validation also sees each clip through closer cameras (how far below the shoulders each one sees), so the saved
# run is the one that copes best with both full and desk-webcam views
VAL_VIEWS = (0.6, 1.0)
VAL_VIEW_TOP, VAL_VIEW_HALF_WIDTH = -1.5, 1.5


def load_clips(landmarks_dir, keys):
    """{split: (list of whole (T, F) clips, labels)} from the prepared .npz files"""
    index = {key: i for i, key in enumerate(keys)}
    clips = {}
    for split in ("train", "val", "test"):
        frames, labels = [], []
        for path in sorted((Path(landmarks_dir) / split).glob("*/*.npz")):
            with np.load(path) as data:
                features = data["features"]
            frames.append(features)
            labels.append(index[path.parent.name])
        clips[split] = (frames, np.array(labels, dtype=np.int64))
    return clips


def synthetic_clips(rng, num_classes=3, per_split=(12, 4, 4)):
    """A tiny made-up dataset where each class has its own movement, for --smoke runs"""
    clips = {}
    for split, count in zip(("train", "val", "test"), per_split):
        frames, labels = [], []
        for label in range(num_classes):
            for _ in range(count):
                length = int(rng.integers(20, 40))
                clip = rng.normal(0, 0.05, (length, NUM_FEATURES)).astype(np.float32)
                clip[:, label] += np.linspace(0, 1, length, dtype=np.float32)
                frames.append(clip)
                labels.append(label)
        clips[split] = (frames, np.array(labels, dtype=np.int64))
    return clips


def summary_features(X):
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)


def train_baseline(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(summary_features(X), y)


def baseline_probabilities(baseline, X, num_classes):
    probabilities = np.zeros((len(X), num_classes))
    probabilities[:, baseline.classes_] = baseline.predict_proba(summary_features(X))
    return probabilities


def predict(model, X):
    return np.asarray(model.predict(X, batch_size=64, verbose=0))


def training_sequence(clip, rng):
    """One model input for training: the whole clip seen by a random camera, trimmed like the live app, then varied"""
    return augment(raised_part(random_view(clip, rng)), rng)


def validation_set(clips, labels):
    """Validation inputs for the full view followed by each closer view in VAL_VIEWS"""
    views = [clips] + [[crop_view(clip, bottom, VAL_VIEW_TOP, VAL_VIEW_HALF_WIDTH) for clip in clips]
                       for bottom in VAL_VIEWS]
    return np.stack([to_sequence(clip) for view in views for clip in view]), np.tile(labels, len(views))


def train_bigru(train_clips, y_train, X_val, y_val, num_classes, epochs, rng):
    """Fresh augmentations every epoch; keeps the weights with the best validation accuracy"""
    model = build_model(num_classes)
    best_accuracy, best_weights, best_epoch, since_best = -1.0, model.get_weights(), 0, 0
    for epoch in range(1, epochs + 1):
        X = np.stack([training_sequence(clip, rng) for clip in train_clips])
        history = model.fit(X, y_train, batch_size=BATCH_SIZE, epochs=1, shuffle=True, verbose=0)
        val_accuracy = float(np.mean(predict(model, X_val).argmax(1) == y_val))
        if val_accuracy > best_accuracy:
            best_accuracy, best_weights, best_epoch, since_best = val_accuracy, model.get_weights(), epoch, 0
        else:
            since_best += 1
        print(f"epoch {epoch:3d}  loss {history.history['loss'][0]:.3f}  val acc {val_accuracy:.3f}  "
              f"best {best_accuracy:.3f}", flush=True)
        if since_best >= PATIENCE:
            break
    model.set_weights(best_weights)
    return model, best_epoch, best_accuracy


def metrics(probabilities, y, keys):
    """Top-1/top-3 accuracy, macro-F1 and per-word accuracy over the words present in y"""
    from sklearn.metrics import f1_score
    predicted = probabilities.argmax(1)
    top3 = np.argsort(probabilities, axis=1)[:, ::-1][:, :3]
    present = np.unique(y)
    return {
        "top1": float(np.mean(predicted == y)),
        "top3": float(np.mean([label in row for label, row in zip(y, top3)])),
        "macro_f1": float(f1_score(y, predicted, labels=present, average="macro", zero_division=0)),
        "per_word": {keys[i]: float(np.mean(predicted[y == i] == i)) for i in present},
    }


def summarise_runs(runs):
    """Mean and spread (sample standard deviation) of each metric over several training runs"""
    summary = {}
    for name in ("top1", "top3", "macro_f1"):
        values = np.array([run[name] for run in runs])
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


def save_confusion_matrix(probabilities, y, words, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y, probabilities.argmax(1), labels=list(range(len(words))))
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(words)), words, rotation=90, fontsize=7)
    ax.set_yticks(range(len(words)), words, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("INCLUDE-50 test split: BiGRU confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def results_markdown(report):
    baseline, bigru, counts, seeds = report["baseline"], report["bigru"], report["counts"], report["bigru_seeds"]
    runs = len(seeds["runs"])
    hardest = sorted(bigru["per_word"].items(), key=lambda item: item[1])[:5]
    gap = seeds["top1_mean"] - baseline["top1"]
    return "\n".join([
        f"Official INCLUDE-50 test split: {counts['test']} videos "
        f"(trained on {counts['train']}, validated on {counts['val']}).",
        "",
        "| Model | Top-1 | Top-3 | Macro-F1 |",
        "|---|---|---|---|",
        f"| Baseline: logistic regression on summary features (deterministic) | {baseline['top1']:.1%} "
        f"| {baseline['top3']:.1%} | {baseline['macro_f1']:.3f} |",
        f"| BiGRU on landmark sequences, mean ± std over {runs} training runs "
        f"| {seeds['top1_mean']:.1%} ± {seeds['top1_std'] * 100:.1f} | {seeds['top3_mean']:.1%} ± "
        f"{seeds['top3_std'] * 100:.1f} | {seeds['macro_f1_mean']:.3f} ± {seeds['macro_f1_std']:.3f} |",
        f"| **BiGRU, saved model** (best validation accuracy of the {runs} runs) | **{bigru['top1']:.1%}** "
        f"| {bigru['top3']:.1%} | {bigru['macro_f1']:.3f} |",
        f"| INCLUDE paper, best model on INCLUDE-50 | {report['paper_top1']:.1%} | – | – |",
        "",
        f"Averaged over {runs} runs, the BiGRU scores {gap * 100:+.1f} points top-1 against the baseline "
        f"(run-to-run standard deviation {seeds['top1_std'] * 100:.1f} points).",
        "",
        f"BiGRU inference: {report['inference_ms']:.1f} ms per sign on CPU. "
        f"Hardest words for the saved model: "
        + ", ".join(f"{word} ({accuracy:.0%})" for word, accuracy in hardest) + ".",
        "",
        "INCLUDE's 7 signers appear in every split, so these are *seen-signer* results. "
        "Accuracy for a new signer and camera will be lower.",
        "",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate the INCLUDE-50 sign classifiers.")
    parser.add_argument("--smoke", action="store_true", help="quick run on a tiny synthetic dataset")
    parser.add_argument("--out", type=Path, default=ROOT, help="folder that gets models/ and reports/")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seeds", type=int, default=5, help="train the BiGRU this many times with different seeds")
    args = parser.parse_args(argv)

    keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)
    if args.smoke:
        keys = displays = ["a", "b", "c"]
        clips, epochs = synthetic_clips(rng), 2
    else:
        keys = include_data.load_label_keys()
        names = {s.key: s.display for split in ("train", "val", "test") for s in include_data.load_split(split)}
        displays = [names[key] for key in keys]
        clips, epochs = load_clips(LANDMARKS_DIR, keys), args.epochs
    counts = {split: len(clips[split][1]) for split in clips}
    print("clips per split:", counts)
    if not all(counts.values()):
        raise SystemExit("No prepared landmarks found. Run: python scripts/prepare_include50.py")

    X = {split: np.stack([to_sequence(clip) for clip in clips[split][0]]) for split in clips}
    y = {split: clips[split][1] for split in clips}

    baseline = train_baseline(X["train"], y["train"])  # deterministic, so trained once
    X_val, y_val = validation_set(*clips["val"])

    # One BiGRU run can be lucky or unlucky, so train several and report the spread
    runs, best = [], None
    for seed in range(SEED, SEED + args.seeds):
        keras.utils.set_random_seed(seed)
        model, best_epoch, val_accuracy = train_bigru(clips["train"][0], y["train"], X_val, y_val, len(keys),
                                                      epochs, np.random.default_rng(seed))
        test_probabilities = predict(model, X["test"])
        result = metrics(test_probabilities, y["test"], keys)
        runs.append({"seed": seed, "val_accuracy": val_accuracy, "best_epoch": best_epoch,
                     **{name: result[name] for name in ("top1", "top3", "macro_f1")}})
        print(f"seed {seed}: val acc {val_accuracy:.3f}  test top-1 {result['top1']:.3f}", flush=True)
        if best is None or val_accuracy > best["val_accuracy"]:  # chosen by validation, never by test accuracy
            best = {"model": model, "probabilities": test_probabilities, "seed": seed, "epoch": best_epoch,
                    "val_accuracy": val_accuracy}

    recogniser = Recogniser(best["model"], keys, displays)  # timed the way the live app predicts
    recogniser.predict(X["test"][0])
    start = time.perf_counter()
    for sequence in X["test"][:20]:
        recogniser.predict(sequence)
    inference_ms = (time.perf_counter() - start) / min(20, len(X["test"])) * 1000

    report = {"counts": counts,
              "baseline": metrics(baseline_probabilities(baseline, X["test"], len(keys)), y["test"], keys),
              "bigru": metrics(best["probabilities"], y["test"], keys),
              "bigru_seeds": {**summarise_runs(runs), "runs": runs},
              "saved_seed": best["seed"], "epochs_trained": best["epoch"],
              "inference_ms": inference_ms, "paper_top1": PAPER_TOP1}

    models_dir, reports_dir = args.out / "models", args.out / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    best["model"].save(models_dir / "isl50_bigru.keras")
    save_labels(models_dir / "labels.json", keys, displays)
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_confusion_matrix(best["probabilities"], y["test"], displays, reports_dir / "confusion_matrix.png")
    (reports_dir / "results.md").write_text(results_markdown(report), encoding="utf-8")
    print(results_markdown(report))


if __name__ == "__main__":
    main()
