from pathlib import Path

from PIL import Image


def _sanitize_name(name):
    safe_chars = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_"):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_") or "unknown"


def save_misclassified_examples(image_paths, y_true, y_pred, class_names, output_dir, max_examples=None):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for image_path, true_idx, pred_idx in zip(image_paths, y_true, y_pred):
        true_idx = int(true_idx)
        pred_idx = int(pred_idx)
        if true_idx == pred_idx:
            continue

        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]
        source_path = Path(image_path)
        target_dir = output_root / f"true_{_sanitize_name(true_name)}" / f"pred_{_sanitize_name(pred_name)}"
        target_dir.mkdir(parents=True, exist_ok=True)

        target_name = (
            f"{saved_count + 1:04d}__true-{_sanitize_name(true_name)}__"
            f"pred-{_sanitize_name(pred_name)}__{source_path.stem}.png"
        )
        target_path = target_dir / target_name

        with Image.open(source_path) as image:
            image.convert("RGB").save(target_path)

        saved_count += 1
        if max_examples is not None and saved_count >= max_examples:
            break

    print(f"Saved {saved_count} misclassified example(s) to: {output_root}")
    return saved_count