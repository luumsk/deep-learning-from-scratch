

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "vietnamese_food"


def build_vocab() -> dict[str, list[str]]:
    """Return a toy Vietnamese food vocabulary grouped by category."""
    return {
        "dishes": [
            "phở",
            "bún",
            "bún chả",
            "bún bò",
            "hủ tiếu",
            "mì",
            "miến",
            "cháo",
            "xôi",
            "cơm tấm",
            "cơm rang",
            "bánh mì",
            "bánh cuốn",
            "bánh xèo",
            "gỏi cuốn",
            "nem rán",
            "chả giò",
            "bánh chưng",
            "bánh bao",
            "chè",
        ],
        "drinks": [
            "cà phê",
            "cà phê sữa",
            "trà đá",
            "trà chanh",
            "trà đào",
            "nước mía",
            "nước dừa",
            "sinh tố",
            "sữa đậu nành",
            "nước cam",
        ],
        "proteins": [
            "bò",
            "gà",
            "heo",
            "lợn",
            "tôm",
            "cá",
            "mực",
            "trứng",
            "chả",
            "đậu phụ",
        ],
        "vegetables_herbs": [
            "rau",
            "xà lách",
            "rau thơm",
            "hành",
            "ngò",
            "tía tô",
            "giá",
            "dưa leo",
            "cà chua",
            "ớt",
        ],
        "flavors": [
            "ngọt",
            "mặn",
            "chua",
            "cay",
            "béo",
            "thơm",
            "đậm đà",
            "nhẹ",
            "giòn",
            "mềm",
        ],
        "ingredients": [
            "nước mắm",
            "muối",
            "đường",
            "tiêu",
            "chanh",
            "sả",
            "gừng",
            "tỏi",
            "hành phi",
            "dầu ăn",
        ],
        "cooking_methods": [
            "luộc",
            "chiên",
            "xào",
            "nướng",
            "hấp",
            "kho",
            "rim",
            "trộn",
            "nấu",
            "ăn",
        ],
        "actions": [
            "thích",
            "gọi",
            "nấu",
            "ăn",
            "uống",
            "mua",
            "chọn",
            "thử",
            "trộn",
            "thêm",
        ],
        "contexts": [
            "bữa sáng",
            "bữa trưa",
            "bữa tối",
            "buổi sáng",
            "buổi tối",
            "cuối tuần",
            "mỗi ngày",
            "hôm nay",
            "ngày mai",
            "mùa hè",
        ],
        "places_descriptors": [
            "quán",
            "nhà",
            "chợ",
            "đường phố",
            "nhà hàng",
            "ngon",
            "rẻ",
            "nóng",
            "lạnh",
            "tươi",
        ],
    }


def count_vocab_items(vocab: dict[str, list[str]]) -> int:
    """Count the total number of vocabulary items."""
    return sum(len(words) for words in vocab.values())


def generate_sentences(
    vocab: dict[str, list[str]],
    n_sentences: int = 500,
    seed: int = 42,
) -> list[str]:
    """Generate a toy Vietnamese food corpus from simple templates."""
    rng = random.Random(seed)
    sentences: list[str] = []

    templates = [
        lambda: f"tôi thích {rng.choice(vocab['dishes'])}",
        lambda: (
            f"tôi ăn {rng.choice(vocab['dishes'])} "
            f"vào {rng.choice(vocab['contexts'])}"
        ),
        lambda: f"{rng.choice(vocab['dishes'])} có {rng.choice(vocab['proteins'])}",
        lambda: (
            f"{rng.choice(vocab['dishes'])} "
            f"có {rng.choice(vocab['vegetables_herbs'])}"
        ),
        lambda: (
            f"{rng.choice(vocab['drinks'])} "
            f"rất {rng.choice(vocab['flavors'])}"
        ),
        lambda: (
            f"{rng.choice(vocab['dishes'])} "
            f"rất {rng.choice(vocab['flavors'])}"
        ),
        lambda: (
            f"tôi uống {rng.choice(vocab['drinks'])} "
            f"vào {rng.choice(vocab['contexts'])}"
        ),
        lambda: (
            f"{rng.choice(vocab['ingredients'])} "
            f"làm món ăn {rng.choice(vocab['flavors'])}"
        ),
        lambda: (
            f"mẹ {rng.choice(vocab['cooking_methods'])} "
            f"{rng.choice(vocab['dishes'])} "
            f"ở {rng.choice(vocab['places_descriptors'])}"
        ),
        lambda: (
            f"tôi gọi {rng.choice(vocab['dishes'])} "
            f"và {rng.choice(vocab['drinks'])}"
        ),
        lambda: (
            f"{rng.choice(vocab['dishes'])} "
            f"ăn với {rng.choice(vocab['ingredients'])}"
        ),
        lambda: (
            f"{rng.choice(vocab['dishes'])} "
            f"ở {rng.choice(vocab['places_descriptors'])} "
            f"rất {rng.choice(vocab['places_descriptors'])}"
        ),
    ]

    for _ in range(n_sentences):
        sentence = rng.choice(templates)()
        sentences.append(sentence)

    return sentences


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save a dictionary to a UTF-8 JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_text_lines(lines: list[str], path: Path) -> None:
    """Save a list of strings to a text file, one line per entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")


def save_jsonl(sentences: list[str], path: Path) -> None:
    """Save sentences as JSONL records with integer IDs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for idx, sentence in enumerate(sentences):
            record = {"id": idx, "text": sentence}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_sentences: int = 500,
    seed: int = 42,
) -> None:
    """Build and save the toy Vietnamese food dataset."""
    vocab = build_vocab()
    total_items = count_vocab_items(vocab)
    sentences = generate_sentences(
        vocab=vocab,
        n_sentences=n_sentences,
        seed=seed,
    )

    save_json(vocab, output_dir / "vocab.json")
    save_text_lines(sentences, output_dir / "sentences.txt")
    save_jsonl(sentences, output_dir / "dataset.jsonl")

    metadata = {
        "dataset_name": "toy_vietnamese_food_words",
        "num_vocab_items": total_items,
        "num_sentences": len(sentences),
        "seed": seed,
        "files": {
            "vocab": "vocab.json",
            "sentences": "sentences.txt",
            "dataset": "dataset.jsonl",
        },
    }
    save_json(metadata, output_dir / "metadata.json")

    print(f"Saved dataset to: {output_dir}")
    print(f"Vocabulary size: {total_items}")
    print(f"Number of sentences: {len(sentences)}")


if __name__ == "__main__":
    build_dataset()