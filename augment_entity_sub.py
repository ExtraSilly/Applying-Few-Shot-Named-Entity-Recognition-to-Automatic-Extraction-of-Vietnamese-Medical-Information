"""
Entity Substitution Augmentation cho Few-shot NER
Thay the entity trong cau bang entity khac cung loai → sinh du lieu moi

Usage:
    python augment_entity_sub.py               # aug train, luu vao train
    python augment_entity_sub.py --dry-run     # xem truoc, khong luu
    python augment_entity_sub.py --target JOB  # chi aug 1 entity type
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

BASE       = Path(__file__).parent
TRAIN_PATH = BASE / "data/word/train_word.json"
SEED       = 42
AUG_PER_SENT = 3      # so cau aug tao ra tu moi cau goc co entity dich


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_entity_pool(data):
    """
    Xay dung pool entity: {entity_type: [list of entity spans (list of words)]}
    """
    pool = defaultdict(list)
    for item in data:
        words, tags = item["words"], item["tags"]
        i = 0
        while i < len(tags):
            if tags[i].startswith("B-"):
                etype = tags[i][2:]
                span  = [words[i]]
                j = i + 1
                while j < len(tags) and tags[j] == f"I-{etype}":
                    span.append(words[j])
                    j += 1
                pool[etype].append(span)
                i = j
            else:
                i += 1

    # Loai bo trung lap trong moi pool
    deduped = {}
    for etype, spans in pool.items():
        seen   = set()
        unique = []
        for span in spans:
            key = tuple(span)
            if key not in seen:
                seen.add(key)
                unique.append(span)
        deduped[etype] = unique

    return deduped


def get_entity_spans(words, tags):
    """Tra ve list cac (start, end, etype) trong mot cau."""
    spans = []
    i = 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            etype = tags[i][2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{etype}":
                j += 1
            spans.append((i, j, etype))
            i = j
        else:
            i += 1
    return spans


def substitute_entity(words, tags, start, end, etype, new_span):
    """Thay the entity [start:end] bang new_span, cap nhat tags."""
    new_words = words[:start] + new_span + words[end:]
    new_tags  = (tags[:start]
                 + [f"B-{etype}"] + [f"I-{etype}"] * (len(new_span) - 1)
                 + tags[end:])
    return new_words, new_tags


def augment(data, pool, target_types, n_per_sent, rng):
    augmented = []

    for item in data:
        words = item["words"]
        tags  = item["tags"]
        spans = get_entity_spans(words, tags)

        # Chon cac span thuoc target_types
        target_spans = [(s, e, t) for s, e, t in spans if t in target_types]
        if not target_spans:
            continue

        generated = set()
        attempts  = 0

        while len(generated) < n_per_sent and attempts < n_per_sent * 10:
            attempts += 1
            # Chon ngau nhien 1 entity span de thay the
            start, end, etype = rng.choice(target_spans)
            original_span = tuple(words[start:end])

            # Chon entity thay the khac voi entity goc
            candidates = [s for s in pool.get(etype, [])
                          if tuple(s) != original_span]
            if not candidates:
                continue

            new_span = rng.choice(candidates)
            new_words, new_tags = substitute_entity(words, tags, start, end,
                                                     etype, new_span)

            key = (tuple(new_words), tuple(new_tags))
            if key not in generated:
                generated.add(key)
                augmented.append({"words": new_words, "tags": new_tags})

    return augmented


def print_stats(data, label=""):
    from collections import Counter
    counts = Counter()
    for item in data:
        for tag in item["tags"]:
            if tag.startswith("B-"):
                counts[tag[2:]] += 1
    print(f"\n{label} ({len(data)} sentences):")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<25} {v:>5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", nargs="+",
                        default=["JOB", "TRANSPORTATION", "NAME"],
                        help="Entity types can de aug")
    parser.add_argument("--n", type=int, default=AUG_PER_SENT,
                        help="So cau aug / cau goc")
    parser.add_argument("--dry-run", action="store_true",
                        help="Xem truoc, khong luu file")
    args = parser.parse_args()

    rng  = random.Random(SEED)
    data = load_jsonl(TRAIN_PATH)
    pool = extract_entity_pool(data)

    print("Entity pool sizes:")
    for etype in args.target:
        print(f"  {etype}: {len(pool.get(etype, []))} unique entities")

    print_stats(data, "BEFORE augmentation")

    aug_data = augment(data, pool, set(args.target), args.n, rng)
    print(f"\nGenerated: {len(aug_data)} augmented sentences")
    print_stats(aug_data, "Augmented sentences only")

    if args.dry_run:
        print("\n--- 5 samples ---")
        for item in aug_data[:5]:
            ents = [(w, t) for w, t in zip(item["words"], item["tags"])
                    if t != "O"]
            try:
                print(f"  {' '.join(item['words'][:10])}...")
                print(f"  {ents}\n")
            except UnicodeEncodeError:
                print(f"  [sentence contains Vietnamese characters]")
                print(f"  entities: {[(t,) for _, t in ents]}\n")
        return

    # Merge vao train
    import shutil
    backup = TRAIN_PATH.with_suffix(".json.aug_bak")
    shutil.copy(TRAIN_PATH, backup)
    print(f"\nBackup: {backup.name}")

    # Check overlap voi data goc
    orig_set = set(tuple(i["words"]) for i in data)
    truly_new = [i for i in aug_data if tuple(i["words"]) not in orig_set]
    print(f"Truly new (no overlap): {len(truly_new)}")

    with open(TRAIN_PATH, "a", encoding="utf-8") as f:
        for item in truly_new:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    final = load_jsonl(TRAIN_PATH)
    print_stats(final, "AFTER augmentation")
    print(f"\nDone. Train: {len(data)} -> {len(final)} sentences (+{len(final)-len(data)})")


if __name__ == "__main__":
    main()
