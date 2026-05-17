"""
Convert augmented_full_40.json (tokens/labels schema, array format)
→ word JSONL format (words/tags schema, one JSON per line)
→ Merge vào train_word.json
"""

import json
import shutil
from pathlib import Path

BASE = Path(__file__).parent
SRC  = BASE / "data/syllable/augmented_full_40.json"
DEST = BASE / "data/word/train_word.json"
BACKUP = BASE / "data/word/train_word.json.bak"

# 1. Load augmented data
with open(SRC, encoding="utf-8") as f:
    aug_data = json.load(f)

print(f"Loaded {len(aug_data)} sentences from {SRC.name}")

# Validate schema
for i, item in enumerate(aug_data):
    assert "tokens" in item and "labels" in item, \
        f"Sentence {i} missing 'tokens' or 'labels' key: {list(item.keys())}"
    assert len(item["tokens"]) == len(item["labels"]), \
        f"Sentence {i}: token/label length mismatch ({len(item['tokens'])} vs {len(item['labels'])})"

# 2. Convert schema: tokens→words, labels→tags
converted = [{"words": item["tokens"], "tags": item["labels"]} for item in aug_data]

# 3. Backup train_word.json
shutil.copy(DEST, BACKUP)
print(f"Backup saved -> {BACKUP.name}")

# 4. Count before merge
with open(DEST, encoding="utf-8") as f:
    original_lines = [l for l in f if l.strip()]
print(f"Original train size: {len(original_lines)} sentences")

# 5. Append converted sentences to train_word.json
with open(DEST, "a", encoding="utf-8") as f:
    for item in converted:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 6. Verify result
with open(DEST, encoding="utf-8") as f:
    new_lines = [l for l in f if l.strip()]
print(f"New train size:      {len(new_lines)} sentences (+{len(new_lines) - len(original_lines)})")

# 7. Entity distribution after merge
from collections import Counter
counts_before = Counter()
counts_after  = Counter()

for line in original_lines:
    for tag in json.loads(line)["tags"]:
        if tag.startswith("B-"):
            counts_before[tag[2:]] += 1

for line in new_lines:
    for tag in json.loads(line)["tags"]:
        if tag.startswith("B-"):
            counts_after[tag[2:]] += 1

print("\nEntity distribution (before → after merge):")
all_types = sorted(counts_after.keys(), key=lambda k: -counts_after[k])
for k in all_types:
    diff = counts_after[k] - counts_before.get(k, 0)
    marker = f"  (+{diff})" if diff else "       "
    print(f"  {k:<25} {counts_before.get(k, 0):>5} → {counts_after[k]:>5}{marker}")

print("\nDone.")
