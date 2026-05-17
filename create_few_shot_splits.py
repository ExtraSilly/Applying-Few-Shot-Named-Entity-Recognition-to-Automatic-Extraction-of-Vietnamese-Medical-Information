"""
Tao fixed few-shot splits tu train_word.json
Output: data/few_shot/k{K}/support.json  (K = 1, 5, 10, 20)
        data/few_shot/k{K}/query.json    (phan con lai cua train)
        data/few_shot/episodes/          (N-way K-shot episodes cho meta-learning)

Moi lan chay voi cung SEED se cho ket qua giong het nhau.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
SEED       = 42
K_SHOTS    = [1, 5, 10, 20]
N_EPISODES = 100          # so episode cho meta-learning
N_WAY      = 5            # so entity type moi episode
K_SUPPORT  = 5            # so cau support moi class trong episode
K_QUERY    = 10           # so cau query moi class trong episode

BASE      = Path(__file__).parent
TRAIN_SRC = BASE / "data/word/train_word.json"
OUT_DIR   = BASE / "data/few_shot"
# ───────────────────────────────────────────────────────────────────────────


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_entity_types(data):
    types = set()
    for item in data:
        for tag in item["tags"]:
            if tag.startswith("B-"):
                types.add(tag[2:])
    return sorted(types)


def index_by_entity(data):
    """Tra ve dict: entity_type -> [cac index cau co entity do]"""
    idx = defaultdict(list)
    for i, item in enumerate(data):
        seen = set()
        for tag in item["tags"]:
            if tag.startswith("B-"):
                seen.add(tag[2:])
        for etype in seen:
            idx[etype].append(i)
    return idx


def make_k_shot_split(data, entity_index, k, rng):
    """
    Voi moi entity type: chon K cau chua entity do (khong trung nhau neu co the).
    Tra ve (support_indices, query_indices).
    """
    support_idx = set()
    for etype, indices in entity_index.items():
        available = [i for i in indices if i not in support_idx]
        if len(available) < k:
            # Neu khong du, lay tat ca roi bo sung tu pool goc
            chosen = available + rng.sample(
                [i for i in indices if i not in available],
                min(k - len(available), len(indices) - len(available))
            )
        else:
            chosen = rng.sample(available, k)
        support_idx.update(chosen)

    query_idx = [i for i in range(len(data)) if i not in support_idx]
    return sorted(support_idx), query_idx


def make_episodes(data, entity_index, entity_types, n_episodes, n_way, k_support, k_query, rng):
    """
    Tao N-way K-shot episodes cho meta-learning.
    Moi episode: chon n_way entity types, moi type lay k_support cau support + k_query cau query.
    """
    episodes = []
    for ep in range(n_episodes):
        # Chon n_way entity types ngau nhien
        # Chi chon type co du (k_support + k_query) cau
        eligible = [t for t in entity_types if len(entity_index[t]) >= k_support + k_query]
        if len(eligible) < n_way:
            print(f"  Warning: episode {ep} - chi co {len(eligible)} eligible types (can {n_way})")
            chosen_types = eligible
        else:
            chosen_types = rng.sample(eligible, n_way)

        support, query = [], []
        for etype in chosen_types:
            indices = entity_index[etype][:]
            rng.shuffle(indices)
            s_idx = indices[:k_support]
            q_idx = indices[k_support:k_support + k_query]
            support.append({
                "entity_type": etype,
                "sentences": [data[i] for i in s_idx]
            })
            query.append({
                "entity_type": etype,
                "sentences": [data[i] for i in q_idx]
            })

        episodes.append({
            "episode_id": ep,
            "n_way": len(chosen_types),
            "k_support": k_support,
            "k_query": k_query,
            "entity_types": chosen_types,
            "support": support,
            "query": query
        })
    return episodes


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    rng = random.Random(SEED)

    print(f"Loading {TRAIN_SRC.name}...")
    data = load_jsonl(TRAIN_SRC)
    entity_types = get_entity_types(data)
    entity_index = index_by_entity(data)

    print(f"  {len(data)} sentences, {len(entity_types)} entity types: {entity_types}")
    print(f"  Seed = {SEED}")
    print()

    # 1. K-shot splits
    for k in K_SHOTS:
        rng_k = random.Random(SEED)   # reset seed moi K de doc lap nhau
        support_idx, query_idx = make_k_shot_split(data, entity_index, k, rng_k)

        support = [data[i] for i in support_idx]
        query   = [data[i] for i in query_idx]

        out = OUT_DIR / f"k{k}"
        save_jsonl(out / "support.json", support)
        save_jsonl(out / "query.json",   query)

        # Entity distribution trong support set
        from collections import Counter
        counts = Counter()
        for item in support:
            seen = set()
            for tag in item["tags"]:
                if tag.startswith("B-"):
                    seen.add(tag[2:])
            counts.update(seen)

        print(f"k={k:>2}  support={len(support):>4} sentences  query={len(query):>4} sentences")
        for t in entity_types:
            print(f"       {t:<25} {counts.get(t, 0):>3} support sentences")
        print()

    # 2. Episode splits cho meta-learning
    print(f"Tao {N_EPISODES} episodes ({N_WAY}-way {K_SUPPORT}-shot)...")
    episodes = make_episodes(
        data, entity_index, entity_types,
        N_EPISODES, N_WAY, K_SUPPORT, K_QUERY, rng
    )
    ep_path = OUT_DIR / "episodes" / f"episodes_{N_WAY}way_{K_SUPPORT}shot.json"
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ep_path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)

    print(f"  Saved {len(episodes)} episodes -> {ep_path}")
    print()

    # 3. In cau truc thu muc
    print("Output structure:")
    for p in sorted(OUT_DIR.rglob("*")):
        rel = p.relative_to(OUT_DIR)
        if p.is_file():
            size_kb = p.stat().st_size // 1024
            print(f"  data/few_shot/{rel}  ({size_kb} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
