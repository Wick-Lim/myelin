import json, glob, unicodedata, random

S = "/private/tmp/claude-501/-Users-wicklim-Workspaces-myelin/b051aeeb-439e-4ab7-8e7b-b5ae2db52370/scratchpad"

docs = []
for fp in sorted(glob.glob(S + "/rows/r_*.json")):
    with open(fp) as f:
        data = json.load(f)
    if "rows" not in data:
        print("BAD FILE", fp, list(data.keys()))
        continue
    docs.extend(row["row"]["text"] for row in data["rows"])

random.seed(0)
random.shuffle(docs)
n_hold = max(1, len(docs) // 10)
hold, train = docs[:n_hold], docs[n_hold:]

def stats(ds):
    b = sum(len(t.encode("utf-8")) for t in ds)
    c = sum(len(t) for t in ds)
    h = sum(1 for t in ds for ch in t if 0xAC00 <= ord(ch) <= 0xD7A3)
    return b, c, h

tb, tc, th = stats(train)
hb, hc, hh = stats(hold)
print(f"train: docs={len(train)} bytes={tb} chars={tc} hangul_chars={th} ({th/tc:.1%}) bytes/char={tb/tc:.3f}")
print(f"hold:  docs={len(hold)} bytes={hb} chars={hc} hangul_chars={hh} ({hh/hc:.1%})")

# distinct syllables in the corpus
syls = {}
for t in docs:
    for ch in t:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            syls[ch] = syls.get(ch, 0) + 1
tot = sum(syls.values())
freq = sorted(syls.values(), reverse=True)
cum = 0
marks = {}
for i, v in enumerate(freq, 1):
    cum += v
    for k in (500, 1000, 1500, 2000, 2350, 3000):
        if i == k:
            marks[k] = cum / tot
print(f"distinct syllables used: {len(syls)} / 11172 possible; total hangul chars={tot}")
for k in sorted(marks):
    print(f"  top {k} syllables cover {marks[k]:.4%}")

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

results = []
for vocab in (2048, 4096, 8192, 16384, 32768):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<|endoftext|>"], show_progress=False)
    tok.train_from_iterator(train, trainer)
    ntok = sum(len(tok.encode(t).ids) for t in hold)
    ntok_tr = sum(len(tok.encode(t).ids) for t in train[:200])
    b200, c200, _ = stats(train[:200])
    # how many single-hangul-syllable tokens exist in vocab
    v = tok.get_vocab()
    syl_tokens = 0
    multi_syl = 0
    for token_str in v:
        try:
            decoded = tok.decoder.decode([token_str])
        except Exception:
            continue
        hang = [ch for ch in decoded if 0xAC00 <= ord(ch) <= 0xD7A3]
        stripped = decoded.strip()
        if len(stripped) == 1 and hang:
            syl_tokens += 1
        elif len(hang) >= 2:
            multi_syl += 1
    results.append((vocab, hb / ntok, hc / ntok, ntok / hc, syl_tokens, multi_syl))
    print(f"vocab={vocab:6d}  held-out: bytes/token={hb/ntok:.3f} chars/token={hc/ntok:.3f} tokens/char={ntok/hc:.3f} | train-subset chars/token={c200/ntok_tr:.3f} | single-syllable tokens={syl_tokens} multi-syllable tokens={multi_syl}")

FULL = 1_412_099_944  # bytes, wikimedia/wikipedia 20231101.ko dataset card
print("\nExtrapolated total tokens for full 20231101.ko (1.412 GB text):")
for vocab, bpt, cpt, tpc, st, ms in results:
    print(f"  vocab={vocab:6d}: ~{FULL/bpt/1e6:.0f}M tokens")
