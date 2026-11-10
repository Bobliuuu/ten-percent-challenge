# L1 artifact plan: making L1 fileable (PREPARATION ONLY, nothing is uploaded or published)

Status 2026-11-09. Nothing here has been executed except the read-only checks. Every upload/publish step below is for the user to approve and run.

## 1. What L1 is and where it lives now

L1 = Bonsai-8B (natively 1-bit, lossless repack) + closed-form `scale`-carrier feedback; task SKILL7 (7 skills, new label `Recipe`).

| piece | where | public? |
|---|---|---|
| Code: `answer.py`, `feedback.py`, `bonsai_loader.py`, `common.py`, `skill_task.json`, `feedback_calib.jsonl` (182 unlabeled prompts), `tensor_manifest.json`, `README.md` | repo `v2/lanes/L1/` (26,881 B code + 45,503 B calibration) | yes (repo is public) |
| Weights tree (`bits.bin`, `scales.bin`, `small.pt`, `index.json`, tokenizer, config, chat template, LICENSE, NOTICE) | PRIVATE Modal Volume `tenpct-l1-art`, dir `/art`, 14 files, 1,168,032,850 B (1.088 GiB) | NO |
| sha256 of every weights file | repo `v2/lanes/L1/build/SHA256SUMS.artifact` (+ `SHA256SUMS`, `SHA256SUMS.build`), also inside the volume | yes |
| Locked exam, teach, annotation pool, neighbours | repo `v2/exam/data/*.jsonl|json`, sealed by `v2/exam/MANIFEST.json` (sha256 per file), loader `v2/exam/loader.py` | yes |
| Rebuild recipe for the weights | `build/pack1.py` + `build/modal_publish.py` (downloads public `prism-ml/Bonsai-8B-unpacked`, repacks, checks every hash against `SHA256SUMS`, prints ALL_MATCH) | yes |
| Fresh-clone proof | `build/modal_dry_run.py`, `results/dry_run.json` | yes |

Size check: weights tree 1,168,032,850 B + code/calibration/manifest -> 1,168,130,816 B; plus the feedback patch (`scale` carrier, 4,720,692 B) = 1,172,851,508 B = 7.160% of Qwen3-8B fp16 (16,381,470,720 B). Largest files: `bits.bin` 1,023,529,984 B, `scales.bin` 127,941,248 B, `tokenizer.json` 11,422,654 B.

The weights are a bit-exact repack of a PUBLIC checkpoint (Apache-2.0, Prism ML, built from Qwen3-8B Apache-2.0). Redistribution is allowed; `NOTICE.txt` asks for attribution ("Created using Bonsai by Prism ML."). Keep `LICENSE` + `NOTICE.txt` in whatever is hosted.

## 2. What a maintainer needs to reproduce

1. The artifact directory (weights tree above) with the repo code files copied next to it (the dry run does exactly this).
2. The locked files: `v2/exam/data/skill7_exam.jsonl` (2744 items), `skill7_teach.jsonl` (164 unseen Recipe items), `skill7_ann_pool.jsonl`, `skill7_neighbours.json`, plus `v2/exam/MANIFEST.json` and `loader.py` (`python3 -I v2/exam/loader.py` verifies the seal). Already in the repo; hashes verified intact (see LOCK_AUDIT.md).
3. Environment: CUDA GPU ~17 GB, `torch==2.5.1`, `transformers==4.56.2`, `numpy`, `accelerate`.
4. Answer command (one label per line, greedy, free text):
   `python answer.py --artifact DIR --task skill_task.json [--in items.jsonl|-] ["text" ...]`
5. Feedback command (<= 25 rows `{"text","label"}`, label may be new; writes `patch.pt`; an empty file restores the identical tree):
   `python feedback.py --artifact DIR --task skill_task.json --ann ann.jsonl`
6. Own-data check: the maintainer's own held-out data and own annotations go through the same two commands; no training, no stored annotations.

Gap to close before filing: the maintainer should not need Modal. Provide the plain-GPU instructions in the filing text (steps in section 5). `modal_dry_run.py` is Modal-specific but its logic is just: copy code next to weights, verify sha256, answer -> feedback -> answer -> empty feedback.

Known weakness: `modal_publish.py` calls `snapshot_download("prism-ml/Bonsai-8B-unpacked")` with no revision pin. If upstream changes the repo, a rebuild would fail the hash check (it fails loudly, it cannot silently differ). Pinning the revision is a one-line fix; do it before relying on the rebuild route.

## 3. Options for a durable public copy

Payload: 1,168,032,850 B weights (+ ~0.1 MB code). Single-file tarball is ~1.09 GiB uncompressed (bits.bin is already near-incompressible; do not bother compressing, `zstd` saves < 3% and costs the maintainer a step).

| option | durability | size/limits | cost | needs | notes |
|---|---|---|---|---|---|
| A. Hugging Face Hub model repo (public, new repo, NOT a GitHub repo) | high; revision-pinned URLs (commit sha); same host as the base model | no file-size problem (Xet/LFS) | $0 | HF account + write token | native fit: `hf download` / `snapshot_download` is what a model maintainer expects; LFS shows sha256 per file. Creates a new public repo on HF (user said no public repos for now: user decides) |
| B. GitHub Release asset on the existing public challenge repo | medium-high; tied to the repo that holds the issue | 2 GiB per asset limit: `bits.bin` (976 MiB) fits; ship as one tar (1.09 GiB) or the files separately | $0 | `gh` auth (already used) | no new repo; keeps git history small; asset URL is stable per tag; deletable by repo owner |
| C. Zenodo deposit | very high (CERN, DOI, immutable once published) | 50 GB | $0 | Zenodo token | immutable also means you cannot retract; DOI is a good citation in the issue |
| D. Public object-store bucket (Cloudflare R2 / S3) | as long as the owner pays | any | R2: ~$0.02/month storage, $0 egress; S3: ~$0.03/month storage + ~$0.10 per full download | cloud account | bucket URL exposes account naming; most operational overhead; avoid for a prize entry |
| E. No hosting: rebuild from the public snapshot | depends on upstream HF repo staying up | 16 GB download for the maintainer | $0 | CPU + disk 60 GB | `pack1.py` + sha256 list gives bit-exact result; as a FALLBACK only (slow, upstream can move) |

## 4. Recommendation

Primary: **B. GitHub Release asset on the existing public challenge repo**, with **E as the documented fallback** (and `SHA256SUMS.artifact` in the issue so any copy is checkable).
Why: no new repo or account; 1.09 GiB fits under the 2 GiB asset limit; the maintainer already reads that repo; zero cost; the bytes are provably the public Bonsai repack (hash list + rebuild recipe). If the user prefers a more neutral permanent home, add **A (Hugging Face)** as a mirror; Zenodo (C) only if a DOI is wanted, because it cannot be retracted.

Counter-risk to know: GitHub Release assets can be deleted/changed by the owner after the deadline passes, which a maintainer may distrust; the sha256 list in the issue text and the Zenodo/HF mirror answer that.

## 5. Exact steps (NOT executed; run only after the user approves)

```
# 0. stage (private, local; ~1.2 GB; run in an empty dir)
mkdir -p ~/l1-stage && cd ~/l1-stage
modal volume get tenpct-l1-art art ./art
cd art && shasum -a 256 -c <(awk 'NF>=3{print $1"  "$2}' ~/ten-percent/v2/lanes/L1/build/SHA256SUMS.artifact | grep -v ALL_MATCH) && cd ..
# 1. add code from the repo at the filing commit, next to the weights
L=~/ten-percent/v2/lanes/L1
cp $L/{answer.py,feedback.py,bonsai_loader.py,common.py,skill_task.json,feedback_calib.jsonl,tensor_manifest.json,README.md} art/
cp -r ~/ten-percent/v2/exam art-exam    # or tell the maintainer to use the repo checkout
# 2. one tar, hash it
tar --sort=name --owner=0 --group=0 --mtime=@0 -cf l1-bonsai8b-artifact.tar art
shasum -a 256 l1-bonsai8b-artifact.tar | tee l1-bonsai8b-artifact.tar.sha256
ls -l l1-bonsai8b-artifact.tar        # expect ~1.17e9 B, below the 2 GiB asset limit
# 3. (APPROVAL NEEDED) publish option B
gh release create l1-artifact-v1 l1-bonsai8b-artifact.tar l1-bonsai8b-artifact.tar.sha256 \
  --repo <the public challenge repo> --title "L1 artifact" --notes "Bonsai-8B 1-bit repack, Apache-2.0, see SHA256SUMS.artifact"
# 3 alt A. (APPROVAL NEEDED) Hugging Face mirror
hf repo create <user>/<name> --type model && hf upload <user>/<name> ./art .
# 4. verify the public copy from a clean directory
mkdir -p /tmp/verify && cd /tmp/verify && curl -L -o a.tar <asset url> && shasum -a 256 a.tar   # must equal the .sha256 file
```
Maintainer-side run on any 17 GB+ CUDA box:
```
tar xf l1-bonsai8b-artifact.tar && cd art
pip install torch==2.5.1 transformers==4.56.2 numpy accelerate
python answer.py --artifact . --task skill_task.json "wake me at six" "how do I boil an egg"
printf '{"text":"how long do I boil an egg","label":"Recipe"}\n' > ann.jsonl
python feedback.py --artifact . --task skill_task.json --ann ann.jsonl
python answer.py --artifact . --task skill_task.json "how do I roast a turkey"
```

## 6. Open items (for the user)
- Choice of host (B recommended) and whether any public upload is OK at all before 2026-11-12 17:05 UTC; deadline leaves 3 days.
- Pin `snapshot_download(..., revision=...)` in `build/modal_publish.py` (not done here).
- `answer.py` is plain HF generate with `bonsai_loader.py`; the maintainer needs a GPU. Without one there is no CPU path (8B in fp16).

## 7. Final fresh-clone run (2026-11-09, Modal L40S, detached)
`build/modal_dry_run.py` against a shallow clone of public `main` at commit 3b12a02 (HEAD at launch; later commits only touch v2/FILING and other lanes): weights from the private volume, 0 sha256 mismatches, shipped answer.py -> feedback.py (8 annotations, `scale` carrier, patch 4,720,692 B) -> answer.py: 4/4 unseen recipe prompts answer `Recipe`, 7/7 neighbour prompts unchanged, empty feedback removes the patch and the tree is identical to base. Raw: `v2/lanes/L1/results/dry_run.json`. This is the 3-item-class smoke from the script, NOT the 2744-item locked run (that is `results/locked_L1.json`, done 2026-11-08).
