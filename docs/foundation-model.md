# Scope: a pretrained terrain encoder as the foundation

Status: scoping, with one measured pilot. Nothing here is built beyond the pilot. Numbers marked *measured*
come from this repository's own runs on an Apple M4 Max; *estimate* means derived from measured numbers with
stated assumptions; *unverified* means I have not checked it.

## 1. Goal and non-goals

**Goal.** One encoder, pretrained without labels on a large, multi-country corpus of open terrain, whose dense
per-pixel features let a user find *any* shape from a few clicked examples, at the speed of the streaming pipeline
we already have. Prompting, not per-type training, is the product.

**Non-goals.** Claiming discoveries; identifying cultural affiliation (lidar sees shape, not people); publishing
locations of unrecorded sites; replacing archaeologists or the communities whose heritage this concerns.

## 2. Why this is the right next investment

What the current system shows (*measured*, English terrain, 13 tiles, polygon-containment hits, 100 candidates
per 9 km² tile, 10 example prompts):

| Feature source | Round barrow | Cairn | Hut circle | Stone setting |
|---|---|---|---|---|
| Hand terrain features | 35% | 16% | 1% | 0% |
| Barrow-trained U-Net's dense features | 61% | 32% | 2% | 6% |
| Contrastive encoder (global patch embedding) | 26% | 18% | 2% | 4% |
| Template matching on raw terrain | 47% | 10% | 0% | 0% |
| Supervised for the type | n/a | 37% | 16% | n/a |

Prompting works for the one shape a supervised network was trained on, and for nothing else. The features are
the bottleneck: a probe cannot recover shapes the encoder never learned to represent. Better features, learned
from far more terrain than five tiles, are the lever that also makes the project hard to copy.

## 3. Pilot (measured): does a dense, label-free objective help?

Setup: the mound U-Net's architecture (1.0 M parameters) trained to reconstruct randomly hidden 16 m blocks of
terrain relief, on **every terrain tile we hold**: 18 terrains, 154 km², three countries, no labels. 3,000 steps
at batch 64 (192,000 crops, about 20 passes), 594 s, **323 crops/s** of 128 m x 128 m, masked reconstruction
error 0.17 -> 0.014. Its decoder features were then dropped into the unchanged prompt benchmark.

| Feature source (10 prompts) | Round barrow | Cairn | Hut circle | Stone setting |
|---|---|---|---|---|
| **Masked-terrain features (pilot)** | **12%** | **6%** | 1% | 0% |
| Hand terrain features | 35% | 16% | 1% | 0% |
| Contrastive encoder | 26% | 18% | 2% | 4% |
| Barrow-trained U-Net features | 61% | 32% | 2% | 6% |

**H1 fails at pilot scale with this objective**: masked reconstruction produced features that prompt *worse than
hand-built terrain features*. That is a result about this objective at this scale, not about the idea. A
plausible cause (a hypothesis, not tested): a barrow is 10-30 m across, so 16 m masks hide it entirely and the
network only learns to inpaint smooth ground, never encoding mounds. **Tested next**: the same pretraining with 4 m masks (corpus and steps unchanged): barrows 17%, cairns 6%, hut
circles 3% at 10 prompts. Better than 16 m masks but still below hand-built features, so mask size is not the
main cause; the objective as formulated (reconstruct hidden terrain from context) does not produce features
that separate shapes, at this scale.

Consequence for the plan: **do not scale the corpus yet.** Scaling a failing objective wastes the expensive
part. Objective experiments are cheap (about 20 minutes each at this corpus size) and come first.

## 4. Hypotheses and how each is tested

| # | Hypothesis | Test | Go / no-go |
|---|---|---|---|
| H1 | A dense masked-terrain objective gives better prompt features than the supervised network's features | Frozen prompt benchmark (section 6), same protocol as the table above | Mean own-type recall over the four types beats the U-Net-feature row, and hut circles leave chance level |
| H2 | More unlabelled terrain helps (data scaling) | Same encoder at 1x, 10x, 100x corpus, fixed steps per km² | Monotone gain; stop scaling where the curve flattens |
| H3 | Multi-country data improves transfer to a held-out country | Train without country X, test on X (forest, upland, arid) | Held-out gap shrinks versus single-country training |
| H4 | Capacity matters | 1 M, 10 M, 50 M parameter encoders at matched compute | Gain per 10x parameters justifies the Mac training time |
| H5 | Multi-scale input lets one encoder find 10 m and 100 m structures | Pyramid inputs, platform-scale test on Quiriguá-like data | Needs labelled large structures (none open found); may stay qualitative |

Objective ladder, cheapest first: masked reconstruction (pilot) -> latent prediction in feature space (JEPA /
DINO-style, avoids wasting capacity on noise) -> combined with a small amount of supervised signal only if the
label-free route plateaus. Each step is adopted only if it moves H1.

## 5. Data plan

**Corpus rules.** Open licence verified from the publisher's page; provenance (source, licence, attribution,
resolution, date) stored in a manifest beside every tile; no terrain from sources whose terms forbid this use.
Held-out regions fixed up front and never trained on: one upland, one forest, one flat arable region.

**Verified sources today.** England EA 1 m DTM (OGL v3), Netherlands AHN 0.5 m DTM (CC0, 1 m served by the
server), generic cloud GeoTIFFs, USGS 3DEP point clouds (built, not re-run since refactor), Quiriguá UAV DTM
(CC BY 4.0). Not usable here: Mexico's INEGI (non-commercial). Unverified: Canada, France, New Zealand, Nordic.

**Volume (*measured* sizes, *estimate* totals).** Compressed 1 m terrain costs about 2.6 MB per km² (Dartmoor
cloud GeoTIFF, 2.62; AHN, 2.54); the 0.25 m survey costs 28 MB per km². Training at 1 m therefore keeps the corpus
small:

| Corpus | Size at ~2.6 MB/km² | Download at 20 Mbit/s | At 5 Mbit/s |
|---|---|---|---|
| Today (154 km²) | 0.4 GB | minutes | minutes |
| 10,000 km² | 26 GB | 2.9 h | 11.6 h |
| 100,000 km² | 260 GB | 29 h | 116 h |

Raw point clouds are not needed for pretraining; gridded terrain is. Higher-resolution sources are averaged to
1 m (already implemented), so volume stays bounded.

## 6. Evaluation (the second half of the moat)

A frozen, versioned benchmark, published with the code: typed monuments from several countries as prompts and
targets, polygon-containment hits at fixed K per 9 km², region-held-out splits, and always the same controls
(random placement, supervised ceiling for the type, the existing baselines). What exists: four English types
(`prompt_types`, `ceiling`), Dutch tumuli via OpenStreetMap (label-bias caveat). What is missing: more countries
with typed, precisely located labels, and non-archaeological object classes (unverified availability).
Release gates are numeric and set before training, not after.

## 7. Systems work that makes it fast (and what to build only if profiling demands it)

*Built and measured:* streaming from cloud GeoTIFFs (first result 3.3 s versus 42 s at 20 Mbit/s), memory-gated
tiling with seam-exact results, PyTorch feature stack 50x faster than SciPy on Metal, dense prompt search in
about 90 ms. *Next:* a training data loader that reads blocks over range requests and caches decoded crops
(training must not be network-bound). *Low-level kernels:* none yet; the point-cloud ingest path (decode, ground
filter, rasterise) is the likely hotspot but was never profiled live, so profile first, write kernels only where
a measured hotspot exceeds the accelerator's headroom.

## 8. Compute plan (*estimate* from the pilot's measured throughput)

Measured on the pilot: **323 crops/s** (128 m x 128 m crops, 1.0 M-parameter network, M4 Max). One
non-overlapping crop covers 0.0164 km², so one pass over a corpus takes (km² x 61 crops) / 323 seconds:

| Corpus | One pass | 10 passes |
|---|---|---|
| 154 km² (pilot) | 29 s | 5 min |
| 10,000 km² | 31 min | 5.2 h |
| 100,000 km² | 5.2 h | 52 h |

Capacity (H4) is *unmeasured*: a 10 M-parameter encoder will be slower per step by an amount I have not
measured, so the 100,000 km² row only holds for the small network. Reading crops from a local cache needs about
42 MB/s, trivial; the download times in section 5 dominate the first epoch, not training.

## 9. Milestones with gates (no calendar dates)

1. **M1 pilot** (done): masked pretraining on 154 km², evaluated on the frozen benchmark. Gate: decides whether
   the objective is worth scaling.
2. **M2 benchmark v1**: freeze types, splits and controls; add at least one more country with typed labels.
3. **M3 scale to ~10,000 km²** across England and the Netherlands plus any newly verified source; H2 and H3 curves.
4. **M4 objective upgrade** (latent prediction) only if M3 plateaus below the gates.
5. **M5 release**: weights, manifest, benchmark, `scan --prompt`, model card with failure modes.

Stop rule: if H1 fails at M1 and again after the objective upgrade, the right conclusion is that open lidar at
1 m does not carry enough signal for few-shot shapes, and effort moves to better labelled data, not bigger models.

## 10. Risks

- **Label bias.** OpenStreetMap tumuli are often traced from lidar hillshades and flatter every method.
- **Domain shift.** English downland is open ground; forest and jungle terrain behave differently (Quiriguá is
  only a qualitative check, with no labels).
- **Label imprecision.** Scheduled polygons are protected areas, not object positions; hit definitions must be
  polygon-based and stated.
- **Licences.** Some sources forbid commercial use (INEGI) or require share-alike for derived data (OpenStreetMap).
  The manifest enforces which tiles may enter a released model.
- **Sensitivity.** Predictions can point at unrecorded sites. Keep coordinates local by default and involve the
  relevant heritage authorities and communities before any publication.
- **Overclaiming.** "Foundation model" is a claim to be earned by the gates above, not a name.

## 11. Decisions needed

1. **Order of work.** Recommended: objective experiments on the current 154 km² first (cheap, decides whether
   scaling is worth it), then benchmark v1, then scale. Alternative: scale first and hope; I advise against it.
2. **Sources to verify next** (each needs your approval before any download): which countries or terrains
   matter to you (forest, arid, jungle, upland), since held-out regions are chosen from that.
3. **Compute you are willing to spend** on this Mac: a 10,000 km² run is about 5 hours at the small size.
4. **Release policy.** Apache-2.0 weights and benchmark are consistent with the repository, but weights trained on
   OpenStreetMap-derived labels or non-commercial terrain are not; the manifest decides. Also decide the policy
   for sharing detections of unrecorded sites (default here: local only).
