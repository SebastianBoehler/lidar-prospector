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

### 3b. Objective experiments and a data-scaling probe (measured)

Same corpus (154 km²), same 3,000 steps, same frozen prompt benchmark, 10 example prompts, own-type recall:

| Label-free objective / feature source | Round barrow | Cairn | Hut circle | Stone setting |
|---|---|---|---|---|
| Masked reconstruction, 16 m masks (decoder features) | 12% | 6% | 1% | 0% |
| Masked reconstruction, 4 m masks (decoder features) | 17% | 6% | 3% | 0% |
| Masked reconstruction, 4 m masks (**encoder** features) | 22% | 7% | 3% | 4% |
| Global contrastive encoder (earlier run, 5 English tiles) | 26% | 18% | 2% | 4% |
| **Dense contrastive** (same place, different view) | **27%** | 5% | 3% | 0% |
| *Reference: hand terrain features* | 35% | 16% | 1% | 0% |
| *Reference: barrow-trained U-Net features (needs labels)* | 61% | 32% | 2% | 6% |

Data-scaling probe (dense contrastive, same steps): trained on **45 km²** (5 terrains) it reaches barrows 12%,
cairns 2%, hut circles 1%; on **154 km²** (3.4x more) barrows 27%, cairns 5%, hut circles 3%.

What this supports, and what it does not:

- No label-free objective tried beats hand-built terrain features at this corpus size, and all trail the
  supervised network's features by a wide margin. Labels (even from five tiles) carry much more signal than 154 km²
  of unlabelled terrain does.
- The best label-free objective **improves clearly with data** (barrows 12% -> 27% over 3.4x more terrain). Two
  points on a noisy benchmark (about +/-5 points) are a direction, not a curve, and I cannot say where it
  flattens or whether 65x more terrain (10,000 km²) would reach the supervised features' 61%.
- Hut circles and stone settings stay near chance for every label-free variant, as for every method so far.

Consequence for the plan: scaling is now worth testing, but in stages with a measured curve, not in one jump.
The objective upgrade (latent prediction) is deferred until a third data point says whether dense contrastive
plateaus; and a labelled route is back on the table as a comparison (section 9, M3b).

## 4. Hypotheses and how each is tested

| # | Hypothesis | Test | Go / no-go |
|---|---|---|---|
| H1 | A label-free objective gives better prompt features than the supervised network's features | Frozen prompt benchmark (section 6), same protocol as the table above | Mean own-type recall over the four types beats the U-Net-feature row, and hut circles leave chance level |
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

## 8. Compute plan (*estimate* from measured throughput)

Measured on the 1.0 M-parameter network (M4 Max): masked reconstruction **323 crops/s**; dense contrastive
**117-128 crops/s** (two views per crop, so about 2.7x the cost). One non-overlapping 128 m crop covers
0.0164 km², so a pass over a corpus takes (km² x 61 crops) / throughput. For the objective that currently works
best (dense contrastive, 120 crops/s):

| Corpus | One pass | 10 passes |
|---|---|---|
| 154 km² (current) | 78 s | 13 min |
| 1,500 km² (M3) | 13 min | 2.1 h |
| 10,000 km² | 85 min | 14 h |
| 100,000 km² | 14 h | 141 h |

Capacity (H4) is *unmeasured*: a larger encoder is slower per step by an amount I have not measured, so the
larger rows only hold for the small network. Reading crops from a local cache needs tens of MB/s, trivial; the
download times in section 5 dominate the first epoch, not training.

## 9. Milestones with gates (no calendar dates)

1. **M1 pilot** (done): masked reconstruction on 154 km². Negative.
2. **M1b objective experiments** (done): three label-free families, and a 45 vs 154 km² scaling probe (3b).
3. **M2 benchmark v1**: freeze types, splits and controls; add at least one more country with typed labels.
4. **M3 staged scaling**: grow the corpus to about 1,500 km² (about 10x; roughly 4-6 GB, which needs your
   approval per source) and retrain dense contrastive with identical settings. Third point on the scaling curve.
   Gate: barrows at 10 prompts must clearly exceed hand features (35%) or the curve must still be rising steeply;
   otherwise stop scaling.
5. **M3b labelled comparison** (optional, needs a decision): pretrain the same network on *open typed heritage
   polygons* from a much larger area (many types, not just barrows) as weak supervision, and compare on the
   frozen benchmark against the label-free route at equal compute. The supervised features already beat every
   label-free result, so this is the honest baseline the foundation model has to beat.
6. **M4 objective upgrade** (latent prediction) only if M3 plateaus below the gates.
7. **M5 release**: weights, manifest, benchmark, `scan --prompt`, model card with failure modes.

Stop rule: if M3 plateaus below hand features while the labelled comparison clearly wins, the conclusion is that
open lidar at 1 m carries too little label-free signal for few-shot shapes at laptop scale, and effort moves to
weak supervision from the open typed records, not bigger label-free models.

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

1. **Approve staged scaling (M3).** About 10x the current terrain from the sources already verified (England EA,
   Netherlands AHN); roughly 4-6 GB of downloads. Each source and size needs your explicit approval before I fetch.
2. **Is the labelled comparison (M3b) acceptable?** It uses open typed monument records as a training signal, which is
   the opposite of "no label hunting", but it is the honest baseline: it shows what the label-free model must beat.
3. **Which terrains matter** (forest, arid, jungle, upland), since held-out regions are chosen from that.
4. **Compute you will spend.** A 10,000 km² run is about 5 hours per 10 passes at the small network; unmeasured for
   larger ones.
5. **Release and sensitivity policy.** Weights trained on OpenStreetMap-derived labels or non-commercial terrain
   cannot all be Apache-2.0; the manifest decides. Detections of unrecorded sites stay local by default.
