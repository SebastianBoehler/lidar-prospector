# lidar-prospector

Find earthworks in open airborne lidar. Stream only the window you need, build a bare-earth terrain
model, and surface candidate mounds, pits and linear banks, with a benchmark that scores the detectors
against known monuments.

> **Status: early research prototype.** Measured on English chalk downland only. Output is a ranked list
> of *candidates*, not discoveries. See [Limitations](#limitations).

## What is in here

| Piece | What it does | Module |
|---|---|---|
| Point-cloud streaming | Fetch only the octree tiles that intersect a window from USGS 3DEP (EPT on S3, no login) | `ept`, `plan`, `run` |
| Terrain | Bare-earth grid from classified ground returns, local relief, multi-direction hillshade | `terrain` |
| English terrain model | 1 m Environment Agency LiDAR composite DTM windows via WCS | `wcs` |
| Features | 13 multi-scale per-pixel descriptors (band-pass relief, Hessian shape, slope, roughness) | `features` |
| Barrow / mound detector | Gradient-boosted pixel classifier, tested leave-one-tile-out | `learn`, `crossval` |
| Hand-tuned baseline | Relief thresholds with erosion suppression | `detect` |
| Pits and linear earthworks | Closed-depression finder; bank/ditch finder with an OpenStreetMap road mask | `shapes`, `roads` |
| Labels and benchmark | Historic England scheduled monuments as ground truth | `labels`, `bench` |
| Terrain encoder (optional `embed` extra) | Self-supervised CNN trained on unlabelled terrain patches, no labels | `embed`, `train_embed` |
| Prompt-by-example search | Linear probe on embeddings: click a few examples, find similar places; simulated-user benchmark | `retrieve`, `prompt_bench` |
| Click-to-find viewer | Local web page: click examples, press search, accept or reject results | `serve` |
| Discovery review | Look-alikes of known barrows outside scheduled areas, as a contact sheet | `discover` |
| Profiler | Wall-clock and throughput per pipeline stage | `profile_stages` |

## Results

Leave-one-tile-out on five 3 km tiles of Wessex chalk downland around Stonehenge: train on four tiles,
score on the fifth. 119 held-out single barrows pooled; a hit is a candidate within 10 m of a scheduled
barrow polygon; K is candidates per 9 km² tile.

| K | Learned | Hand-tuned | Random |
|---|---|---|---|
| 25 | 13% | 3% | 1% |
| 50 | 24% | 7% | 1% |
| 100 | 35% | 15% | 3% |
| 200 | 50% | 25% | 6% |

### Prompt-by-example search

A 2.3 MB encoder trained without labels (about 9 minutes on an M4 Max) embeds every 64 m patch. A
simulated user marks N known barrows as examples and the system ranks the tile; the table is recall of
the remaining scheduled barrows at K = 200 candidates per tile (3 tiles, 114 barrows, 30 trials).

| Examples given | Search (linear probe) | Random |
|---|---|---|
| 1 | 19% | 6% |
| 3 | 31% | 5% |
| 5 | 34% | 6% |
| all ~114 (leave-one-tile-out) | 49% | 6% |

Matching the mean of the examples is clearly worse (25% with 5 examples); the probe also learns what to
ignore. Two rounds of accept/reject feedback add only 1-2 points on top of the probe. Mining hard
negatives lifts the all-labels case from 50% to 53%; a gradient-boosted head on the same embeddings is
not better (47-48%), so the ceiling is the representation and the label count, not the classifier.

### Rule-based detectors

The pit and linear-earthwork detectors are rule-based. The linear detector, after masking mapped roads,
puts 24% of its line pixels inside scheduled linear-monument zones that cover 8% of the tile (a 3x lift on
a single tile; the labels are whole protected areas, so this is a weak check). The pit detector has no
labels to score against yet.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m lidarprospector.plan --project LA_NortheastDOTD_2_2017 \
    --lat 32.637 --lon -91.407 --half-m 500        # dry run: how much data would this fetch?
uv run python -m lidarprospector.run --project LA_NortheastDOTD_2_2017 \
    --lat 32.637 --lon -91.407 --half-m 500 --utm-epsg 32615
uv run python -m lidarprospector.bench --bbox 410800 140800 413800 143800
uv run python -m lidarprospector.crossval
```

Prompt-by-example search needs the optional PyTorch extra and trains its own encoder (weights are not
published yet; `output/` is gitignored):

```bash
uv sync --extra embed
uv run python -m lidarprospector.crossval                 # downloads the five benchmark tiles first
uv run python -m lidarprospector.train_embed --steps 4000 # ~9 min on Apple Silicon (Metal)
uv run python -m lidarprospector.prompt_bench             # simulated-user benchmark
uv run python -m lidarprospector.serve --bbox 410800 140800 413800 143800   # click-to-find viewer
uv run python -m lidarprospector.discover                 # contact sheets of unscheduled look-alikes
uv run python -m lidarprospector.profile_stages
```

Commands that fetch data cache it under `data/` (gitignored). Rough sizes: a 1 km² window of dense US
lidar is about 165 MB; one 3 km English DTM tile is about 38 MB. `plan` reports the size first.

## Data sources and licences

| Data | Source | Licence |
|---|---|---|
| Airborne lidar point clouds (US) | [USGS 3DEP on AWS](https://registry.opendata.aws/usgs-lidar/) | US Government public data |
| 1 m terrain model (England) | Environment Agency LIDAR Composite DTM | Open Government Licence v3.0, (c) Environment Agency copyright and/or database right |
| Scheduled monuments (labels) | Historic England | Open Government Licence v3.0, (c) Crown Copyright and database right; contains Ordnance Survey data |
| Roads and railways (mask) | OpenStreetMap via Overpass | ODbL, (c) OpenStreetMap contributors |

## Responsible use

Locations of unrecorded archaeological sites are sensitive; publishing them invites looting. Treat
detections as hypotheses, check them against the relevant heritage record, and report plausible new sites
to the responsible authority (for example the local Historic Environment Record in England, or INAH in
Mexico) rather than publishing coordinates.

## Limitations

- Tested only on open English downland: short grass, excellent lidar, well-mapped monuments. Forest,
  jungle, wetland and non-English terrain are untested.
- Labels are *scheduled* monuments, the known and protected ones, so recall says nothing about finding
  new ones. Unlabelled candidates are not necessarily false, and not necessarily real.
- Ranking at the top of the list is weak (13-18% recall in the top 25 per tile).
- Discovery output is not credible yet. Once barrows inside scheduled areas are removed, the top-ranked
  unscheduled places are mostly field boundaries, ditches and lynchets, so treat the contact sheets as a
  review aid, not a list of finds.
- The encoder is trained at 1 m resolution only, on five tiles of one landscape; other resolutions and
  structure sizes need retraining with scale augmentation.
- Tiles are fixed 3 km boxes with edge masking, so a structure across a tile seam can be cut or counted
  twice; a halo-and-merge tiling engine is planned.
- Linear detections still include hedge banks, tracks and valley edges.
- Enclosures, hillforts and field systems are not detected.

## Performance (one 9 km² tile, Apple M4 Max)

| Stage | Seconds |
|---|---|
| Download 1 m terrain tile (network) | 44-53 |
| 13-channel feature stack (NumPy/SciPy, CPU) | 7.6 |
| Encoder over the tile (Metal, stride 8) | 6.3 |
| Mound + pit + linear detectors | 5.1 |

The pipeline is network-bound today, so the first optimisation is overlapped, parallel downloading;
after that the feature stack and the encoder (which recomputes heavily overlapping windows) are the
compute hotspots.

## Roadmap

- Tiling engine: halo and seam merging so structures across tile edges are not lost, plus a resolution
  pyramid for large structures (hillforts, cursus), with user control over structure size.
- Source adapters for other countries' open lidar (US, England done; Canada's COPC on AWS is next).
- Hardware-adaptive runtime: pipelined I/O, accelerator-agnostic filters, memory budget from available RAM.
- Published pretrained encoder weights.

## Development

```bash
uv sync --group dev
uv run ruff check . && uv run ruff format --check .
uv run pytest
uv run pre-commit install   # optional
```

CI runs the same checks on Python 3.11 and 3.13. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0, see [LICENSE](LICENSE).
