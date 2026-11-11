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
- Ranking at the top of the list is weak (13% recall in the top 25 per tile).
- Linear detections still include hedge banks, tracks and valley edges.
- Enclosures, hillforts and field systems are not detected.

## Roadmap

- Prompt-by-example search: a self-supervised terrain encoder (optional `embed` extra, uses PyTorch) so a
  few clicked examples find similar structures without labels.
- A local click-to-find viewer.
- Profiling and low-level optimisation of the ingest and feature stages.

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
