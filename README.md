# lidar-prospector

Find earthworks in open airborne lidar. Stream only the window you need, build a bare-earth terrain
model, and surface candidate mounds, pits and linear banks, with a benchmark that scores the detectors
against known monuments. One command scans an area on whatever hardware you have.

> **Status: early research prototype.** Measured on English chalk downland only. Output is a ranked list
> of *candidates*, not discoveries. See [Limitations](#limitations).

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra embed                       # PyTorch gives GPU (Metal / CUDA) acceleration
uv run python -m lidarprospector.runtime    # "doctor": what this machine can do and the plan it makes

# scan 6 km x 6 km around a point (England: ~150 MB download, cached under data/; not yet run live,
# the GeoTIFF form below is the one exercised end to end)
uv run python -m lidarprospector.scan --source england --lonlat -1.8262 51.1789 --size-km 6

# any elevation GeoTIFF / cloud-optimised GeoTIFF, local or over HTTP; licence text is required
uv run python -m lidarprospector.scan --source geotiff:https://example.org/dem.tif \
    --licence "<publisher's licence>" --attribution "<publisher's credit>" --bbox XMIN YMIN XMAX YMAX
```

The output is GeoJSON (WGS84) with the data licence and attribution embedded. To add mound detection,
train the U-Net on the benchmark tiles and pass it in:

```bash
uv run python -m lidarprospector.crossval                         # downloads the five benchmark tiles
uv run python -m lidarprospector.train_unet                       # 1,500 steps, ~2 min on an M4 Max
uv run python -m lidarprospector.scan --source england --lonlat -1.8262 51.1789 --size-km 6 \
    --mounds output/unet.pt
```

## How it scales

- **Hardware-adaptive.** `runtime` detects cores, *currently available* RAM and the accelerator, and
  sizes tiles and workers to fit a memory budget (default: half of free RAM after a 4 GB reserve). Other
  apps share the machine, so a memory gate re-checks before each tile and waits if RAM was taken meanwhile.
  Re-run the doctor under load and the plan shrinks accordingly.
- **Seam-safe tiling.** Each tile is processed with a halo of context and every detection is owned by
  exactly one tile; line fragments that meet at a seam are joined before length filtering. A tiled scan
  reproduces a one-big-tile scan exactly for mounds and lines (tested on synthetic and real terrain).
  Pits match except for rare borderline blobs (1 of 40 on the real tile).
- **Pipelined.** Downloads run in parallel with compute.
- **Structure-size control.** `--min-size-m` / `--max-size-m` pick the pyramid levels for lines, and
  `--max-mound-m` adds coarser levels for the mound network (de-duplicated across levels), so structures
  of hundreds of metres are searched on coarser grids.
- **Accelerator-agnostic filters.** The 13-channel feature stack is separable convolutions in PyTorch:
  0.2 s on Metal versus 7.8 s in SciPy for a 9 MP tile (CPU torch is slower than SciPy, so the CPU
  path stays SciPy).

## Streaming

Real streaming needs a source that supports HTTP range requests: a cloud-optimised GeoTIFF (COG). For those,
a scan reads only the blocks each tile touches, shares one dataset handle so overlapping halos are not downloaded
twice, and shows results as tiles finish. Measured with real range requests against a local server with a shared
bandwidth cap and latency (`python -m lidarprospector.stream_bench`), on an 18 km² file that is natively 1 m:

| | 20 Mbit/s | 5 Mbit/s |
|---|---|---|
| Download everything, then process | first result 42 s, total 42 s | 100 s, 100 s |
| Stream tiles at 1 m | first result **3.3 s**, total 24 s | **9 s**, 60 s |
| Coarse look from the overviews (4 m cells) | 0.9 s, 8 s total, 4.2 MB | 1.6 s, 12 s total |

The streamed scan fetches 34.6 MB against a 47.2 MB file (empty blocks are never requested); a coarse look
costs 4.2 MB, 11x less, and finds structures a few tens of metres or larger. Parallel reads do not speed it up
(the link is the limit; they only hide latency), and without the shared handle the same scan fetched 107 MB.
These are a simulated link and one synthetic COG built from two Environment Agency tiles, not a remote server.

Three things this is not. (1) The England (WCS) and USGS sources download whole windows, with no range
requests; the Dutch WCS can scale on the server, so coarse looks are cheap there too. (2) Pruning: fetching only
the blocks a coarse pass scores highest costs recall for small objects (half the blocks keeps 73% of the
barrows with the best scorer, against 49% for random blocks), so it is not the default; the savings are real
for large structures. (3) GDAL's HTTP driver was fetching a whole directory listing on every open, which the
GeoTIFF source now disables.

## Terrain sources

Every source returns the same object: a metric, north-up elevation grid plus its licence and attribution.

| Source | Status | Notes |
|---|---|---|
| England, Environment Agency 1 m DTM (WCS) | verified on the benchmark windows; a live multi-tile `scan --source england` has not been run yet | OGL v3 |
| Any GeoTIFF / COG, local or HTTP, any CRS | verified | windowed reads and on-the-fly reprojection; checked against a remote global 30 m COG and a synthetic lat/lon raster |
| Netherlands AHN 0.5 m DTM (PDOK WCS, server-side scaling to 1 m) | verified: four tiles downloaded (88 MB) and scored | CC0 1.0 |
| Scotland SRSP DTM (public S3 tiles: 1 m in phases 1-2, 50 cm in 3-6; range-readable) | adapter built; tile index (1,518 tiles) and a real 128 m window read verified; not yet scanned or scored | OGL v3 |
| OpenTopography UAV lidar, Quiriguá Guatemala (`--source preset:quirigua`) | verified: file downloaded (28.6 MB) and scanned | CC BY 4.0, 0.25 m DTM under canopy; terms read from the publisher's page |
| USGS 3DEP point clouds (EPT, no login) | built from earlier-verified pieces | gridded from classified ground returns; not re-run live after the refactor; needs the survey name (`ept:PROJECT`) |
| Canada CanElevation (COPC on AWS) | not built | COPC point clouds, Open Government Licence, listed on the AWS registry |
| Mexico INEGI | not built | free for researchers but non-commercial, ~5 m grids from sparse point clouds: too coarse for small structures |
| France, Netherlands, New Zealand, Norway, others | not checked | endpoints and licences unverified |

Add a country by implementing `fetch(bbox)` and `covers(lon, lat)`, or point the GeoTIFF source at its DEM.

## Results

Leave-one-tile-out on five 3 km tiles of Wessex chalk downland around Stonehenge: train on four tiles,
score on the fifth. 119 held-out single barrows pooled; a hit is a candidate within 10 m of a scheduled
barrow polygon; K is the number of top-ranked candidates per 9 km² tile.

### Model families compared

| Method | K=25 | K=50 | K=100 | K=200 |
|---|---|---|---|---|
| Hand-tuned relief thresholds | 3% | 7% | 15% | 25% |
| Pixel gradient boosting on 13 terrain features | 13% | 21% | 35% | 51% |
| Linear probe on self-supervised embeddings | 17% | 27% | 37% | 48% |
| U-Net from SSL-pretrained encoder | 34% | 49% | 61% | 67% |
| **U-Net from scratch (shipped)** | 29% | 50% | 67% | **75%** |
| Ensemble: GBM + U-Net(SSL) | 18% | 32% | 50% | 63% |
| Random | 1% | 1% | 3% | 6% |

A small supervised U-Net (1.0 M parameters, 2 relief channels, dihedral and amplitude augmentation)
beats the other families by a wide margin. Self-supervised pretraining helps the top of the ranking
slightly but loses at 200 candidates, and the ensemble is worse than the U-Net alone. These runs use one
seed and 1,500 training steps per model, and three of the five tiles contain 4 or fewer barrows, so
differences of a few points are within noise.

### Other landscapes (zero-shot)

The U-Net was trained on Wessex only and scored on eight tiles it never saw, in three landscapes: North
York Moors heather moorland (2 tiles), the Peak District (2) and Dartmoor (4). 135 single barrows and cairns
in total (cairns count as barrows; this changed none of the Wessex labels). Recall at K candidates per
9 km² tile:

| Method | K=25 | K=50 | K=100 | K=200 |
|---|---|---|---|---|
| Hand-tuned relief thresholds | 3% | 6% | 7% | 10% |
| Pixel gradient boosting | 12% | 19% | 26% | 32% |
| **U-Net (trained on Wessex only)** | 34% | 36% | 41% | **45%** |
| Random | 0% | 1% | 2% | 4% |

By landscape (U-Net, recall at K=25 / 100 / 200): North York Moors 32% / 41% / 45% (22 labels), Peak District
41% / 65% / 65% (17), Dartmoor 33% / 38% / 42% (96). So the U-Net carries over best of the methods, but
drops from 75% in-region to about 45% on unseen terrain, and it is a ranking, not a calibrated detector.
Seed-to-seed noise on these 135 labels is small (three seeds: +/-1 point). Training longer clearly hurts on
unseen terrain: recall at K=200 against training steps is 43% (600), 44% (1,000), **46% (1,500)**, 43% (2,500)
and 39% (4,000), so it overfits Wessex, and the default is 1,500 steps. Widening the network by 1.5x changes
nothing (46%). An earlier four-tile run (39 labels) suggested adding a few tiles of the new
landscapes does not help; that was not repeated with Dartmoor.

### A second country and a forest (Netherlands, zero-shot)

Four 3 km tiles of the Utrechtse Heuvelrug (forest and heath, AHN terrain at 1 m, CC0) with the 46
OpenStreetMap tumulus nodes inside them as labels (a hit is a candidate within 15 m). Models trained on
Wessex only:

| Method | K=25 | K=50 | K=100 | K=200 |
|---|---|---|---|---|
| Hand-tuned relief thresholds | 0% | 0% | 0% | 0% |
| Pixel gradient boosting | 13% | 15% | 24% | 33% |
| **U-Net (trained on Wessex only)** | 37% | 54% | 65% | **67%** |
| Random | 0% | 1% | 1% | 1% |

Caveat: OpenStreetMap tumuli are crowd-sourced and often mapped from lidar hillshades, so these labels likely
favour features a terrain model can see; 46 labels is also few. The ranking of the methods is the more
trustworthy part than the absolute numbers.

### Linear monuments, enclosures and field systems (coverage check)

The line finder was never scored on these. Pooled over all 13 English tiles (zones of about 1 km² per
class; labels are whole protected areas, so this is coverage, not precise recall), the share of a zone
within 10 m of a detected line, against the same share over the whole tile:

| Class | Pyramid level | Zone covered by a line | Tile-wide base rate | Lift in line length |
|---|---|---|---|---|
| Linear monuments (dykes, cursus) | 1 m | 24% | 13% | **2.0x** |
| | 2 m / 4 m / 8 m | 16% / 7% / 2% | 13% / 7% / 3% | 1.3x / 0.9x / 0.8x |
| Enclosures (hillforts, henges) | 1 m | 11% | 11% | 0.9x |
| Field systems | 1 m | 9% | 14% | 0.6x |

Only linear monuments at the finest level stand out. The detector does not find enclosures, and it
finds field systems less often than chance (they are low-relief), so it should not be read as a detector for
either. Roads are not masked in this check.

### Promptable detection across object types (no per-type training)

A prompt is a few example locations. Every pixel gets a dense feature vector, a regularised linear probe
separates the examples from background, and its peaks are the candidates, so changing the examples changes the
object type with no retraining. Tested on four object types from Historic England's scheduled monuments
(single monuments of <= 0.5 ha, 13 English tiles), with examples drawn from *other* tiles. A hit is a scheduled
polygon that contains one of the top 100 candidates in a 9 km² tile; "random" places 100 random points.

Own-type recall with 10 example prompts:

| Feature source for the probe | Round barrow (146) | Cairn (91) | Hut circle (45) | Stone setting (17) |
|---|---|---|---|---|
| Hand terrain features | 35% | 16% | 1% | 0% |
| Barrow-trained U-Net's dense features (trained on the 5 Wessex tiles: the barrow figure is **partly in-sample**, the cairn figure is on tiles it never saw) | 61%* | 32% | 2% | 6% |
| Same network trained with labels on 103 *other* tiles (about 50 labelled barrows and cairns; clean) | 41% | 27% | 1% | n/a |
| Same, trained on 133 *other* tiles (about 520 labelled barrows, cairns, hut circles, stone settings; clean) | **54%** | **42%** | **10%** | n/a |
| Same, trained on monuments of *other* types only (no barrows, cairns, hut circles; 133 tiles) | 10% | 4% | 1% | 0% |
| Self-supervised encoder features | 26% | 18% | 2% | 4% |
| Template matching on raw terrain (no learning) | 47% | 10% | 0% | 0% |
| Random placement | 2% | 1% | 1-2% | 3% |
| *Supervised, trained for the type (Dartmoor)* | n/a | *37%* | *16%* | n/a |

What this shows. Prompting works for dome-shaped mounds: 3 examples give 57% on barrows, and for cairns 10
examples (32%) come within reach of a model trained on labelled cairns (37%). It does **not** work for shapes
the base features never learned: hut circles and stone settings are at chance for every method, and even the
trained model finds only 16% of hut circles (they are small, subtle, and the polygons are protected areas, not
hut positions; a contact sheet showed many labels are offset or invisible, which is why a candidate has to fall
inside the polygon and not within 10 m of a centre). The prompts also steer weakly by *type*: a cairn or
hut-circle prompt still returns barrows (63% and 33% recall of barrows), because the probe mostly learns "mound".
Small enclosures could not be scored (2 single examples). The click-to-find viewer uses the U-Net probe:
with three clicked barrows on the Stonehenge tile, 12 of the 30 results fall on other known barrows (the previous
embedding-grid search got 4 of 30), and a query takes about 90 ms.

### Earlier prompt-by-example search (8 m embedding grid, superseded by the above)

A 2.3 MB encoder trained without labels embeds every 64 m patch; a linear probe on the embeddings turns
a few clicked examples into a search. Recall at K = 200 (3 tiles, 114 barrows, simulated user, 30 trials):

| Examples given | Search | Random |
|---|---|---|
| 1 | 19% | 6% |
| 3 | 31% | 5% |
| 5 | 34% | 6% |
| all ~114 | 49% | 6% |

Matching the mean of the examples is worse (25% with 5), and accept/reject feedback adds only 1-2 points.
The click-to-find viewer is `python -m lidarprospector.serve --bbox ...` (it needs `output/unet.pt` from `train_unet`).

### Rule-based detectors

Pits and linear earthworks are rule-based and unscored except for one check: after masking mapped roads
the linear detector puts 24% of its line pixels inside scheduled linear-monument zones that cover 8% of the
tile (3x lift; the labels are whole protected areas, so this is weak). Its thresholds are absolute
(so tiles agree at seams) and were set from the median of five tiles whose own percentile thresholds
differed up to 3x, so rough terrain yields more line detections than a per-tile rule would.

## Performance (one 9 km² tile, Apple M4 Max)

| Stage | Seconds |
|---|---|
| Download 1 m terrain tile (network) | 44-53 |
| 13-channel feature stack, SciPy | 7.8 |
| 13-channel feature stack, PyTorch on Metal | 0.2 |
| Encoder over the tile (Metal, stride 8) | 6.3 |
| Mound + pit + linear detectors, CPU | 5.1 |
| Nine-tile real-terrain scan, tiled | 13 |

Network transfer dominates; the pipeline overlaps it with compute. Memory per 9 MP tile: 370 B/px for
SciPy, about 300 B/px (host plus Metal) for the PyTorch path.

## Data sources and licences

| Data | Source | Licence |
|---|---|---|
| 1 m terrain model (England) | Environment Agency LIDAR Composite DTM | Open Government Licence v3.0, (c) Environment Agency copyright and/or database right |
| Airborne lidar point clouds (US) | [USGS 3DEP on AWS](https://registry.opendata.aws/usgs-lidar/) | US Government public data |
| Scheduled monuments (labels) | Historic England | Open Government Licence v3.0, (c) Crown Copyright and database right; contains Ordnance Survey data |
| Roads and railways (mask) | OpenStreetMap via Overpass | ODbL, (c) OpenStreetMap contributors |

## Responsible use

Locations of unrecorded archaeological sites are sensitive; publishing them invites looting. Treat
detections as hypotheses, check them against the relevant heritage record, and report plausible new sites
to the responsible authority (for example the local Historic Environment Record in England, or INAH in
Mexico) rather than publishing coordinates.

## Limitations

- Scored only in England: Wessex downland, North York Moors and Peak District (open ground, excellent
  lidar, well-mapped monuments). Forest, jungle, wetland and non-English terrain have no scored benchmark;
  the Quiriguá run is a qualitative demo. The U-Net was trained on five tiles of one landscape.
- Labels are *scheduled* monuments, the known and protected ones, so recall says nothing about finding
  new ones. Unlabelled candidates are not necessarily false, and not necessarily real.
- Mound output is ranked, not calibrated: the scan keeps the best 22 per km². In the top 25 per tile
  only about 30% of the scheduled barrows are found.
- Discovery output is not credible yet. Once barrows inside scheduled areas are removed, the top-ranked
  unscheduled places from the embedding search are mostly field boundaries, ditches and lynchets.
- The mound network is tuned to barrow-sized mounds (10-30 m) at 1 m cells; larger Maya-style platforms need
  a coarser mound level or scale-augmented training. Models and detectors assume 1 m cells. Coarser sources (such as 5 m or 30 m grids) are resampled and
  small structures are lost; the scan warns when the source is coarser than 2 m.
- Linear detections still include hedge banks, tracks and valley edges. `--mask-roads` removes the parts
  of lines that run along OpenStreetMap roads (unit-tested with synthetic roads; the live Overpass request
  in this form has not been run).
- Prompting works for dome-shaped mounds only; hut circles, stone settings and other shapes the base features
  never learned are at chance (see above), and prompts steer weakly by type.
- Enclosures, hillforts and field systems are not detected (measured: at chance or below, see above). The pit
  finder works at 1 m only.

## Roadmap

The plan for a pretrained terrain encoder is in [docs/foundation-model.md](docs/foundation-model.md). So far no
label-free pretraining objective (masked reconstruction, contrastive, dense contrastive) beats hand-built terrain
features. The best one (dense contrastive) went 12% -> 27% -> 20% on barrow prompts at 45, 154 and 1,477 km² of
unlabelled terrain (fixed training steps), so more label-free data alone has not closed the gap to hand features (35%).
What does work is labels: the same network trained on benchmark-disjoint tiles reaches 41% / 27% (barrows / cairns) with
about 50 labelled objects and 54% / 42% (hut circles 10%) with about 520, and its own detector scores 58% / 47% / 18%.
It transfers poorly to types it was not trained on (monuments of other types: 10% / 4%).

- Weights as a release download, so the quick start needs no training step.
- Ground-return-density map inside `scan`, so the tool knows where canopy hides the ground.
- More sources (Canada COPC first), verified one at a time.
- Labels for enclosures, hillforts and field systems (none open and typed enough found yet).
- Training examples of large rectilinear platforms (Maya-style), so the coarse mound levels find them.

## Development

```bash
uv sync --group dev
uv run ruff check . && uv run ruff format --check .
uv run pytest
uv run pre-commit install   # optional
```

CI runs the same checks on Python 3.11 and 3.13 without PyTorch; torch-only tests skip there. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0, see [LICENSE](LICENSE).
