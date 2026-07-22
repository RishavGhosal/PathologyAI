# PathologyAI

PathologyAI is a student-friendly **Research/Education Prototype** for organizing
pathology images for human review. It validates uploaded images, checks basic
presentation quality, builds a review queue, and records reviewer feedback.
When explicitly enabled, it can also use a local UNI or Hibou-B encoder for an
exploratory feature-variation visualization and a separate experimental MHIST
annotator-agreement proxy head for review ordering.

> **This research and education prototype provides review-priority suggestions only. It does not provide a medical diagnosis and does not replace review by a qualified pathologist.**

## Review Priority labels

The three labels describe queue order only:

| Label | Meaning |
| --- | --- |
| **Review First** | The image passed the quality checks and the active review-order method placed it earlier in the human-review queue. This is not a disease likelihood or clinical urgency. |
| **Needs Better Image** | A blocking quality check found blur, very small dimensions, excessive darkness or brightness, or a blank/nearly uniform image. A clearer image is requested. Possible edge truncation is shown separately as a nonblocking advisory. |
| **Lower Priority** | The image passed the quality checks and the active review-order method placed it later in the queue. It still requires human review. |

A reviewer may confirm or override a suggestion, add notes, assign a
de-identified case/slide group ID, and mark an image reviewed. Overrides remain
review-priority choices, not diagnoses.

## Run the app

Install the frontend dependencies and create the production build from
PowerShell in the project folder:

```powershell
npm.cmd ci
npm.cmd run build
```

Then start the Python server:

```powershell
.\venv\Scripts\Activate.ps1
python app.py
```

This automatically opens the built React frontend at `http://127.0.0.1:8501`.
`app.py` serves `dist/index.html` and its hashed assets alongside the local HTTP
API, which calls the existing `pathology_ai` processing, triage,
dashboard-metric, and export utilities. No `templates/` directory or Streamlit
server is used. Press `Ctrl+C` in the PowerShell window to stop the server.

Activation is optional; the equivalent explicit command is:

```powershell
.\venv\Scripts\python.exe app.py
```

The app binds to `127.0.0.1:8501` by default. Set `PATHOLOGYAI_PORT` before
launching if that local port is already in use. Run `npm.cmd run build` again
after changing the React frontend; the generated `dist/` directory is retained
so ordinary launches only require `python app.py`.

Only if the environment must be recreated, install the small runtime dependency
set with:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The app does not need internet access at runtime and does not download model
weights.

### Optional local UNI mode

UNI is optional. Install the CPU inference packages with:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-uni.txt
```

Place the gated checkpoint at `models/uni/pytorch_model.bin`. The checkpoint is
ignored by Git and must not be redistributed. When the checkpoint and optional
packages are ready, select **Local UNI feature exploration** in the upload
settings form.
The first analysis loads the approximately 1.2 GB ViT-L checkpoint and may be
slow or use substantial memory on CPU. Select the deterministic option for the
faster fallback. Systems with a supported NVIDIA
GPU should install a matching PyTorch build using the official PyTorch selector
instead of the CPU pins in `requirements-uni.txt`.

If the local quick prototype head also exists under
`models/review_priority_head/` and `requirements-training.txt` is installed, a
the adjacent model checkbox enables its MHIST annotator-agreement proxy. The head
requires UNI and automatically falls back to the deterministic rule if it is
missing, disabled, incompatible, or fails on an image.

### Optional local Hibou-B mode

Hibou-B is an Apache-2.0 pathology feature encoder used here only for local,
exploratory feature-variation maps. Accept its Hugging Face access terms, then
place the complete `histai/hibou-b` snapshot in `models/hibou-b/` and install:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-hibou.txt
```

The snapshot and any Hugging Face token remain ignored by Git. Choose **Local
Hibou-B feature exploration (CPU)** in Model Settings after it is detected.
It is not a tissue classifier, diagnosis model, or review-priority model, and
it does not use the UNI-only MHIST proxy head. The app deliberately does not
download model weights or remote code during normal use.

## Publish safely to GitHub

The repository includes a `.gitignore` that excludes the virtual environment,
Python caches, environment files, uploaded data, and model
weights. UNI weights and Hugging Face access tokens must never be committed or
redistributed.

Before the first push, create an empty GitHub repository without an additional
README or `.gitignore`, then run:

```powershell
git init -b main
git add .
git status
git commit -m "Add PathologyAI research prototype"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Always inspect `git status` before committing. The included GitHub Actions
workflow type-checks and tests the React frontend, creates its production build,
then runs the Python syntax check and test suite for pushes to `main` and pull
requests. No Hugging Face token or model weight is needed by these tests.

This repository does not currently declare an open-source license. Add one only
after choosing terms that are appropriate for your own code and compatible with
all third-party assets. The separately downloaded UNI model remains governed by
its own non-commercial license and must not be placed in this repository.
Reuse information for the small test images that are safe to publish is recorded
in `THIRD_PARTY_NOTICES.md`; local images with unverified terms are ignored.

## Inputs and safeguards

- Direct uploads: PNG, JPG/JPEG, and TIFF (`.tif` or `.tiff`).
- Batch uploads: ZIP files containing supported images. Supported images in
  nested folders are read in memory; files are not extracted to disk.
- Each image is limited to 40 MB and 40 million decoded pixels.
- A complete upload batch is limited to 60 million decoded pixels so many
  individually valid images cannot exhaust application memory.
- A ZIP is limited to 100 MB compressed, 300 files, and 250 MB total
  uncompressed content. Suspicious compression ratios, unsafe paths, encrypted
  entries, system metadata, unsupported types, empty files, and unreadable or
  corrupted files are skipped with a reason.
- For a multi-page TIFF, this MVP previews and evaluates the first frame.

The Image Quality checks are conservative presentation checks, not pathology
findings. Corruption, very small dimensions, extreme darkness/brightness,
blankness, and blur are blocking failures that produce `Needs Better Image`.
Possible cropping or edge contact is an advisory only because tissue commonly
touches the edge of a microscopy field; it does not reject an otherwise usable
image.

## Model attention and local encoder limitations

The always-available attention view is a repeatable, deterministic visualization
based on edges, local contrast, and color variation. The optional local UNI mode
loads the approved pretrained pathology encoder and compares its patch-token
representations within a letterboxed 224 × 224 copy of the image. The resulting
overlay shows relative feature variation in that resized view.

UNI and Hibou-B are feature encoders, not diagnosis models or review-priority
classifiers. Hibou-B is CPU-only in this prototype and does not feed the MHIST
agreement-proxy head; that head remains UNI-specific.
When the optional local prototype head is enabled, UNI supplies a 1,024-value
embedding and the separate head suggests `Review First` or `Lower Priority`.
Otherwise those labels use the deterministic visual-complexity fallback.
`Needs Better Image` always comes from image-quality checks and overrides the
head. The UNI overlay is exploratory and must not be interpreted as a
validated clinical attention map, pathology finding, cancer prediction, or
medical conclusion. No weights are downloaded at runtime.

The included quick head was fitted to an MHIST colorectal-polyp
annotator-agreement proxy. Other tissues are out of domain. Its score is not
calibrated confidence, disease likelihood, or clinical urgency. It exists only
so the model-to-browser flow can be tested before a more carefully trained
Kaggle artifact replaces it.

The browser viewer shows the original image, feature overlay, and heatmap. The
browser's native image controls can be used to inspect a preview.

## Manual review and session state

`Review Queue` is the default tab. It sorts uploaded images by effective
reviewer priority, experimental proxy score when present, and filename.
Reviewers can filter by Awaiting/Reviewed/All and priority, choose an image,
inspect its original/overlay/heatmap views, save a review, reopen one, or save
and move to the next matching unreviewed image.

A de-identified case/slide group ID is optional for review and export, but is
required by the later grouped-training preparation workflow. Reviewer notes are
required only when the suggested priority is overridden. The bulk group action
applies the current ID to every image from the same top-level uploaded source;
it does not infer case membership from filenames, folders, or ZIP structure.

Reviewer notes, group IDs, confirmed or overridden priorities, completion time,
and reviewed status are kept in a local in-memory browser session and associated
with the image content. They are not saved to a database, written into the
source image, exported to an EMR, or shared automatically. Starting a new batch
or restarting the server removes them.

## Operational and model dashboards

`Operational Dashboard` reports queue progress, effective priority counts,
blocking image-quality findings, nonblocking advisories, reviewer agreement,
and deterministic/experimental/quality-gate/runtime-fallback provenance.

The configurable screening baseline defaults to 30 seconds per image. The
displayed **Estimated time avoided reviewing unusable images** is calculated as
`(Needs Better Image + skipped/failed inputs) × configured seconds`. It is a
user-configured estimate for unusable inputs only, not measured workflow
efficiency and not evidence that priority ranking saves time.

`Model Evaluation & Limits` displays the local feature-provider status and the
validated experimental-head metrics exposed by the server. Numeric values use
three decimal places. Missing or invalid evaluation values are never invented.

## Reviewed-label research export

Each reviewed image has an optional de-identified slide/case grouping code,
reviewer notes when needed, a confirmed/overridden priority, and completion
metadata. The export panel downloads a CSV for future review-priority model
development. It includes only reviewed images from the current batch, structured
priority and fallback provenance, stable quality codes and readable reasons,
domain declaration, proxy score, and 1,024 UNI feature columns when UNI
succeeded. Other feature encoders export their model metadata and dimension for
audit; their vectors are not placed into the fixed UNI training columns. It
excludes raw images, filenames, upload paths, and unreviewed rows.

For batches scored by the experimental head, the dashboard also bins
agreement-proxy scores by reviewer-confirmed, overridden, and awaiting-review
outcomes. This is a descriptive current-batch view, not a calibration plot or
clinical-confidence display.

The app checks group-ID length and format but cannot determine whether a value is
actually de-identified. Before download, the reviewer must confirm that notes and
group IDs contain no names, medical record numbers, dates of birth, or other
identifying information. The CSV is local and unencrypted. It is a
research/education review-order artifact, not a diagnostic dataset.

Multiple exports can be validated and prepared for Kaggle with:

```powershell
.\venv\Scripts\python.exe scripts\prepare_review_training_data.py `
  export-one.csv export-two.csv `
  --output-dir data\prepared-review-training
```

The utility removes exact duplicates, rejects conflicting image IDs, requires
valid group IDs and finite 1,024-value embeddings, and reports label/group
balance. It creates a fixed deterministic five-fold `StratifiedGroupKFold`
manifest only when both labels occur across at least five distinct groups and
every fold contains both labels with no group leakage. It never reduces the fold
count. Failure writes `validation_report.json`, exits nonzero, and leaves no
split-ready manifest or embedding artifact.

## Project layout

- `app.py` - standard-library local HTTP API, session state, and safe serving of
  the Vite production build.
- `index.html` and `src/` - Vite entrypoint and typed React frontend source.
- `dist/` - generated production frontend served by `app.py`.
- `pathology_ai/pipeline.py` - input validation and safe in-memory ZIP handling.
- `pathology_ai/quality.py` - deterministic image-quality checks.
- `pathology_ai/triage.py` - the three review-order labels and sorting rules.
- `pathology_ai/attention.py` - demonstration attention provider and future
  model-adapter interface.
- `pathology_ai/uni_provider.py` - optional local UNI loader and exploratory
  feature-variation overlay.
- `pathology_ai/review_export.py` - reviewed-only, de-identified label and UNI
  embedding CSV export.
- `pathology_ai/dashboard_metrics.py` - pure operational metric calculations.
- `pathology_ai/review_model.py` - optional trusted-local prototype head loader,
  metadata checks, score validation, and graceful fallback support.
- `requirements-uni.txt` - optional CPU packages for local UNI inference.
- `requirements-training.txt` - optional dependencies for extracting MHIST UNI
  embeddings and training the experimental downstream head.
- `scripts/extract_uni_embeddings.py` - resumable, batched MHIST UNI extraction.
- `scripts/train_review_head.py` - validated training and official-split
  evaluation for the agreement-proxy classifier.
- `scripts/prepare_review_training_data.py` - combines reviewer exports and
  creates group-safe Kaggle artifacts only after validation passes.
- `tests/test_pipeline.py` - in-memory upload, failure, ZIP, quality, and label
  tests.
- `tests/test_app.py` - HTTP API, Vite asset, session, image, review, and export
  smoke tests.

## Verification

Run the frontend checks and production build, followed by the Python checks:

```powershell
npm.cmd run typecheck
npm.cmd test
npm.cmd run build
.\venv\Scripts\python.exe -m compileall -q app.py pathology_ai scripts tests
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\venv\Scripts\python.exe -m pip check
```

Then start the app and manually confirm the viewer and review workflow with a
valid PNG/JPG, an unsupported file (directly or inside a ZIP), a corrupted
image, a blurry or very small image, a mixed valid/invalid ZIP, and a ZIP with
no supported images.

## Experimental MHIST training workflow

MHIST can be used for a limited experiment in which seven-pathologist agreement
acts as a proxy for review difficulty. This is not a clinical priority label.
Votes of 2-5 for SSA map to `Review First`; votes of 0, 1, 6, or 7 map to
`Lower Priority`. `Needs Better Image` remains controlled only by image-quality
checks. The MHIST majority-vote disease category is retained only for auditing
and is never passed to the classifier as a feature.

With `annotations.csv` and the extracted images under `data/mhist/`, run:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\venv\Scripts\python.exe scripts\extract_uni_embeddings.py --threads 6
.\venv\Scripts\python.exe scripts\train_review_head.py --threads 6
```

The extractor resumes from committed batches in
`data/mhist/uni_embeddings.sqlite3`. The trainer fits only on MHIST's official
train partition and evaluates once on its official test partition. Local data,
embeddings, checkpoints, and trained model artifacts are ignored by Git.

The supplied MHIST annotations do not include patient/case/slide group IDs, so
patient-level leakage cannot be independently verified. Reported metrics measure
prediction of this dataset-specific agreement proxy only; they are not clinical
accuracy, diagnosis accuracy, or evidence of real-world review urgency.

The model-evaluation tab includes the seven-pathologist agreement distribution,
a compact dataset/model card, and a threshold explorer based on precomputed
held-out MHIST proxy outcomes. The explorer changes only the displayed
evaluation summary; it never changes the active model threshold or represents a
calibrated probability.

## Replacing the quick prototype head

The app can now load the fixed trusted-local artifact at
`models/review_priority_head/review_priority_head.joblib` together with its
`metadata.json` and optional `metrics.json`. These files are ignored by Git and
are never accepted through image upload. A future Kaggle-trained replacement
must keep the same 1,024-value UNI input and two review-order classes, and should
be copied with matching metadata only after its provenance, train-only model
selection, threshold, limitations, and patient/slide-level evaluation are
documented. Raw UNI magnitude or an arbitrary threshold is not a valid model.
