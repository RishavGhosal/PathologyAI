# PathologyAI

PathologyAI is a student-friendly **Research/Education Prototype** for organizing
pathology images for human review. It validates uploaded images, checks basic
presentation quality, shows a deterministic demonstration attention map, and
suggests a review order. When explicitly enabled, it can also use a local UNI
encoder for an exploratory feature-variation visualization.

> **This research and education prototype provides review-priority suggestions only. It does not provide a medical diagnosis and does not replace review by a qualified pathologist.**

## Review Priority labels

The three labels describe queue order only:

| Label | Meaning |
| --- | --- |
| **Review First** | The image passed the quality checks and the deterministic demonstration found comparatively strong visual texture or contrast, so it is placed earlier in the human-review queue. This is not a disease likelihood. |
| **Needs Better Image** | One or more quality checks found a problem such as blur, very small dimensions, excessive darkness or brightness, a blank/nearly uniform image, or possible edge truncation. A clearer or more complete image is requested. |
| **Lower Priority** | The image passed the quality checks and has comparatively less visual variation under the demonstration rules. It still requires human review. |

A reviewer may confirm or override a suggestion, add notes, and mark an image as
reviewed. Overrides remain review-priority choices, not diagnoses.

## Run the app

The existing `venv` already contains the packages used by this MVP. From
PowerShell in the project folder:

```powershell
.\venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Activation is optional; the equivalent explicit command is:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

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
packages are ready, the sidebar toggle **Use local UNI feature visualization**
is enabled by default. The first analysis loads the approximately 1.2 GB ViT-L
checkpoint and may be slow or use substantial memory on CPU. The toggle can be
disabled for the faster deterministic fallback. Systems with a supported NVIDIA
GPU should install a matching PyTorch build using the official PyTorch selector
instead of the CPU pins in `requirements-uni.txt`.

## Publish safely to GitHub

The repository includes a `.gitignore` that excludes the virtual environment,
Python caches, Streamlit secrets, environment files, uploaded data, and model
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
workflow runs the syntax check and test suite for pushes to `main` and for pull
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

## Model Attention and UNI limitations

The always-available attention view is a repeatable, deterministic visualization
based on edges, local contrast, and color variation. The optional local UNI mode
loads the approved pretrained pathology encoder and compares its patch-token
representations within a letterboxed 224 × 224 copy of the image. The resulting
overlay shows relative feature variation in that resized view.

UNI is a feature encoder, not a diagnosis model and not a trained
review-priority classifier. In this MVP, UNI does **not** generate `Review First`
or `Lower Priority`; those labels continue to use the documented deterministic
visual-complexity rule, while `Needs Better Image` comes from image-quality
checks. The UNI overlay is exploratory and must not be interpreted as a
validated clinical attention map, pathology finding, cancer prediction, or
medical conclusion. No weights are downloaded at runtime.

The viewer supports zoom, pan, fit, and reset when its interactive component is
available, with a built-in static/zoom fallback so image review remains usable.

## Manual review and session state

Reviewer notes, confirmed or overridden priorities, and reviewed status are kept
in Streamlit session state and associated with the image content. They remain
available during the current browser session but are not saved to a database,
written into the source image, exported to an EMR, or shared automatically.
Starting a new session or clearing Streamlit state removes them.

## Reviewed-label research export

Each image can be assigned an optional de-identified slide/case grouping code,
reviewer notes, a confirmed/overridden priority, and reviewed status. After at
least one image is marked reviewed, the export panel can download a CSV for
future review-priority model development. The CSV includes only reviewed images
from the current batch, their reviewer labels, quality provenance, and 1,024 UNI
feature columns when UNI succeeded. It excludes raw images, filenames, upload
paths, and unreviewed rows.

The reviewer must confirm that notes and grouping codes contain no names,
medical record numbers, dates of birth, or other identifying information. The
CSV is local and unencrypted. It is a research/education review-order artifact,
not a diagnostic dataset.

## Project layout

- `app.py` - Streamlit interface, dashboard, viewer, and review controls.
- `pathology_ai/pipeline.py` - input validation and safe in-memory ZIP handling.
- `pathology_ai/quality.py` - deterministic image-quality checks.
- `pathology_ai/triage.py` - the three review-order labels and sorting rules.
- `pathology_ai/attention.py` - demonstration attention provider and future
  model-adapter interface.
- `pathology_ai/uni_provider.py` - optional local UNI loader and exploratory
  feature-variation overlay.
- `pathology_ai/review_export.py` - reviewed-only, de-identified label and UNI
  embedding CSV export.
- `requirements-uni.txt` - optional CPU packages for local UNI inference.
- `tests/test_pipeline.py` - in-memory upload, failure, ZIP, quality, and label
  tests.
- `tests/test_app.py` - Streamlit startup, dashboard, viewer configuration, and
  session-review workflow smoke tests.

## Verification

Run the syntax check and automated test suite:

```powershell
.\venv\Scripts\python.exe -m compileall -q app.py pathology_ai tests
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Then start the app and manually confirm the viewer and review workflow with a
valid PNG/JPG, an unsupported file (directly or inside a ZIP), a corrupted
image, a blurry or very small image, a mixed valid/invalid ZIP, and a ZIP with
no supported images.

## Adding a review-priority model later

To let UNI influence `Review First` or `Lower Priority`, first train and validate
a separate downstream review-priority head on appropriate human review-order
labels, using patient- or slide-level data splits. Keep that head disabled until
its provenance, threshold, limitations, and evaluation are documented. Raw UNI
embedding magnitude or an arbitrary threshold is not a valid substitute for a
trained head.
