# PathologyAI

PathologyAI is a student-friendly **Research/Education Prototype** for organizing
pathology images for human review. It validates uploaded images, checks basic
presentation quality, shows a deterministic demonstration attention map, and
suggests a review order.

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
findings. Possible cropping is reported only when frame-edge patterns make it
reasonably detectable, and the message asks for manual verification.

## Model Attention demonstration

No trained EfficientNet-B0/PCam pathology model is bundled. The included
attention view is a repeatable, deterministic visualization based on edges,
local contrast, and color variation. It is clearly labeled as a demonstration
and must not be interpreted as a learned pathology finding, cancer prediction,
or medical conclusion.

The viewer supports zoom, pan, fit, and reset when its interactive component is
available, with a built-in static/zoom fallback so image review remains usable.

## Manual review and session state

Reviewer notes, confirmed or overridden priorities, and reviewed status are kept
in Streamlit session state and associated with the image content. They remain
available during the current browser session but are not saved to a database,
written into the source image, exported to an EMR, or shared automatically.
Starting a new session or clearing Streamlit state removes them.

## Project layout

- `app.py` - Streamlit interface, dashboard, viewer, and review controls.
- `pathology_ai/pipeline.py` - input validation and safe in-memory ZIP handling.
- `pathology_ai/quality.py` - deterministic image-quality checks.
- `pathology_ai/triage.py` - the three review-order labels and sorting rules.
- `pathology_ai/attention.py` - demonstration attention provider and future
  model-adapter interface.
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

## Adding a local model later

A future reviewed local adapter can implement `AttentionProvider.analyze()` in
`pathology_ai/attention.py` and be selected by `get_attention_provider()`. An
ImageNet-pretrained EfficientNet-B0 or locally available PCam-derived model can
be integrated there without changing the upload or review UI. Keep weights
local, do not auto-download them, label the model and its limitations clearly,
and validate it separately before making any performance claim.
