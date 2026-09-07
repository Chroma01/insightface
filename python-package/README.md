# InsightFace Python Library 2.0

## License

The code of InsightFace Python Library is released under the MIT License. There is no limitation for both academic and commercial usage.

**The pretrained models we provided with this library are available for non-commercial research purposes only, including both auto-downloading models and manual-downloading models.**

InsightFace 2.0 uses ONNX Runtime for FaceAnalysis, ModelZoo, PrivateFrame, and
the desktop Evaluation Studio.

## Installation

### Choose an installation

| Use case | Command |
|---|---|
| FaceAnalysis and ModelZoo | `pip install insightface` |
| Video face blur/mosaic with the PrivateFrame API and CLI | `pip install "insightface[privateframe]"` |
| Evaluation Studio GUI, including PrivateFrame | `pip install "insightface[gui]"` |

The base package installs `onnxruntime`. The `privateframe` extra additionally
installs PyAV and PyYAML; the `gui` extra includes those dependencies plus the
Qt desktop application.

InsightFace sets `ORT_DISABLE_TELEMETRY=1` before importing ONNX Runtime; no
shell configuration is required. This also overrides an existing value of
`0`. On runtimes that support this switch, it prevents the non-Windows
telemetry uploader from starting. Older runtimes that do not recognize the
variable ignore it, so it does not introduce a newer ONNX Runtime requirement.
Import InsightFace before importing ONNX Runtime elsewhere in your process:
the switch cannot undo telemetry initialization that has already happened.
See [ONNX Runtime's telemetry documentation](https://github.com/microsoft/onnxruntime/blob/v1.29.0/docs/Privacy.md#disabling-telemetry)
for platform-specific behavior.

### Automatic Provider selection

When callers do not pass an explicit Provider list, InsightFace inspects the
Providers reported by the installed ONNX Runtime and selects the first
available entry in this order:

```text
CoreMLExecutionProvider → CUDAExecutionProvider → CPUExecutionProvider
```

Only one accelerated Provider is selected. CPU is appended as its fallback
when available. For example, a runtime that reports both CoreML and CUDA uses
CoreML + CPU, not CoreML + CUDA + CPU. Explicit `providers=[...]` arguments and
PrivateFrame's explicit `runtime.provider` setting take precedence over this
automatic policy.

Check what the current Python environment can actually use:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### macOS and CoreML

No separate InsightFace CoreML package is required. CoreML is selected
automatically only when the installed ONNX Runtime reports
`CoreMLExecutionProvider`; otherwise InsightFace uses the next available
Provider.

SCRFD uses reusable fixed-shape Sessions per detection resolution by default.
For InsightFace-managed CoreML Sessions, compiled artifacts are stored under
`~/.insightface/cache/coreml/v1`, scoped by the model and input signature. A
new signature may take longer on its first compilation. InsightFace first
tries all CoreML compute units and falls back to CPU + GPU when necessary;
valid cache hits are reused without another warmup.

### NVIDIA CUDA

The Python package intentionally has no `gpu` extra. Install InsightFace, then
replace the default runtime with the GPU distribution:

```bash
pip install insightface              # or insightface[privateframe] / [gui]
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

Do not keep `onnxruntime` and `onnxruntime-gpu` installed together. Installing
or upgrading InsightFace may install its declared `onnxruntime` dependency
again, so repeat the replacement afterward on NVIDIA systems.

### Launch the optional applications

```bash
# Desktop GUI
insightface-gui

# PrivateFrame CLI
insightface-privateframe --help
# Equivalent: python -m insightface_privateframe_bootstrap --help
```

## PrivateFrame

PrivateFrame detects and tracks faces in local videos and renders those face
regions with Gaussian blur or mosaic for privacy. It never modifies or uploads
the source video. PrivateFrame also treats the analysis JSON as a first-class
result, so an analysis can be inspected or edited and rendered again without
rerunning model inference. Stable default
names pair it with the rendered video in one output directory:

```text
video_privateframe.json
video_privateframe.mp4
```

The result JSON is a compact, portable rendering document. It stores the source
file name and video geometry, final per-frame face boxes, optional identity
decisions, and effective rendering defaults. It does not store absolute source
paths, source-video content hashes, model fingerprints, Git state, or internal tracking
evidence. When rendering it again, PrivateFrame checks the input dimensions,
frame rate, and decoded frame count rather than hashing the video contents.

With `--output-dir`, PrivateFrame derives a private
`.<input_stem>_privateframe_work` directory there; its temporary SQLite packet
cache is removed after analysis. An explicit `--workdir` overrides that runtime
location. For compatibility, using `--workdir` without `--output-dir` or
`--result` stores the JSON as `<workdir>/result.privateframe.json`; an explicit
`--result` always takes precedence.

Use `analyze` for JSON only, `process` for JSON plus the redacted video, and
`render` to render an existing or edited JSON without running the models again:

```bash
# Analysis only: writes /data/output/video_privateframe.json
insightface-privateframe analyze \
  --input /data/video.mp4 \
  --output-dir /data/output

# Analyze and render: also writes /data/output/video_privateframe.mp4
insightface-privateframe process \
  --input /data/video.mp4 \
  --output-dir /data/output

# Render after inspecting or editing the JSON
insightface-privateframe render \
  --input /data/video.mp4 \
  --output-dir /data/output
```

`analyze` and `process` use the packaged `configs/base.yaml` by default. Pass
`--config /path/to/custom.yaml` only when a custom configuration is needed.
A custom YAML automatically inherits the packaged Base, so it normally contains
only the changed fields:

```yaml
schema_version: 1
scan:
  max_analysis_fps: 15  # Lower the default 30 for faster processing.
```

An explicit `base_config` field remains available when the custom YAML must
inherit a different complete parent configuration. CLI dotted options are
applied last, after the Base and custom YAML layers.

### Choosing who to blur with reference photos

The default `recognition.mode: all` blurs every detected face and needs no
reference photos. To select particular people, put their photos together in
one folder and supply `recognition.reference_dir`. No person names or per-person
subfolders are needed; different people and multiple photos of the same person
can share the folder. Reference-photo selection is available in the GUI, CLI,
and Python API.

| Mode | People matched to the reference photos | Unmatched or uncertain people with the default `unknown_action: auto` |
|---|---|---|
| `all` | Blurred; photos are not read | Blurred |
| `blur_only` | Blurred | Kept visible |
| `exempt` | Kept visible | Blurred |

```bash
# Blur only the people shown in reference photos.
insightface-privateframe process \
  --input /data/video.mp4 --output-dir /data/output \
  --recognition.mode blur_only \
  --recognition.reference_dir /data/reference_photos

# Keep people shown in reference photos visible; blur everyone else.
insightface-privateframe process \
  --input /data/video.mp4 --output-dir /data/output \
  --recognition.mode exempt \
  --recognition.reference_dir /data/reference_photos
```

JPG, JPEG, PNG, and WebP files directly inside the reference folder are imported;
hidden files and symlinks are skipped, and identical copies are deduplicated.
Different photos of the same person remain valid references. Each photo
contributes **only its largest detected face**, so clear single-person photos
are recommended. For group photos, a log message states how many faces were
found and that only the largest was selected. If that face is unsuitable for
recognition, the photo is skipped with a reason; the importer does not switch to
a smaller face. An import summary reports used and skipped photo counts, not
the number of different people. These messages go to standard error; with
`--progress jsonl`, application progress and reference-photo diagnostics are
JSONL records. Third-party runtime diagnostics may still be plain text on
standard error; standard output remains one final status JSON object.

Both photo modes fail before video analysis if no reference photo supplies a
usable face. Valid references with no matching person in the video are a normal
completed result. Model loading and inference errors stop processing instead of
being treated as unmatched people. `--dry-run` checks configuration and file
readiness without running face detection; photo suitability is checked during
execution.

Keep `recognition.unknown_action: auto` to follow the table. Set it explicitly
to `blur` or `keep` only when a different treatment of unmatched or uncertain
people is intended; `all` always blurs every detected face. **With `keep`,
including the default `blur_only` behavior, a target person who is not recognized
can remain visible.** The effective policy and matching decisions are stored in
the result JSON, so a later `render` uses the same treatment without rereading
the reference photos or rerunning recognition.

### Analysis sampling rate

`scan.max_analysis_fps` defaults to **30** in the packaged configuration and
the GUI. It sets an approximate ceiling on how many input frames receive
regular full-frame face detection per second **of video time**. Actual
processing speed depends on the hardware, scene, and rendering cost. The
result video retains every source frame and its original frame rate.

The runtime samples at a uniform integer interval with a 5% rate tolerance:

| Source video FPS | Default 30: regular scans per video second | Lowered to 15: regular scans per video second |
|---|---|---|
| 25 | 25 (every frame) | 12.5 (every 2 frames) |
| 30 | 30 (every frame) | 15 (every 2 frames) |
| 60 | 30 (every 2 frames) | 15 (every 4 frames) |
| 15 or below | Same as source FPS (every frame) | Same as source FPS (every frame) |

This is a soft ceiling: scene changes, video endpoints, and newly discovered
tracks can trigger additional scans. Frames between regular scans are still
decoded and rendered. By default, existing face tracks use interpolated
regions between detection frames; a face visible only in that gap may be
missed.

- **Lower to 15** when faster processing and less detector work take priority,
  especially in scenes with limited motion. A lower value such as 10 can reduce
  sampling further. Wider gaps increase the risk of missing briefly visible faces;
  review representative output before using a lower rate broadly.
- **Keep 30** for fast motion, brief face appearances, frequent occlusion, or
  a need for greater detection coverage.
- **Raise toward the source video's FPS** on higher-FPS input when more temporal
  detail matters. Setting it to the source FPS scans every frame. More sampling
  costs more compute and still cannot guarantee every face is found.

Use `--scan.max_analysis_fps 15` to lower the default, or set the same field
in custom YAML. Positive fractional values are also accepted. Nearby values
may produce the same sampling interval, so performance does not change
continuously with this number. The GUI initially selects **Normal (target 30
analysis FPS)** and also offers **Fast (target 15 analysis FPS)**.

### CLI automation contract

The CLI is safe to invoke from shell scripts and vendor-neutral AI coding tools
without parsing human-oriented log text. An unfamiliar automation client should
run `insightface-privateframe describe` and read these high-level fields first:

- `tool.summary`, `tool.purpose_id`, and `tool.capabilities` explain that the
  tool detects and tracks face regions and renders Gaussian blur or mosaic.
- `discovery` maps common user intentions to the correct command and gives a
  safe dry-run/execution policy.
- `config.groups` organizes 30 common and intermediate controls for models,
  analysis, privacy policy, redaction, video, and audio. Their
  `config.dotted_options` entries contain types, defaults, constraints,
  `description`, `when_to_use`, and `tradeoff` guidance.
- `config.full_reference.path` locates the complete Markdown configuration
  reference installed with the package. Read it for advanced settings; options
  omitted from self-description still work in YAML and CLI overrides.
- `config.dotted_options["scan.max_analysis_fps"]` provides the sampling
  default, units, behavior, and `tuning_guidance` for choosing a higher or lower
  rate. Automation should read this before changing the analysis rate.
- `primary_io` distinguishes file artifacts from the final status JSON on
  stdout.
- `recommended_workflows` supplies executable argument templates for immediate
  redaction and for the `analyze → edit JSON → render` workflow.
- `commands.*.reads`, `commands.*.outputs`, and `commands.*.when_to_use` make
  each command's data flow explicit; `outputs` means file artifacts, while
  every command still returns its status on stdout. In particular, `render`
  reads both the original source video and the result JSON.

For a normal request to blur or anonymize faces, automation should select
`process`, run the supplied command template once with `--dry-run`, inspect
both `stdout.ok` and `stdout.ready`, and then repeat it without `--dry-run`.
Mosaic/pixelation is selected with `--render.redaction.method mosaic`. Existing
files are never replaced unless `--overwrite` is deliberately added. On
success, read final resolved file paths from `stdout.artifacts`; a dry-run puts
the planned paths in `stdout.plan.artifacts`.

`analyze`, `render`, `process`, `describe`, and
`doctor` write exactly one compact status JSON object to standard output. A
successful execution has this stable envelope:

```json
{"status_schema_version":1,"ok":true,"command":"analyze","artifacts":{"result_json":"/output/input_privateframe.json","result_video":null},"runtime":{"provider":"CPUExecutionProvider"},"timings":{"total_seconds":8.2},"summary":{"frame_count":300,"face_tracks":3,"face_regions":615}}
```

The success summary contains only stable user-facing counts. Detailed tracking,
recognition, sampling, cache, model, and backend diagnostics are not written to
standard output; development runs can retain them in the separate developer
report.

`timings.total_seconds` measures the reported analysis/render stages. It excludes
process startup, argument handling, and shutdown; measure the subprocess wall
time separately when comparing end-to-end processing speed.

Failures use the same standard-output channel and include a structured `error`
with `code`, `stage`, `type`, `message`, `retryable`, and `hints`. Standard error
is reserved for diagnostics and progress, so redirecting it never corrupts the
final status object. Ctrl-C returns a `cancelled` status and exit code 130. The
old `--json` switch is no longer needed; `analyze`
remains the subcommand that selects JSON-only analysis.

Discover the installed CLI instead of hard-coding its options:

```bash
insightface-privateframe --version
insightface-privateframe describe
insightface-privateframe doctor
```

`describe` returns commands and workflows, common/intermediate configuration,
artifact contracts, status-output rules, exit codes, and examples. Full
configuration is documented in the bundled
[configuration reference](insightface/app/privateframe/docs/configuration.md),
including advanced tracking and detector tuning. `doctor` reports readiness
checks for the runtime, models, media support, output location, and safety settings.

Self-description uses **`contract_schema_version: 2`**.
`config.dotted_options` contains the selected 30 fields, their defaults appear
in each field specification, and `config.groups` organizes them by purpose.
Full configuration is available at `config.full_reference.path`; read it when
an unlisted option is needed. The execution status uses
**`status_schema_version: 1`**. The default analysis FPS is **30**.
Video output defaults to **libx264, CRF 23, and the medium preset**, balancing
quality and file size for sharing. Use CRF 18 to retain more detail or CRF 28
for smaller files. Encoding quality does not change face detection or matching.

The reference is generated from the complete configuration catalog and checked
for drift in tests. After changing configuration defaults or documentation,
maintainers regenerate and verify it with:

```bash
python -m insightface.app.privateframe.config_reference
python -m insightface.app.privateframe.config_reference --check
```

Every execution command accepts `--progress auto|text|jsonl|none`. `auto` uses
human-readable progress on an interactive terminal and JSONL otherwise.
`jsonl` writes each application progress event or reference-photo diagnostic as
one compact JSONL record to standard error. Reference-photo diagnostics have
`log_schema_version: 1`, `event: "log"`,
`level`, `stage: "recognition"`, and `message`; callers should distinguish them
from progress records. Third-party runtime diagnostics may still be plain text
on standard error, so the entire stream is not guaranteed to be JSONL.
`none` suppresses progress but retains reference-photo
diagnostics as text on standard error. In every mode, standard output remains
the single final JSON object.

Use `--dry-run` to resolve and validate the configuration, input, work
directory, output artifacts, and dotted overrides without model inference or
artifact rendering. It may open an in-memory codec context (or encode one
synthetic frame to a null sink) to validate the final encoder options. It does
not download models, create ONNX Runtime sessions, compile CoreML models, warm
up inference, or create output files/directories:

```bash
insightface-privateframe process \
  --input /data/video.mp4 \
  --output-dir /data/output \
  --scan.max_analysis_fps 15 \
  --progress jsonl \
  --dry-run
```

PrivateFrame protects existing public artifacts by default. Review the dry-run
plan first, then pass `--overwrite` explicitly when replacing the reported
JSON or video is intentional. Concurrent CLI writers targeting the same output
or work directory are serialized with adjacent lock files; locks owned by dead
processes are reclaimed, while `output_busy` is safe to retry after the active
invocation finishes.

Development install:

```
cd python-package
pip install -e ".[gui]"
# For NVIDIA CUDA, replace onnxruntime after this install:
# python -m pip uninstall -y onnxruntime
# python -m pip install onnxruntime-gpu
insightface-gui
```

Equivalent launch commands:

```
insightface-eval-studio
insightface-desktop
python -m insightface.gui
```

The GUI is called **InsightFace Evaluation Studio**. It provides local 1:1 face
compare, People Library management, 1:N face search, multi-face photo
recognition, batch folder processing, album people clustering, enterprise
evaluation reports, and a face swap entry point. Workspace data such as
embeddings, databases, and reports is stored locally by default under
``~/.insightface/gui`` and is not uploaded automatically. PrivateFrame result
files instead default to the platform's user-visible Videos directory (Movies
on macOS).
Image and video previews are clickable upload targets: click
``Click to upload or drag a file here`` or drop a file onto the preview. The
preview changes color on hover and during drag-over. Loaded previews show a
small delete button and can be replaced by dragging in another file.

The desktop app uses mode-based navigation. **PrivateFrame** is the first
workflow, followed by **Face Recognition**, **Album Management**, **Face Swap**,
and **Enterprise Evaluation** in the persistent **Workflows** rail.
PrivateFrame accepts a local video and output directory, initially selects the
system Videos directory (Movies on macOS), runs the Python pipeline on a
background worker, and always creates a reusable `_privateframe.json`.
It uses the GUI's global model, model root, and provider rather than a separate
PrivateFrame model selector. Processing is enabled for `raccoon_s` and
`raccoon_l`; a missing package may download into the configured root on first
use, while an invalid installed V2 package is rejected before the job starts.
Its output mode can stop after analysis or immediately render the paired
`_privateframe.mp4`, without blocking the GUI. The main page exposes
analysis frequency, which people to blur,
the redaction style, and whether to preserve supported audio. Video preview
and metadata share the input card, followed by a full-width output directory.
Start/cancel actions and progress remain below the settings area. **Processing
details** opens the live run log and output filenames in a separate, non-modal
window, so an empty log does not take space from the settings. Output paths
are also available in the output directory's tooltip. The two photo
modes show a reference-photo folder directly below the policy: no person names
or per-person folders are needed, and only the largest face in each photo is
used. The folder's tooltip explains supported photos. Reference selection and
skipped-photo reasons appear in the run log. **More Options** shows the global
model/provider information and groups controls into redaction appearance (coverage and
between-scan tracking), video output (quality, encoding speed, and whether to
render a video), and person matching (sampling profile and base similarity
threshold). Model problems that prevent processing still appear on the main
page. Person matching is shown only for photo modes, with a default
threshold of 0.40. Unconfirmed faces follow the configured `unknown_action`;
its default `auto` follows the selected photo policy. There is no separate GUI
override for it. Initial processing values and each group's
reset values are read from the packaged `configs/base.yaml` when the page is
created, without loading models. GUI presets are shortcuts; a configured value
outside those shortcuts is shown as the configured value instead of being
replaced. Non-CRF rate
control is inherited unless the user explicitly chooses a CRF quality setting.
Global model/root/provider selections remain the GUI's explicit overrides.
The More Options button indicates modified settings. Face Recognition
is a single **Query & Gallery** workspace: upload
one query image and one gallery image for 1:1 compare, or upload multiple
gallery images / a folder for 1:N gallery search. Album Management uses a
single **Album** workspace for adding one or more folders, refreshing new
images, DBSCAN clustering with a default cosine similarity threshold of `0.48`,
and reviewing original photo thumbnails. Album directories and clustering results are saved
locally for the next launch. Enterprise Evaluation is a single workspace for
local 1:1 and 1:N identity-folder evaluation, Auto Split, metrics, and PDF
report export. Enterprise datasets must pass validation before evaluation; the
validator checks folder layout, gallery/probe rules, and the selected
multi-face handling policy.
Global utilities are available from the top bar and **Tools** menu:
**Settings**, **Models**, and **License**. **Settings** controls the UI theme
and language. Language defaults to the operating system when it is supported,
otherwise English. Supported GUI languages are English, Chinese, Japanese,
Korean, Spanish, French, German, Portuguese, and Russian. Available themes
include System, Precision Light, Studio Dark, Graphite Pro, Azure Lab, Emerald
Focus, and Crimson Audit. Workspace paths are chosen on first launch and are
not changed from the settings dialog.

Face-recognition and face-swap models are not downloaded automatically by the
general GUI model manager. Open **Models > Downloads**,
click **Refresh Download URLs** to read the dedicated
[`model-zoo`](https://github.com/deepinsight/insightface/releases/tag/model-zoo)
release asset URLs, then explicitly download the selected package. Downloaded
zip files are cached under ``~/.insightface/gui/cache/models`` and extracted
under ``<model_root>/models/<model_name>/`` (the default model root is
``~/.insightface``).
The Downloads tab also lists GFPGANv1.4 as a third-party face restoration
model. After it is downloaded, enable **GFPGAN post-processing** in
**Models > Runtime** to run 512x512 GFPGAN restoration after face swap.
Detection size defaults to **Auto**, which runs joint 128x128 and 640x640
detection. Face swap models are selected in **Models > Runtime** from already
downloaded swap models only; the Face Swap workspace loads the configured swap
model only when a swap is run.
New GUI configurations default to `raccoon_s`; existing JSON configurations
retain their saved model. The shared catalog also includes `raccoon_l`,
`buffalo_l`, `buffalo_m`, `buffalo_s`, `buffalo_sc`, and `antelopev2`.
PrivateFrame is the exception to manual downloads: when the global model is a
Raccoon package, it may download under the configured global model root on the
first run. It does not fall back to another root or model. Each running job
keeps its startup model/root/provider snapshot, and PrivateFrame does not share
its inference Sessions with the ordinary GUI engine. The GUI prevents model
downloads and PrivateFrame processing from running at the same time, including
when a download continues after the Models dialog has been closed.

### Optional face3d Build

The default package does not build the optional ``face3d`` Cython/C++ extension.
This keeps the default install lighter and avoids local compiler
requirements. Users who need the legacy mask renderer / face3d path can opt in:

```
pip install -e ".[face3d]" --no-build-isolation --config-settings editable_mode=compat
python setup.py build_ext --inplace --with-face3d
```

The same build can also be enabled with:

```
INSIGHTFACE_WITH_FACE3D=1 python setup.py build_ext --inplace
```

More details:

- ``docs/gui.md``
- ``docs/commercial_evaluation.md``
- ``docs/gui_packaging.md``

## Change Log

### [2.0] - 2026-08-31

#### Added

- Add manifest-backed ModelZoo packages, PrivateFrame video privacy analysis,
  reusable analysis JSON, and the integrated PrivateFrame GUI workflow.
- Add automatic CoreML/CUDA/CPU Provider selection and persistent,
  signature-scoped CoreML compilation caches for managed Sessions.

#### Changed

- Default sampled PrivateFrame modes to detector-anchor interpolation.
- Use reusable fixed-shape SCRFD Sessions per input resolution by default.
- Reorganize installation guidance around ONNX Runtime, CoreML, CUDA,
  PrivateFrame, and the GUI.

### [1.0.1] - 2026-05-23

#### Changed

- Install ``onnxruntime`` by default for CPU and supported macOS CoreML systems.
  NVIDIA CUDA users must replace it with ``onnxruntime-gpu`` after installing
  or upgrading InsightFace; the two runtime distributions must not coexist.
- Remove the PyPI package metadata license classifier field while keeping the README license guidance.
- Move direct `Pillow` and `scikit-learn` requirements to the GUI extra, and `matplotlib` to the optional `face3d` extra.
- Remove unused base dependencies on `easydict` and `prettytable`.

### [1.0] - 2026-05-23

#### Added

- Add **InsightFace Evaluation Studio**, a cross-platform PySide6 desktop GUI for local face recognition, album grouping, enterprise evaluation/report export, and face swap trials.
- Add GUI launch commands: ``insightface-gui``, ``insightface-eval-studio``, ``insightface-desktop``, and ``python -m insightface.gui``.

#### Changed

- ``FaceAnalysis.prepare()`` now defaults ``ctx_id`` to 0 and uses Auto detection size, running SCRFD at both 128x128 and 640x640 before unified NMS.
- Route detection models loaded by ``model_zoo.get_model()`` through ``SCRFD`` by default.
- The optional ``face3d`` Cython/C++ extension is no longer built by default; use ``--with-face3d`` or ``INSIGHTFACE_WITH_FACE3D=1`` to opt in.

### [0.7.1] - 2022-12-14
  
#### Changed
  
- Change model downloading provider to cloudfront.

### [0.7] - 2022-11-28
  
#### Added

- Add face swapping model and example.
 
#### Changed
  
- Set default ORT provider to CUDA and CPU.
 
### [0.6] - 2022-01-29
  
#### Added

- Add pose estimation in face-analysis app.
 
#### Changed
  
- Change model automated downloading url, to ucloud.
 

## Quick Example

```
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis()  # Auto provider: CoreML, then CUDA, then CPU
app.prepare()  # ctx_id=0; Auto detection size: 128x128 + 640x640
img = ins_get_image('t1')
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)
```

On CoreML, SCRFD uses a fixed 640x640 main Session by default and lazily
creates one reusable fixed Session for each additional detection resolution.
Compiled CoreML artifacts are isolated by model and input signature under
``~/.insightface/cache/coreml/v1``. A new signature is warmed up once after
compilation; later cache hits skip that warmup. These cache controls are internal
and do not require additional arguments to ``FaceAnalysis`` or
``model_zoo.get_model()``.

This quick example will detect faces from the ``t1.jpg`` image and draw detection results on it.



## Optional liveness addon

FaceAnalysis can run RGB liveness detection before recognition. Enable it
explicitly; installing an addon file alone does not change existing behavior:

```python
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    addons=["liveness"],
    liveness_mode="normal",
    liveness_threshold=0.8,
)
app.prepare(ctx_id=0)
faces = app.get(image_bgr)

for face in faces:
    result = face.liveness
    if result is None:
        print("Liveness was not run")
    elif result.status == "input_rejected":
        print(result.reason)
    elif result.is_live:
        print("Live:", result.live_score)
    else:
        print("Fake:", result.live_score)

    # None when recognition was not selected or was blocked by normal mode.
    embedding = face.embedding
```

The three keyword-only options are:

| Option | Behavior |
| --- | --- |
| `addons=["liveness"]` | Select the liveness addon independently of `allowed_modules`. Omit it or use `addons=[]` to disable liveness: no addon download, loading or inference, and existing recognition behavior is unchanged. |
| `liveness_mode` | `normal` (default): recognize only faces whose liveness result is `True`. `observe`: run liveness and continue recognition regardless of its classification or input rejection. This option only takes effect when the liveness addon is selected. |
| `liveness_threshold` | Live-score threshold in `[0, 1]`, default `0.8`; equality passes. |

The addon is downloaded from the
[InsightFace model addons Release](https://github.com/deepinsight/insightface-model-addons/releases/download/addons/liveness.onnx)
to **`<root>/addons/liveness.onnx`**, default `~/.insightface/addons/liveness.onnx`.
All addon files use this flat directory. Downloads are verified against the
SHA256 in the packaged addon catalog before installation; cached files are also
verified before loading. For offline use, place the published file at this path
before constructing FaceAnalysis. An existing file with an unexpected digest
raises an error and is not overwritten.

`get()` still returns a list of `Face` objects. Each evaluated face has
three core liveness fields, accessible as attributes or dictionary keys:

| Result | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| Live | `"ok"` | `True` | Model probability |
| Fake | `"ok"` | `False` | Model probability |
| Input rejected | `"input_rejected"` | `None` | `None` |

Only insufficient source-image area around the aligned face produces
`input_rejected`. It adds a human-readable `reason`; live and fake results omit
this field. FaceAnalysis always returns this English text:

> Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image.

Use `status` and `is_live` for program logic, not the wording of `reason`. For
older results without it, a client can use
`result.get("reason") or "Input rejected by liveness detection."`.

When the liveness addon is not selected, the `liveness` key is absent and `face.liveness` returns
`None`, following the existing `Face` attribute convention. No detected faces
still returns `[]`. Fake and rejected faces remain in the list, with their
bounding boxes and landmarks; normal mode skips only the recognition task.
Other selected tasks retain their existing behavior. Invalid landmarks raise
`ValueError`; an alignment failure raises `RuntimeError`. These errors, model-loading
errors, inference failures and invalid model outputs raise exceptions in both
`normal` and `observe`, ending the call;
they are never reported as fake or silently ignored. Recognition and other
models still require their own valid inputs in observe mode.

The adapter accepts the original BGR image and detector five-point landmarks.
It uses a dedicated fixed 80x80 alignment template, rejects aligned crops with
more than 30% missing source area, and fills accepted crop borders by replication.
The model receives RGB float32 NCHW pixels divided by 255 and directly outputs a
live probability. The model's alignment is separate from ArcFace alignment.
Scores can differ across execution providers; validate the operating threshold
with the provider used in deployment.

The model addon is distributed separately from the base models; refer to its
release repository for provenance and applicable notices. The initial threshold
and crop gate are integration defaults, not a production accuracy guarantee.

## Model Zoo

In the latest version of insightface library, we provide following model packs:

Name in **bold** is the default model pack. **Auto** means we can download the model pack through the python library directly.

Once you manually downloaded the zip model pack, unzip it under `~/.insightface/models/` first before you call the program.

| Name          | Detection Model | Recognition Model    | Alignment    | Attributes | Model-Size | Link                                                         | Auto |
| ------------- | --------------- | -------------------- | ------------ | ---------- | ---------- | ------------------------------------------------------------ | ------------- |
| antelopev2    | SCRFD-10GF      | ResNet100@Glint360K  | 2d106 & 3d68 | Gender&Age | 407MB      | [link](https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing) | N             |
| **buffalo_l** | SCRFD-10GF      | ResNet50@WebFace600K | 2d106 & 3d68 | Gender&Age | 326MB      | [link](https://drive.google.com/file/d/1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB/view?usp=sharing) | Y             |
| buffalo_m     | SCRFD-2.5GF     | ResNet50@WebFace600K | 2d106 & 3d68 | Gender&Age | 313MB      | [link](https://drive.google.com/file/d/1net68yNxF33NNV6WP7k56FS6V53tq-64/view?usp=sharing) | N             |
| buffalo_s     | SCRFD-500MF     | MBF@WebFace600K      | 2d106 & 3d68 | Gender&Age | 159MB      | [link](https://drive.google.com/file/d/1pKIusApEfoHKDjeBTXYB3yOQ0EtTonNE/view?usp=sharing) | N             |
| buffalo_sc    | SCRFD-500MF     | MBF@WebFace600K      | -            | -          | 16MB       | [link](https://drive.google.com/file/d/19I-MZdctYKmVf3nu5Da3HS6KH5LBfdzG/view?usp=sharing) | N             |



Recognition Accuracy:

| Name      | MR-ALL | African | Caucasian | South Asian | East Asian | LFW   | CFP-FP | AgeDB-30 | IJB-C(E4) |
| :-------- | ------ | ------- | --------- | ----------- | ---------- | ----- | ------ | -------- | --------- |
| buffalo_l | 91.25  | 90.29   | 94.70     | 93.16       | 74.96      | 99.83 | 99.33  | 98.23    | 97.25     |
| buffalo_s | 71.87  | 69.45   | 80.45     | 73.39       | 51.03      | 99.70 | 98.00  | 96.58    | 95.02     |

*buffalo_m has the same accuracy with buffalo_l.*

*buffalo_sc has the same accuracy with buffalo_s.*



**Note that these models are available for non-commercial research purposes only.**



For insightface>=0.3.3, models will be downloaded automatically once we init
``app = FaceAnalysis()`` instance. Automatic ModelZoo downloads use the
dedicated
[`model-zoo`](https://github.com/deepinsight/insightface/releases/tag/model-zoo)
release.

For insightface==0.3.2, you must first download the model package by command:

```
insightface-cli model.download buffalo_l
```

## Use Your Own Licensed Model

You can simply create a new model directory under ``~/.insightface/models/`` and replace the pretrained models we provide with your own models. And then call ``app = FaceAnalysis(name='your_model_zoo')`` to load these models.

## Call Models

The latest insightface libary only supports onnx models. Once you have trained detection or recognition models by PyTorch, MXNet or any other frameworks, you can convert it to the onnx format and then they can be called with insightface library.

### Call Detection Models

```
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# Method-1, use FaceAnalysis
app = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
app.prepare() # ctx_id=0; Auto detection size: 128x128 + 640x640

# Method-2, load model directly
detector = insightface.model_zoo.get_model('your_detection_model.onnx')
detector.prepare(ctx_id=0) # SCRFD defaults to Auto: 128x128 + 640x640

```

### Call Recognition Models

```
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

handler = insightface.model_zoo.get_model('your_recognition_model.onnx')
handler.prepare(ctx_id=0)

```
