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
paths, content hashes, model fingerprints, Git state, or internal tracking
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
  max_analysis_fps: 15
```

An explicit `base_config` field remains available when the custom YAML must
inherit a different complete parent configuration. CLI dotted options are
applied last, after the Base and custom YAML layers.

The GUI offers **Normal (up to 30 FPS)** by default and **Fast (up to 15 FPS)**
for constrained devices. The equivalent CLI override is
`--scan.max_analysis_fps 15`. PrivateFrame derives a uniform integer sampling
stride for each video; sampled analysis trades some assurance for speed.

### CLI automation contract

The CLI is safe to invoke from shell scripts and vendor-neutral AI coding tools
without parsing human-oriented log text. An unfamiliar automation client should
run `insightface-privateframe describe` and read these high-level fields first:

- `tool.summary`, `tool.purpose_id`, and `tool.capabilities` explain that the
  tool detects and tracks face regions and renders Gaussian blur or mosaic.
- `discovery` maps common user intentions to the correct command and gives a
  safe dry-run/execution policy.
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

`describe` returns the public commands, configuration schema and defaults,
artifact contract, status-output rules, exit codes, and examples. It omits
internal debug controls. `doctor` reports readiness checks for the runtime,
models, media support, output location, and safety settings.

Every execution command accepts `--progress auto|text|jsonl|none`. `auto` uses
human-readable progress only on an interactive terminal; `jsonl` writes one
compact progress event per line to standard error; `none` is useful for quiet
automation. In every mode, standard output remains the single final JSON
object.

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
`_privateframe.mp4`, without blocking the GUI. The main page displays the
global model/root status and exposes performance, privacy policy, redaction,
and output mode; **More Options** adds
advanced processing, face coverage, Medium-default encoding, quality,
audio, and selective-recognition Gallery controls. Face Recognition
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
compilation; later cache hits skip that warmup. The public ``FaceAnalysis`` and
``model_zoo.get_model()`` call signatures are unchanged.

This quick example will detect faces from the ``t1.jpg`` image and draw detection results on it.



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
