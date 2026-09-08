# InsightFace Python Library 2.0

## License

The code of InsightFace Python Library is released under the MIT License. There is no limitation for both academic and commercial usage.

**The pretrained models we provided with this library are available for non-commercial research purposes only, including both auto-downloading models and manual-downloading models.**

InsightFace 2.0 uses ONNX Runtime for FaceAnalysis, ModelZoo, PrivateFrame, and
the desktop Evaluation Studio.

## What's new in 2.0

- **[Liveness update](#optional-liveness-addon):** optional RGB liveness before
  recognition, configurable recognition gating, and per-face scores with
  input-rejection guidance.
- **[PrivateFrame update](#privateframe):** local video face blur/mosaic,
  reference-photo selection, editable analysis JSON, and desktop, CLI, and
  Python API workflows. See the [full guide](insightface/app/privateframe/README.md).
- **Runtime and models:** `raccoon_s` / `raccoon_l` model packages, automatic
  CoreML/CUDA/CPU selection, and reusable CoreML compilation caches.

## Installation

### Choose an installation

| Use case | Command |
|---|---|
| FaceAnalysis and ModelZoo | `pip install insightface` |
| Video face blur/mosaic with the PrivateFrame API and CLI | `pip install "insightface[privateframe]"` |
| Evaluation Studio GUI, including PrivateFrame | `pip install "insightface[gui]"` |

To use the changes in this development branch, install from the repository
root instead:

```bash
python -m pip install -e "./python-package[privateframe]"
# Or install the desktop application, which includes PrivateFrame:
python -m pip install -e "./python-package[gui]"
```

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

PrivateFrame detects and tracks faces in local videos and applies Gaussian
blur or mosaic. Use the desktop Evaluation Studio, CLI, or Python API to blur
all detected faces or select people with reference photos. Processing runs
locally and keeps the source video unchanged.

See the [full guide](insightface/app/privateframe/README.md) for installation,
GUI/Python examples, configuration, and JSON editing, or watch the
[video demo](https://example.com/privateframe-demo).

### Quick start

```bash
insightface-privateframe process \
  --input /data/video.mp4 --output-dir /data/output
```

`process` writes `video_privateframe.json` and `video_privateframe.mp4`.
Use `analyze` for JSON only, or `render` to render existing or edited JSON with
the original video without rerunning inference. Replacing existing outputs
requires `--overwrite`. See the [CLI workflows](insightface/app/privateframe/README.md#command-line-quick-start).

### Choosing who to blur with reference photos

| Mode | Behavior with the default `unknown_action: auto` |
|---|---|
| `all` (default) | Blur every detected face. |
| `blur_only` | Blur people matched to reference photos; keep others visible. |
| `exempt` | Keep matched people visible; blur everyone else. |

Photo modes use `recognition.reference_dir`, a folder of photos with no
per-person subfolders; only the largest face in each photo is used. A missed
match in `blur_only` can leave a target person visible, so review the output.
See [reference-photo setup and examples](insightface/app/privateframe/README.md#choosing-who-to-blur-with-reference-photos).

### Analysis sampling rate

`scan.max_analysis_fps` defaults to **30**, an approximate ceiling on regular
detection scans per second of source video. Lower it to 15 for less detector
work; wider gaps can miss
briefly visible faces. See [sampling and quality tradeoffs](insightface/app/privateframe/README.md#analysis-sampling-rate)
and the [configuration guide](insightface/app/privateframe/README.md#configuration-and-rendering).

### CLI automation contract

Use `insightface-privateframe describe` to discover commands and options, and
`process --dry-run` with your input/output arguments to check readiness.
Commands return a final JSON status on stdout and progress on stderr. See the
[automation guide](insightface/app/privateframe/README.md#cli-discovery-dry-runs-and-automation)
for readiness checks, structured progress, and error handling.

## Evaluation Studio GUI

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
PrivateFrame uses the global model, model root, and provider under **Models**,
and supports `raccoon_s` and `raccoon_l`. Processing runs in the background,
producing analysis JSON and an optional redacted MP4. See the
[PrivateFrame GUI guide](insightface/app/privateframe/README.md#desktop-gui).

Face Recognition
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

### [2.0] - Unreleased

#### Liveness update

- Add opt-in RGB liveness through `FaceAnalysis(addons=["liveness"])`, with
  `normal` / `observe` modes and a configurable live-score threshold (default
  `0.8`).
- Return per-face scores and input-rejection guidance. Normal mode retains
  detections while skipping recognition for faces that do not pass.
- Download and verify the separate addon model under
  `<root>/addons/liveness.onnx`. See [usage and result fields](#optional-liveness-addon).

#### PrivateFrame update

- Add local video face detection/tracking with Gaussian blur or mosaic through
  the Python API, `insightface-privateframe` CLI, and Evaluation Studio GUI.
- Add `all`, `blur_only`, and `exempt` policies, with a folder of reference
  photos for selecting people.
- Export editable analysis JSON and render it again without another inference
  pass; provide `analyze`, `process`, and `render` workflows.
- Add configurable analysis sampling (30 FPS by default), interpolation between
  scans, encoding/audio controls, and structured CLI status and progress.
- Add the [PrivateFrame feature and usage guide](insightface/app/privateframe/README.md).

#### Runtime and model updates

- Add manifest-backed `raccoon_s` / `raccoon_l` model packages and make
  `raccoon_s` the default for new GUI configurations.
- Add automatic CoreML/CUDA/CPU provider selection and persistent CoreML
  compilation caches; use reusable fixed-shape SCRFD sessions per resolution.
- Set `ORT_DISABLE_TELEMETRY=1` before importing ONNX Runtime, where supported.
- Add the `privateframe` installation extra and include it in the `gui` extra.

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
import cv2
from insightface.app import FaceAnalysis

image_bgr = cv2.imread("input.jpg")
if image_bgr is None:
    raise FileNotFoundError("Could not read input.jpg")

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

### Raccoon model packages

InsightFace 2.0 supports `raccoon_s` and `raccoon_l` through task-aware V2
manifests that declare the model files and preprocessing. Use them with
`FaceAnalysis(name="raccoon_s")` or `FaceAnalysis(name="raccoon_l")`.
PrivateFrame requires one of these packages; its default and the default for
new GUI configurations is `raccoon_s`. Ordinary `FaceAnalysis()` continues to
use `buffalo_l` by default.

Model packages are stored under `<root>/models/<name>/` (default root:
`~/.insightface`). PrivateFrame can download a missing selected package on
first use; see its [model setup guide](insightface/app/privateframe/README.md).
Liveness is a separate optional addon and must be enabled explicitly.

### Legacy model packs

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
