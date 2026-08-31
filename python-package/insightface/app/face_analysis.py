# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 


from __future__ import division

import glob
import os.path as osp
from pathlib import Path

import numpy as np
import onnxruntime

from ..model_zoo import model_zoo
from ..model_zoo.onnxruntime_utils import get_default_providers
from ..model_zoo.package_manifest import (
    MODEL_PACKAGE_TASKS,
    ModelPackageDescriptor,
    has_model_package_manifest,
    load_model_package,
)
from ..utils import DEFAULT_MP_NAME, ensure_available
from .common import Face

__all__ = ['FaceAnalysis']

DEFAULT_DET_SIZES = [(128, 128), (640, 640)]


def _provider_name(provider):
    if isinstance(provider, (list, tuple)) and provider:
        return str(provider[0])
    return str(provider)


def _default_coreml_detector_input_size():
    return max(
        DEFAULT_DET_SIZES,
        key=lambda value: int(value[0]) * int(value[1]),
    )

def _is_auto_det_size(det_size):
    if det_size is None:
        return True
    if isinstance(det_size, np.ndarray):
        det_size = det_size.tolist()
    if isinstance(det_size, (list, tuple)) and len(det_size) == 2:
        first, second = det_size
        if not isinstance(first, (list, tuple, np.ndarray)) and not isinstance(second, (list, tuple, np.ndarray)):
            try:
                return int(first) == 0 and int(second) == 0
            except (TypeError, ValueError):
                return False
    return False


class FaceAnalysis:
    def __init__(self, name=DEFAULT_MP_NAME, root='~/.insightface', allowed_modules=None, **kwargs):
        onnxruntime.set_default_logger_severity(3)
        if kwargs.get('providers') is None:
            kwargs['providers'] = get_default_providers()
        providers = kwargs.get('providers') or ()
        if (
            providers
            and _provider_name(providers[0]) == 'CoreMLExecutionProvider'
        ):
            # CoreML cannot safely construct the dynamic SCRFD Session with
            # the default ALL compute-unit policy on some ORT/macOS versions.
            # Keep the public constructor unchanged and pass an internal fixed
            # main resolution to both manifest and legacy model loading.
            if kwargs.get('static_shape_sessions', True) is not False:
                kwargs.setdefault(
                    '_coreml_detector_input_size',
                    _default_coreml_detector_input_size(),
                )
        self.models = {}
        self.det_model = None
        self.model_package = None

        if isinstance(name, ModelPackageDescriptor):
            self.model_package = name
            self.model_dir = str(name.path)
        else:
            direct = Path(str(name)).expanduser()
            if direct.is_dir():
                self.model_dir = str(direct.resolve())
            else:
                self.model_dir = ensure_available('models', name, root=root)

        if (
            self.model_package is not None
            or has_model_package_manifest(self.model_dir)
        ):
            self._load_manifest_package(
                allowed_modules,
                kwargs,
            )
            assert 'detection' in self.models
            self.det_model = self.models['detection']
            return

        onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']

    def _load_manifest_package(
        self,
        allowed_modules,
        kwargs,
    ):
        # Parse once before any Session construction. Invalid manifests must
        # never fall back to filename/shape heuristics.
        if self.model_package is None:
            self.model_package = load_model_package(self.model_dir)
        requested = None if allowed_modules is None else set(allowed_modules)

        model_kwargs = dict(kwargs)

        for task in MODEL_PACKAGE_TASKS:
            if requested is not None and task not in requested:
                continue
            descriptor = self.model_package.task(task)
            model = model_zoo.get_model(
                self.model_dir,
                model_task=task,
                model_descriptor=descriptor,
                **model_kwargs,
            )
            if model.taskname != task:
                raise RuntimeError(
                    f'manifest {task} model reported task {model.taskname}, '
                    f'expected {task}'
                )
            if task in self.models:
                raise RuntimeError(f'duplicated manifest model task: {task}')
            print(
                'find manifest model:',
                descriptor.path,
                model.taskname,
                getattr(model, 'input_shape', None),
                getattr(model, 'input_mean', None),
                getattr(model, 'input_std', None),
            )
            self.models[task] = model

    def prepare(self, ctx_id=0, det_thresh=0.5, det_size=None):
        self.det_thresh = det_thresh
        if _is_auto_det_size(det_size):
            det_size = list(DEFAULT_DET_SIZES)
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0, det_metric='default'):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric=det_metric)
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret

    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(int)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

            #for key, value in face.items():
            #    if key.startswith('landmark_3d'):
            #        print(key, value.shape)
            #        print(value[0:10,:])
            #        lmk = np.round(value).astype(int)
            #        for l in range(lmk.shape[0]):
            #            color = (255, 0, 0)
            #            cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
            #                       2)
        return dimg
