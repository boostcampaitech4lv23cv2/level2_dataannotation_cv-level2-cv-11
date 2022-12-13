import math
from collections import defaultdict
from copy import copy, deepcopy
from datetime import datetime
import itertools
import logging
import multiprocessing
import multiprocessing.dummy
import os
import warnings
import webbrowser

from bson import ObjectId
import jinja2
import numpy as np
import requests
import urllib3

import eta.core.data as etad
import eta.core.image as etai
import eta.core.serial as etas
import eta.core.utils as etau

import fiftyone.constants as foc
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.media as fom
import fiftyone.core.metadata as fomt
from fiftyone.core.sample import Sample
import fiftyone.core.utils as fou
import fiftyone.utils.annotations as foua
import fiftyone.utils.data as foud
import fiftyone.utils.video as fouv
from fiftyone.types.dataset_types import ImageLabelsDataset


logger = logging.getLogger(__name__)


class UFODatasetImporter(
    foud.LabeledImageDatasetImporter, foud.ImportPathsMixin
):
    """
    fiftyone/utils.coco.py  - COCODetectionDatasetImporter 를 참고
    UFO 형식 데이터셋을 불러올 수 있도록 수정함
    """
    def __init__(
        self,
        dataset_dir=None,
        data_path=None,
        labels_path=None,
        include_all_data=False,
        shuffle=False,
        seed=None,
        max_samples=None,
    ):
        if dataset_dir is None and data_path is None and labels_path is None:
            raise ValueError(
                "At least one of `dataset_dir`, `data_path`, and "
                "`labels_path` must be provided"
            )

        data_path = self._parse_data_path(
            dataset_dir=dataset_dir,
            data_path=data_path,
            default="data/",
        )

        labels_path = self._parse_labels_path(
            dataset_dir=dataset_dir,
            labels_path=labels_path,
            default="labels.json",
        )

        super().__init__(
            dataset_dir=dataset_dir,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples,
        )

        self.data_path = data_path
        self.labels_path = labels_path
        self.include_all_data = include_all_data

        self._info = None
        self._image_paths_map = None
        self._cvat_images_map = None
        self._filenames = None
        self._iter_filenames = None
        self._num_samples = None

    def __iter__(self):
        self._iter_filenames = iter(self._filenames)
        return self

    def __len__(self):
        return self._num_samples

    def __next__(self):
        filename = next(self._iter_filenames)

        if os.path.isabs(filename):
            image_path = filename
        else:
            image_path = self._image_paths_map[filename]

        image = self._ufo_images_map.get(filename, None)
        if image is not None:
            # Labeled image
            w = image['img_w']
            h = image['img_h']
            polys = []
            for i, word in image['words'].items():
                default_word = dict(transcription='', language=None, illegibility=False, orientation='Horizontal')
                default_word.update(word)
                word = default_word
                # 좌표 0~1 범위로 정규화
                points = []
                for p in word['points']:
                    x, y = p
                    x /= w
                    y /= h
                    points.append((x, y))

                word_tags = None
                if 'word_tags' in word:
                    word_tags = word['word_tags']
                elif 'tags' in word:
                    word_tags = word['tags']

                poly = fol.Polyline(
                    label = 'word',
                    ufo_id = f'{filename}-{i}',
                    points = [points], filled = True,
                    transcription = word['transcription'].replace('-', '~'),
                    language = str(word['language']),
                    illegibility = word['illegibility'],
                    orientation = word['orientation'],
                    word_tags = str(word_tags)
                )

                polys.append(poly)

            image_metadata = fomt.ImageMetadata(width=w, height=h)
            labels = fol.Polylines(
                label = 'words',
                polylines = polys,
                tags = image['tags'],
                license_tag = image['license_tag']
            )
        else:
            # Unlabeled image
            image_metadata = None
            labels = None

        return image_path, image_metadata, labels

    @property
    def has_dataset_info(self):
        return True

    @property
    def has_image_metadata(self):
        return True

    @property
    def label_cls(self):
        return fol.Polylines

    def setup(self):
        image_paths_map = self._load_data_map(self.data_path, recursive=True)

        if self.labels_path is not None and os.path.isfile(self.labels_path):
            d = etas.load_json(self.labels_path)['images']
        else:
            d = {}

        # Use subset/name as the key if it exists, else just name
        ufo_images_map = {}
        for k, v in d.items():
            ufo_images_map[fou.normpath(k)] = v

        filenames = set(ufo_images_map.keys())

        if self.include_all_data:
            filenames.update(image_paths_map.keys())

        filenames = self._preprocess_list(sorted(filenames))

        self._image_paths_map = image_paths_map
        self._ufo_images_map = ufo_images_map
        self._filenames = filenames
        self._num_samples = len(filenames)

    def get_dataset_info(self):
        return self._info


class UFODatasetExporter(
    foud.LabeledImageDatasetExporter, foud.ExportPathsMixin
):
    """
    fiftyone/utils.coco.py  - COCODetectionDatasetExporter 를 참고
    UFO 형식 어노테이션을 내보낼 수 있도록 수정함
    """
    def __init__(
        self,
        export_dir=None,
        data_path=None,
        labels_path=None,
        rel_dir=None,
        # abs_paths=False,
        info=None,
        # extra_attrs=True,
        # annotation_id=None,
        # num_decimals=None,
        # tolerance=None,
    ):
        data_path, _ = self._parse_data_path(
            export_dir=export_dir,
            data_path=data_path,
            default="data/",
        )

        labels_path = self._parse_labels_path(
            export_dir=export_dir,
            labels_path=labels_path,
            default="labels.json",
        )

        super().__init__(export_dir=export_dir)

        self.data_path = data_path
        self.labels_path = labels_path
        self.rel_dir = rel_dir
        # self.abs_paths = abs_paths
        self.info = info
        # self.extra_attrs = extra_attrs
        # self.num_decimals = num_decimals
        # self.tolerance = tolerance

        self._images = None

    @property
    def requires_image_metadata(self):
        return True

    @property
    def label_cls(self):
        return fol.Polylines

    def setup(self):
        self._images = dict()

    def export_sample(self, image_or_path, label, metadata=None):
        if metadata is None:
            metadata = fom.ImageMetadata.build_for(image_or_path)

        file_name = os.path.basename(image_or_path)

        words = []
        if label is not None:
            if isinstance(label, fol.Polylines):
                labels = label.polylines
            else:
                raise ValueError(
                    "Unsupported label type %s. The supported types are %s"
                    % (type(label), self.label_cls)
                )

            for poly in labels:
                points = []
                for p in poly['points'][0]:
                    x, y = p
                    x *= metadata.width
                    y *= metadata.height
                    points.append([x, y])

                word = {
                    "points": points,
                    "transcription": poly['transcription'] if 'transcription' in poly else '',
                    "language": eval(poly['language']) if 'language' in poly else None,
                    "illegibility": poly["illegibility"],
                    "orientation": poly["orientation"],
                    "tags": eval(poly["word_tags"]) if 'word_tags' in poly else None
                }

                words.append(word)

        words_dict = dict()
        for i, j in enumerate(words):
            words_dict[str(i)] = j

        image = {
            "img_h": metadata.height,
            "img_w": metadata.width,
            "words": words_dict,
            "tags": label['tags'] if label and 'tags' in label else None,
            "license_tag": label['license_tag'] if label and 'license_tag' in label else None
        }

        self._images[file_name] = image

    def close(self, *args):
        labels = {
            "images": self._images
        }

        etas.write_json(labels, self.labels_path, pretty_print=True)


class UFODataset(ImageLabelsDataset):
    def get_dataset_importer_cls(self):
        return UFODatasetImporter

    def get_dataset_exporter_cls(self):
        return UFODatasetExporter


base = ['ko', 'en', 'others']
lang_values = ['None'] # 가능한 모든 language 조합들
for i in range(1, 4):
    for L in itertools.combinations(base, i):
        lang_values.append(str(list(L)))

label_schema = {
    "ground_truth": {
        "type": "polygons",
        "classes": [
            "word"
        ],
        "attributes": {
            "language": {
                "type": "select",
                "values": lang_values,
                "default": "None"
            },
            "orientation": {
                "type": "select",
                "values": ["Horizontal", "Vertical", "Irregular"],
                "default": "Horizontal"
            },
            "illegibility": {
                "type": "checkbox",
                "default": False
            },
            "transcription": {
                "type": "text",
                "default": ""
            },
            "word_tags": {
                "type": "text",
                "default": "[]"
            }
        },
        "existing_field": True,
        "allow_additions": True,
        "allow_deletions": True,
        "allow_label_edits": True,
        "allow_spatial_edits": True
    }
}