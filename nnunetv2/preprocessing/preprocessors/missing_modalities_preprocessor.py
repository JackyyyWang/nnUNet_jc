import os
from typing import List, Union
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json

from .default_preprocessor import DefaultPreprocessor


class MissingModalitiesPreprocessor(DefaultPreprocessor):
    """Preprocessor that fills missing image modalities with zeros."""

    def run_case(
        self,
        image_files: List[str],
        seg_file: Union[str, None],
        plans_manager,
        configuration_manager,
        dataset_json: Union[dict, str],
    ):
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        num_modalities = len(dataset_json.get("channel_names", {}))

        # map modality index to file
        file_map = {}
        for f in image_files:
            base = os.path.basename(f)
            try:
                idx = int(base.split("_")[-1].split(".")[0])
            except ValueError:
                continue
            file_map[idx] = f

        assert 0 in file_map, "T2 modality (_0000) must be present for every case"

        rw = plans_manager.image_reader_writer_class()

        data_list = []
        data0, properties = rw.read_images((file_map[0],))
        data_list.append(data0.astype(np.float32, copy=False))
        shape = data0.shape[1:]

        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None

        for m in range(1, num_modalities):
            if m in file_map:
                img, _ = rw.read_images((file_map[m],))
                img = img.astype(np.float32, copy=False)
            else:
                img = np.zeros((1, *shape), dtype=np.float32)
            data_list.append(img)

        data = np.vstack(data_list)
        data, seg, properties = self.run_case_npy(
            data, seg, properties, plans_manager, configuration_manager, dataset_json
        )
        return data, seg, properties

