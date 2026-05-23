from pathlib import Path
from datasets.base_adapter import BaseDatasetAdapter


class MTDSPAdapter(BaseDatasetAdapter):
    def __init__(self, data_root, config):
        super().__init__(data_root, config)
        self.load_metadata()

    def load_metadata(self):
        root = Path(self.data_root)
        self.radar_echo_files = list(root.rglob("*radar*"))
        self.visible_files = list(root.rglob("*visible*"))
        self.ir_files = list(root.rglob("*infrared*")) + list(root.rglob("*ir*"))
        raise RuntimeError(
            "MTDSP adapter is a template. Please map dataset-specific files to the common Detection schema. "
            "Implement radar detector: CFAR + clustering to convert radar echoes into detections."
        )

    def num_frames(self): return 0
    def get_timestamp(self, frame_idx): return 0.0
    def get_radar_detections(self, frame_idx): return []
    def get_visible_detections(self, frame_idx): return []
    def get_infrared_detections(self, frame_idx): return []
    def get_ground_truth(self, frame_idx): return []
    def has_ground_truth_ids(self): return False
    def coordinate_frame(self): return "UNKNOWN"
