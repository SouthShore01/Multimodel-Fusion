from datasets.base_adapter import BaseDatasetAdapter


class MITMarinePerceptionAdapter(BaseDatasetAdapter):
    def __init__(self, data_root, config):
        super().__init__(data_root, config)
        self.load_metadata()

    def load_metadata(self):
        try:
            import rosbag  # noqa: F401
        except Exception:
            try:
                import rosbags  # noqa: F401
            except Exception:
                raise RuntimeError(
                    "MIT Marine Perception adapter requires rosbag/rosbags. "
                    "Please install the dependency or export the dataset into JSON/CSV first."
                )
        raise RuntimeError("MIT Marine Perception adapter scaffold loaded. TODO: map ROS topics to common Detection schema.")

    def num_frames(self): return 0
    def get_timestamp(self, frame_idx): return 0.0
    def get_radar_detections(self, frame_idx): return []
    def get_visible_detections(self, frame_idx): return []
    def get_infrared_detections(self, frame_idx): return []
    def get_ground_truth(self, frame_idx): return []
    def has_ground_truth_ids(self): return False
    def coordinate_frame(self): return "ROS_WORLD"
