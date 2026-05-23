from abc import ABC, abstractmethod


class BaseDatasetAdapter(ABC):
    def __init__(self, data_root, config):
        self.data_root = data_root
        self.config = config

    @abstractmethod
    def load_metadata(self):
        pass

    @abstractmethod
    def num_frames(self):
        pass

    @abstractmethod
    def get_timestamp(self, frame_idx):
        pass

    @abstractmethod
    def get_radar_detections(self, frame_idx):
        pass

    @abstractmethod
    def get_visible_detections(self, frame_idx):
        pass

    @abstractmethod
    def get_infrared_detections(self, frame_idx):
        pass

    @abstractmethod
    def get_ground_truth(self, frame_idx):
        pass

    @abstractmethod
    def has_ground_truth_ids(self):
        pass

    @abstractmethod
    def coordinate_frame(self):
        pass
