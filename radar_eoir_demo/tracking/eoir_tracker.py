from tracking.radar_tracker import SimpleTracker


class EOIRTracker(SimpleTracker):
    def __init__(self, cfg, meas_noise):
        super().__init__(cfg, meas_noise, sensor_type="eoir")
