class SimulatedLaserScan:
    def __init__(self, angle_min, angle_max, angle_increment, range_min, range_max):
        self.header = {
            "stamp": 0,
            "frame_id": "laser_frame"
        }
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_increment = angle_increment
        self.time_increment = 0.0
        self.scan_time = 0.1
        self.range_min = range_min
        self.range_max = range_max
        self.ranges = []
        self.intensities = []
