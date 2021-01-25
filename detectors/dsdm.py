import numpy as np
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from collections import deque


class DSDM(BaseDriftDetector):

    def __init__(self, min_num_instances=30, drift_detection_level=3.0, stabilization_window_size=5):
        super().__init__()
        self.sample_count = None
        self.miss_prob = None
        self.miss_sd = None
        self.miss_prob_sd_min = None
        self.miss_prob_min = None
        self.miss_sd_min = None
        self.min_instances = min_num_instances
        self.drift_detection_level = drift_detection_level
        self.waiting_for_stabilization = False
        self.in_stabilization = False
        self.stabilization_window_size = stabilization_window_size
        self.stabilization_window = deque()
        self.reset()

    def reset(self):

        super().reset()
        self.sample_count = 1
        self.miss_prob = 1.0
        self.miss_sd = 0.0
        self.miss_prob_sd_min = float("inf")
        self.miss_prob_min = float("inf")
        self.miss_sd_min = float("inf")

    def detected_stabilization(self):
        return self.in_stabilization

    def add_element(self, prediction):

        if self.in_concept_change:
            self.reset()

        if self.in_stabilization:
            self.in_stabilization = False

        self.miss_prob = self.miss_prob + (prediction - self.miss_prob) / float(self.sample_count)
        self.miss_sd = np.sqrt(self.miss_prob * (1 - self.miss_prob) / float(self.sample_count))
        self.sample_count += 1

        self.stabilization_window.append(self.miss_prob)
        if len(self.stabilization_window) > self.stabilization_window_size:
            self.stabilization_window.popleft()

        self.estimation = self.miss_prob
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0

        if self.sample_count < self.min_instances:
            return

        if self.miss_prob + self.miss_sd <= self.miss_prob_sd_min:
            self.miss_prob_min = self.miss_prob
            self.miss_sd_min = self.miss_sd
            self.miss_prob_sd_min = self.miss_prob + self.miss_sd

        diff_quotient = (self.stabilization_window[-1] - self.stabilization_window[0]) / \
                        (self.stabilization_window_size - 1)
        if len(self.stabilization_window) == self.stabilization_window_size and self.waiting_for_stabilization and \
                abs(diff_quotient) < 0.001:
            self.in_stabilization = True
            self.waiting_for_stabilization = False

        if self.miss_prob + self.miss_sd > self.miss_prob_min + self.drift_detection_level * self.miss_sd_min:
            self.in_concept_change = True
            self.waiting_for_stabilization = True
