from dataclasses import dataclass

@dataclass
class PredictionWindowSlopes:
    entry_slope_threshold       : float
    short_entry_slope_threshold : float
    