from dataclasses import dataclass

from lstm_analysis_constants import EntryType

@dataclass
class PredictionWindowSlopes:
    entry_slope_threshold       : float
    short_entry_slope_threshold : float
    exit_slope_threshold        : float
    short_exit_slope_threshold  : float
    long_minus_short_threshold  : float
    entry_type                  : EntryType
    