from enum import Enum


class EngineMode(str, Enum):
    LOCAL_TEST_TRADE = "local_test_trade"
    LIVE_TRADE = "live_trade"
