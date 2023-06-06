from enum import Enum


class ActionType(str, Enum):
    OPEN_SHORT = "open-short"
    OPEN_LONG = "open-long"
    CLOSE_LONG = "close-long"
    CLOSE_SHORT = "close-short"
    LEGACY_OPEN_LONG = "open_long"
    LEGACY_OPEN_SHORT = "open_short"
    LEGACY_CLOSE_LONG = "close_long"
    LEGACY_CLOSE_SHORT = "close_short"
    NOOP = "no_op"


def get_close_action(direction: ActionType):
    if direction == ActionType.OPEN_LONG:
        return ActionType.CLOSE_LONG
    elif direction == ActionType.OPEN_SHORT:
        return ActionType.CLOSE_SHORT
