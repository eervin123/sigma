from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict

from objects.action_type import ActionType
from objects.closing_trigger import ClosingTrigger


@dataclass
class Trade:
    entry_price: float
    direction: ActionType
    entry_time: datetime
    position_id: str
    index: Optional[int] = None
    quantity: Optional[float] = 1
    open_trigger: Optional[str] = None
    close_trigger: Optional[ClosingTrigger] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    profit: Optional[float] = None
    rec_id: Optional[str] = None,
    leverage_frac: Optional[float] = None

    @staticmethod
    def from_dict(row_dict: Dict):
        exit_time = row_dict.get('exit_time')
        if exit_time:
            exit_time = datetime.fromisoformat(exit_time)
        return Trade(
            entry_price=row_dict.get('entry_price'),
            direction=row_dict.get('direction'),
            entry_time=datetime.fromisoformat(row_dict.get('entry_time')),
            position_id=row_dict.get('position_id'),
            index=row_dict.get('index'),
            quantity=row_dict.get('quantity'),
            open_trigger=row_dict.get('open_trigger'),
            close_trigger=row_dict.get('close_trigger'),
            exit_price=row_dict.get('exit_price'),
            exit_time=exit_time,
            profit=row_dict.get('profit'),
            rec_id=row_dict.get('rec_id'),
            leverage_frac=row_dict.get('leverage_frac'))
