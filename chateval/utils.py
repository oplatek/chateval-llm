import numpy as np
import torch
import random
import uuid
from typing import Optional, Any

def set_seed(seed):
    global _chateval_uuid
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    rd = random.Random()
    rd.seed(seed)
    _chateval_uuid = lambda: uuid.UUID(int=rd.getrandbits(128))


def uuid4():
    """
    Generates uuid4's exactly like Python's uuid.uuid4() function.
    When ``fix_random_seed()`` is called, it will instead generate deterministic IDs.
    """
    if _chateval_uuid is not None:
        return _chateval_uuid()
    return uuid.uuid4()


def ifnone(item: Optional[Any], alt_item: Any) -> Any:
    """Return ``alt_item`` if ``item is None``, otherwise ``item``."""
    return alt_item if item is None else item
