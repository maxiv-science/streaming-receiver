import asyncio
import threading
from collections import deque
from typing import Optional, Any, Deque
from concurrent.futures import Future


class Queuey:
    """Hybrid queue allowing both sync and async interfaces"""

    def __init__(self):
        self._lock = threading.Lock()
        self._items: Deque[Any] = deque()
        self._getters: Deque[Future] = deque()

    def put(self, item: Any) -> Optional[Future]:
        with self._lock:
            if self._getters:
                self._getters.popleft().set_result(item)
            else:
                self._items.append(item)

    def _get(self):
        with self._lock:
            if self._items:
                return self._items.popleft(), None
            else:
                future = Future()
                self._getters.append(future)
                return None, future

    def get(self):
        item, future = self._get()
        if future:
            item = future.result()
        return item

    async def get_async(self):
        item, future = self._get()
        if future:
            item = await asyncio.wait_for(asyncio.wrap_future(future), None)
        return item
