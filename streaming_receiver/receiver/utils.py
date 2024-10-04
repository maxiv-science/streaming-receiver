import asyncio
import logging
from asyncio import Task, Future
from typing import Any


async def cancel_and_wait(task: Task[Any] | Future[Any]) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except StopIteration:
        pass
    except Exception as e:
        logging.error("cancel and wait task raised %s", e.__repr__())
