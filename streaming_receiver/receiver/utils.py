import asyncio
import logging
import traceback
from asyncio import Task, Future
from typing import Any


def done_callback(futr: Future[None]) -> None:
    try:
        futr.result()
    except asyncio.exceptions.CancelledError:
        pass
    except Exception as e:
        logging.error(
            "subroutine crashed %s trace: %s",
            e.__repr__(),
            traceback.format_exc(),
        )


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
