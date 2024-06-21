import asyncio
import dataclasses
import sys
import threading
from asyncio.base_events import _run_until_complete_cb


@dataclasses.dataclass
class AsyncRunUntilCompleteContext:
    loop: asyncio.BaseEventLoop
    future: asyncio.Future
    new_task: bool


def setup_run_until_complete(
    loop: asyncio.BaseEventLoop,
    future: asyncio.Future,
) -> AsyncRunUntilCompleteContext:
    loop._check_closed()
    loop._check_running()

    new_task = not asyncio.futures.isfuture(future)
    future = asyncio.tasks.ensure_future(future, loop=loop)
    if new_task:
        # An exception is raised if the future didn't complete, so there
        # is no need to log the "destroy pending task" message
        future._log_destroy_pending = False

    future.add_done_callback(_run_until_complete_cb)

    # set up run forever
    loop._set_coroutine_origin_tracking(loop._debug)

    loop._old_agen_hooks = sys.get_asyncgen_hooks()
    loop._thread_id = threading.get_ident()
    sys.set_asyncgen_hooks(
        firstiter=loop._asyncgen_firstiter_hook,
        finalizer=loop._asyncgen_finalizer_hook,
    )
    asyncio.events._set_running_loop(loop)
    return AsyncRunUntilCompleteContext(loop=loop, future=future, new_task=new_task)


def teardown_run_util_complete(ctx: AsyncRunUntilCompleteContext):
    ctx.loop._stopping = False
    ctx.loop._thread_id = None
    asyncio.events._set_running_loop(None)
    ctx.loop._set_coroutine_origin_tracking(False)
    # Restore any pre-existing async generator hooks.
    if ctx.loop._old_agen_hooks is not None:
        sys.set_asyncgen_hooks(*ctx.loop._old_agen_hooks)
        ctx.loop._old_agen_hooks = None

    ctx.future.remove_done_callback(_run_until_complete_cb)

    if not ctx.future.done():
        raise RuntimeError("Event loop stopped before Future completed.")


def raise_asyncio_exception(
    ctx: AsyncRunUntilCompleteContext, raise_error: bool = True
):
    if ctx.new_task and ctx.future.done() and not ctx.future.cancelled():
        # The coroutine raised a BaseException. Consume the exception
        # to not log a warning, the caller doesn't have access to the
        # local task.
        ctx.future.exception()

    try:
        teardown_run_util_complete(ctx)
    except RuntimeError as e:
        if raise_error:
            raise e

    if raise_error:
        raise
