"""Bound Kit shutdown while preserving the task's exit status."""
import os
import sys
import threading
import traceback


def run_and_close(main, app):
    status = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        status = 1
    def close():
        try:
            app.close(wait_for_replicator=False)
        except BaseException:
            traceback.print_exc()
    thread = threading.Thread(target=close, daemon=True)
    thread.start()
    thread.join(10)
    if thread.is_alive():
        print("Isaac shutdown exceeded 10 seconds; exiting completed worker.", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
