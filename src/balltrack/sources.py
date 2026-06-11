"""
sources.py — how frames get read from the video: all of them, or live.

Two readers, one per prediction mode:

`frames(video)` is the offline reader: a plain iterator that yields every
frame in order. The model sets the pace, however long a frame takes, the
next read just waits.

`LatestFrameReader` is the live reader. A file has no clock,
so a background thread decodes it at the video's own FPS, like a camera
filming the same footage, the model can no longer pause the world. The thread 
keeps only the newest frame; `read()` returns that. If the model falls behind,
the frames it missed are gone, which is the point: the prediction always refers 
to now, not to a growing backlog. Frame numbers keep counting through the
drops, so the caller can tell the tracker how many frames really passed (`n_steps`).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def open_capture(video: str | Path) -> "cv2.VideoCapture":
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    return cap


def frames(video: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_no, frame) for every frame, in order (offline use)."""
    cap = open_capture(video)
    try:
        frame_no = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                return
            frame_no += 1
            yield frame_no, frame
    finally:
        cap.release()


class LatestFrameReader:
    """Replays a video live: a thread decodes at native FPS, keeping only the
    newest frame. `read()` blocks until an unseen frame exists, then returns
    (frame_no, frame); whatever the caller didn't pick up in time is gone."""

    def __init__(self, video: str | Path):
        self._cap = open_capture(video)
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._interval = 1.0 / fps if fps > 0 else 0.0  # 0 = unknown FPS, no pacing
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._frame_no = -1      # newest captured frame
        self._last_served = -1   # newest frame handed to the consumer
        self._stopped = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        # cap.read() decodes far faster than real time, so without this timer
        # the video would fast-forward. Sleeping until each frame's due time
        # holds the pace at native FPS and then the file behaves like a live camera.
        next_due = time.perf_counter()
        while True:
            if self._interval:
                now = time.perf_counter()
                if now < next_due:
                    time.sleep(next_due - now)
                next_due += self._interval
            ret, frame = self._cap.read()
            with self._cond:
                if self._stopped:
                    return
                if not ret:  # end of video
                    self._stopped = True
                else:
                    self._frame_no += 1
                    self._frame = frame
                self._cond.notify_all()
                if self._stopped:
                    return

    def read(self) -> tuple[int, np.ndarray] | None:
        """Block for the next unseen frame; None once the source has ended."""
        # _frame_no == _last_served means the shelf still holds the frame we
        # already took: sleep until the capture thread puts a newer one there.
        # While we run the model, that thread keeps overwriting the shelf, so
        # the next read() skips straight to the newest frame, anything between
        # was dropped, and the gap in frame numbers becomes the caller's n_steps.
        with self._cond:
            while not self._stopped and self._frame_no == self._last_served:
                self._cond.wait()
            if self._frame_no == self._last_served:
                return None  # stopped and nothing new
            self._last_served = self._frame_no
            return self._frame_no, self._frame

    def close(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        self._cap.release()

    def __enter__(self) -> "LatestFrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
