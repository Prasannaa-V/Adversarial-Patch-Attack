"""
inference/frame_store.py
Saves flagged attack frames as JPEG files with timestamp-based naming.
Auto-cleans frames older than a configured number of days.
"""

import os
import cv2
import glob
import logging
import time
import numpy as np
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class FrameStore:
    """
    Persists adversarial attack frames to disk as JPEG images.
    Directory: data/frames/YYYY-MM-DD/frame_<id>.jpg
    """

    def __init__(self, frames_dir: str = "data/frames", max_days: int = 7):
        self.frames_dir = frames_dir
        self.max_days = max_days
        os.makedirs(frames_dir, exist_ok=True)
        logger.info(f"FrameStore initialized at '{frames_dir}' (retention: {max_days} days)")

    def save_frame(self, frame: np.ndarray, frame_id: str) -> str:
        """
        Save an attack frame as JPEG.

        Args:
            frame: BGR numpy array
            frame_id: unique frame identifier

        Returns:
            Absolute path to saved file, or empty string on failure
        """
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day_dir = os.path.join(self.frames_dir, date_str)
        os.makedirs(day_dir, exist_ok=True)

        filename = f"attack_{frame_id}_{int(time.time() * 1000)}.jpg"
        filepath = os.path.join(day_dir, filename)

        success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if success:
            logger.debug(f"Frame saved: {filepath}")
        else:
            logger.error(f"Failed to save frame: {filepath}")
            return ""
        return os.path.abspath(filepath)

    def cleanup_old_frames(self, max_days: int = None) -> int:
        """
        Delete frames older than max_days. Returns count of deleted files.
        Defaults to self.max_days when not specified.
        """
        retain_days = max_days if max_days is not None else self.max_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=retain_days)
        deleted = 0
        pattern = os.path.join(self.frames_dir, "????-??-??", "*.jpg")
        for path in glob.glob(pattern):
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
                if mtime < cutoff:
                    os.remove(path)
                    deleted += 1
            except OSError:
                pass
        if deleted:
            logger.info(f"Cleaned up {deleted} old frame(s) from '{self.frames_dir}'")
        return deleted

    def list_frames(self, date: str = None) -> list:
        """List all saved frame paths, optionally filtered by date (YYYY-MM-DD)."""
        if date:
            pattern = os.path.join(self.frames_dir, date, "*.jpg")
        else:
            pattern = os.path.join(self.frames_dir, "????-??-??", "*.jpg")
        return sorted(glob.glob(pattern), reverse=True)
