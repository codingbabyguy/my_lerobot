#!/usr/bin/env python3

from __future__ import annotations

import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass


@dataclass
class ControlState:
    paused: bool = False
    estop: bool = False
    quit: bool = False
    last_key: str = ""


class KeyboardController:
    """Non-blocking keyboard control for Linux terminal.

    Keys:
    - p: pause motion
    - c: continue motion
    - e: emergency stop and quit
    - q: graceful quit
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and sys.stdin.isatty()
        self.state = ControlState()
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._old_term = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._old_term = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._old_term is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_term)

    def snapshot(self) -> ControlState:
        with self._lock:
            return ControlState(
                paused=self.state.paused,
                estop=self.state.estop,
                quit=self.state.quit,
                last_key=self.state.last_key,
            )

    def clear_last_key(self) -> None:
        with self._lock:
            self.state.last_key = ""

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            key = sys.stdin.read(1).strip().lower()
            if not key:
                continue
            with self._lock:
                self.state.last_key = key
                if key == "p":
                    self.state.paused = True
                elif key == "c":
                    self.state.paused = False
                elif key == "e":
                    self.state.estop = True
                    self.state.quit = True
                elif key == "q":
                    self.state.quit = True
            time.sleep(0.01)
