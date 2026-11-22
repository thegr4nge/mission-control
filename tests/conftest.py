"""Tests never push to a live browser: the default bridge address is switched off for the whole run.
Tests that exercise the bridge set their own address (monkeypatch or server_params(bridge=...))."""

import os

os.environ["MISSION_CONTROL_BRIDGE"] = "off"
