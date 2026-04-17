# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for PoseAnalyzer — shelf-to-waist sliding window detection."""

import asyncio
import time

import pytest

from services.pose_analyzer import PoseAnalyzer, PoseFrame


class FakeConfig:
    def get_pose_config(self):
        return {
            "window_size": 10,
            "confidence_threshold": 0.5,
            "waist_proximity_threshold": 0.10,
        }


@pytest.fixture
def analyzer():
    return PoseAnalyzer(FakeConfig())


def _shelf_frame(ts, side="right"):
    """Wrist raised above hip level."""
    f = PoseFrame(
        timestamp=ts,
        left_hip=(0.5, 0.6),
        right_hip=(0.55, 0.6),
        left_hip_conf=0.9,
        right_hip_conf=0.9,
    )
    if side == "right":
        f.right_wrist = (0.52, 0.3)   # well above hip_y=0.6
        f.right_wrist_conf = 0.9
    else:
        f.left_wrist = (0.48, 0.3)
        f.left_wrist_conf = 0.9
    return f


def _waist_frame(ts, side="right"):
    """Wrist at waist level (close to hip midpoint 0.525, 0.6)."""
    f = PoseFrame(
        timestamp=ts,
        left_hip=(0.5, 0.6),
        right_hip=(0.55, 0.6),
        left_hip_conf=0.9,
        right_hip_conf=0.9,
    )
    if side == "right":
        f.right_wrist = (0.525, 0.605)   # very close to hip midpoint
        f.right_wrist_conf = 0.9
    else:
        f.left_wrist = (0.525, 0.605)
        f.left_wrist_conf = 0.9
    return f


def _neutral_frame(ts):
    """Hand at middle height — neither shelf nor waist."""
    return PoseFrame(
        timestamp=ts,
        left_hip=(0.5, 0.6),
        right_hip=(0.55, 0.6),
        left_hip_conf=0.9,
        right_hip_conf=0.9,
        right_wrist=(0.52, 0.45),
        right_wrist_conf=0.9,
    )


@pytest.mark.asyncio
async def test_shelf_to_waist_detected(analyzer):
    """Full shelf-to-waist pattern triggers the flag handler."""
    flagged = []

    async def on_flag(oid, wrist, frames):
        flagged.append((oid, wrist))

    analyzer.register_flag_handler(on_flag)

    # First 5 frames: shelf (wrist above hip) — need ≥2 consecutive
    for i in range(5):
        await analyzer.on_pose_update("p1", _shelf_frame(float(i)))

    # Next 5 frames: waist (wrist near hip midpoint) — need ≥3 consecutive
    for i in range(5, 10):
        await analyzer.on_pose_update("p1", _waist_frame(float(i)))

    assert len(flagged) == 1
    assert flagged[0] == ("p1", "right")


@pytest.mark.asyncio
async def test_no_flag_without_shelf(analyzer):
    """Only waist frames without shelf step should NOT flag."""
    flagged = []

    async def on_flag(oid, wrist, frames):
        flagged.append(oid)

    analyzer.register_flag_handler(on_flag)

    for i in range(10):
        await analyzer.on_pose_update("p1", _waist_frame(float(i)))

    assert len(flagged) == 0


@pytest.mark.asyncio
async def test_no_flag_without_waist(analyzer):
    """Only shelf frames without waist step should NOT flag."""
    flagged = []

    async def on_flag(oid, wrist, frames):
        flagged.append(oid)

    analyzer.register_flag_handler(on_flag)

    for i in range(10):
        await analyzer.on_pose_update("p1", _shelf_frame(float(i)))

    assert len(flagged) == 0


@pytest.mark.asyncio
async def test_reset_clears_window(analyzer):
    """After reset(), the window starts fresh."""
    flagged = []

    async def on_flag(oid, wrist, frames):
        flagged.append(oid)

    analyzer.register_flag_handler(on_flag)

    for i in range(5):
        await analyzer.on_pose_update("p1", _shelf_frame(float(i)))

    analyzer.reset("p1")

    # Should need full 10 frames again to trigger
    for i in range(5):
        await analyzer.on_pose_update("p1", _shelf_frame(float(i)))

    assert len(flagged) == 0  # only 5 frames, not enough
