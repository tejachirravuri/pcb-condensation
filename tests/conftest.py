import cv2
import numpy as np
import pytest


def comb_image(h=240, w=320, fingers=6, finger=10, gap=8):
    """Synthetic test structure: two interlocking combs, bluish metal on a
    yellow board. Returns (rgb, labels) with labels 1 and 2 for the combs."""
    board = np.zeros((h, w, 3), np.uint8)
    board[:] = (190, 170, 60)                       # yellow solder mask (RGB)
    lab = np.zeros((h, w), np.uint8)
    top, bottom, left, right = 20, h - 20, 30, w - 30
    lab[top:top + finger, left:right] = 1           # spine of comb 1
    lab[bottom - finger:bottom, left:right] = 2     # spine of comb 2
    pitch = finger + gap
    for k in range(fingers * 2):
        x = left + 10 + k * pitch
        if x + finger > right - 10:
            break
        if k % 2 == 0:
            lab[top:bottom - finger - gap, x:x + finger] = 1
        else:
            lab[top + finger + gap:bottom, x:x + finger] = 2
    board[lab > 0] = (150, 170, 210)                # metal
    return board, lab


@pytest.fixture
def combs():
    return comb_image()


def disc(shape, cy, cx, r):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m
