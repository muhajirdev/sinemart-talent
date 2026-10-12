import math

import numpy as np

from sinemart.crop import EYE_Y, GREY, OUT, SIDE_IOD, Face, align_matrix, assess, grey_background, yaw_proxy


def face(re, le, nose=None, score=0.95, box=(0, 0, 100, 100)):
    nose = nose or ((re[0] + le[0]) / 2, (re[1] + le[1]) / 2 + 20)
    return Face(box, re, le, nose, score)


def apply(m, p):
    return m @ np.array([p[0], p[1], 1.0])


def test_tilted_eyes_land_level_and_fixed_distance():
    f = face((200, 300), (260, 340))  # head tilted
    m = align_matrix(f)
    r, l = apply(m, f.right_eye), apply(m, f.left_eye)
    assert abs(r[1] - l[1]) < 1e-6  # level
    assert abs(math.dist(r, l) - OUT / SIDE_IOD) < 1e-6  # same size for everyone
    mid = (r + l) / 2
    assert np.allclose(mid, [OUT / 2, EYE_Y * OUT])  # same place for everyone


def test_yaw_proxy():
    assert yaw_proxy(face((0, 0), (60, 0), nose=(30, 25))) < 1e-9
    assert abs(yaw_proxy(face((0, 0), (60, 0), nose=(48, 25))) - 0.3) < 1e-9


def test_grey_background_keeps_person_only():
    crop = np.full((4, 4, 3), 200, np.uint8)
    mask = np.zeros((4, 4))
    mask[1:3, 1:3] = 1
    out = grey_background(crop, mask)
    assert (out[0, 0] == GREY).all() and (out[1, 1] == 200).all()


def test_quality_gate_reasons():
    rng = np.random.default_rng(0)
    sharp = rng.integers(60, 200, (OUT, OUT, 3), dtype=np.uint8)
    good = face((0, 0), (60, 0), nose=(30, 25))
    assert assess([good], sharp).ok
    flat = np.full((OUT, OUT, 3), 120, np.uint8)
    q = assess([face((0, 0), (20, 0), nose=(19, 10), score=0.5)], flat)
    assert set(q.reasons) == {"low_face_score", "face_too_small", "blurry", "not_frontal"}
    group = [good, face((0, 0), (60, 0), box=(200, 0, 90, 90))]
    assert "multiple_faces" in assess(group, sharp).reasons
    assert assess([], sharp).reasons == ["no_face"]
