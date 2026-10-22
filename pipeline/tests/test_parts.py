import numpy as np

from sinemart.facegeom import KEY
from sinemart.parts import PARTS, boxes


def test_boxes_sit_in_face_order():
    # A rough frontal face on a 512 px crop (eyes level at y=205).
    at = {"top": (256, 90), "chin": (256, 420), "cheek_r": (150, 300), "cheek_l": (362, 300),
          "eye_r_out": (190, 205), "eye_l_out": (322, 205), "pupil_r": (205, 205), "pupil_l": (307, 205),
          "nose_tip": (256, 280), "ala_r": (230, 285), "ala_l": (282, 285),
          "mouth_r": (220, 335), "mouth_l": (292, 335), "lip_top": (256, 325), "lip_bottom": (256, 345)}
    p = np.zeros((478, 3))
    for name, i in KEY.items():
        p[i, :2] = at[name]
    b = boxes(p)
    assert list(b) == list(PARTS)
    for left, top, right, bottom in b.values():
        assert 0 <= left < right <= 512 and 0 <= top < bottom <= 512
    centre_y = {k: (v[1] + v[3]) / 2 for k, v in b.items()}
    assert centre_y["eyes"] < centre_y["nose"] < centre_y["mouth"] < centre_y["jaw"]
