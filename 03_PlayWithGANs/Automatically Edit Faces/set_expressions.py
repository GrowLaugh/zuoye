import enum
import typing as tp

import numpy as np
from numpy.typing import NDArray

class FaceExpression(enum.Enum):
    CLOSE_EYES  = 0
    ENLARGE_EYES = 1
    CLOSE_MOUTH  = 2
    SMILE_MOUTH = 3
    SLIM_FACE   = 4

    @staticmethod
    def value_of(description: str) -> tp.Self:
        return {
            'close eyes':  FaceExpression.CLOSE_EYES,
            'enlarge eyes': FaceExpression.ENLARGE_EYES,
            'close mouth':  FaceExpression.CLOSE_MOUTH,
            'smile mouth': FaceExpression.SMILE_MOUTH,
            'slim face':   FaceExpression.SLIM_FACE,
        }[description.lower()]

def close_eyes(
    in_points: NDArray,
) -> NDArray:
    return in_points[[37, 38, 43, 44]], in_points[[41, 40, 47, 46]]

def enlarge_eyes(
    in_points: NDArray,
) -> NDArray:
    out_points: NDArray = in_points.copy()
    out_points[[37, 38]] += 0.75 * (out_points[[37, 38]] - out_points[[41, 40]])
    out_points[[43, 44]] += 0.75 * (out_points[[43, 44]] - out_points[[47, 46]])
    return in_points[[37, 38, 43, 44]], out_points[[37, 38, 43, 44]]

def close_mouth(
    in_points: NDArray,
) -> NDArray:
    out_points: NDArray = in_points.copy()
    diff: NDArray = (out_points[[63, 62, 61]] - out_points[[65, 66, 67]]).mean(axis=1)
    out_points[[65, 66, 67]] = out_points[[63, 62, 61]]
    out_points[55:58, 0] += 1.0 * diff
    return in_points[[63, 62, 61]], out_points[[65, 66, 67]]

def smile_mouth(
    in_points: NDArray,
) -> NDArray:
    out_points: NDArray = in_points.copy()
    diff: float = (out_points[54, 1] - out_points[48, 1]).item()
    out_points[54] += np.array([-diff, diff]) * 0.1
    out_points[48] += diff * -0.1
    out_points[64] += np.array([-diff, diff]) * 0.05
    out_points[60] += diff * -0.05
    return in_points[[54, 48, 64, 60]], out_points[[54, 48, 64, 60]]

def slim_face(
    in_points: NDArray,
) -> NDArray:
    out_points: NDArray = in_points.copy()
    diff: NDArray = out_points[0:8] - out_points[9:17]
    out_points[0:8]  -= 0.05 * diff
    out_points[9:17] += 0.05 * diff
    return in_points[0:17], out_points[0:17]

def raise_eyebrows(in_points: NDArray) -> NDArray:
    out_points = in_points.copy()
    out_points[17:22] += np.array([0, 0.1]) * 5
    out_points[22:27] += np.array([0, 0.1]) * 5
    return in_points[17:27], out_points[17:27]


def transform(
    expr_id: FaceExpression,
) -> tp.Callable[[NDArray], NDArray]:
    return {
        FaceExpression.CLOSE_EYES:  close_eyes,
        FaceExpression.ENLARGE_EYES: enlarge_eyes,
        FaceExpression.CLOSE_MOUTH:  close_mouth,
        FaceExpression.SMILE_MOUTH: smile_mouth,
        FaceExpression.SLIM_FACE:   slim_face,
    }[expr_id]