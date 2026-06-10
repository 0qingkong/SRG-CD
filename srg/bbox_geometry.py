from __future__ import annotations

import math


def clip_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box

    x1 = max(0.0, min(float(x1), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    x2 = max(0.0, min(float(x2), float(width)))
    y2 = max(0.0, min(float(y2), float(height)))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def intersection_box(box_a: list[float], box_b: list[float]) -> list[float]:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    return [
        max(ax1, bx1),
        max(ay1, by1),
        min(ax2, bx2),
        min(ay2, by2),
    ]


def intersection_area(box_a: list[float], box_b: list[float]) -> float:
    return box_area(intersection_box(box_a, box_b))


def iou(box_a: list[float], box_b: list[float]) -> float:
    inter = intersection_area(box_a, box_b)
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def containment_ratio(inner_box: list[float], outer_box: list[float]) -> float:
    inner_area = box_area(inner_box)
    if inner_area <= 0:
        return 0.0

    inter = intersection_area(inner_box, outer_box)
    return inter / inner_area


def center_distance(box_a: list[float], box_b: list[float]) -> float:
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)

    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def normalized_center_distance(
    box_a: list[float],
    box_b: list[float],
    width: int,
    height: int,
) -> float:
    diag = math.sqrt(width ** 2 + height ** 2)
    if diag <= 0:
        return 0.0

    return center_distance(box_a, box_b) / diag


def overlap_ratio_x(box_a: list[float], box_b: list[float]) -> float:
    ax1, _, ax2, _ = box_a
    bx1, _, bx2, _ = box_b

    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    min_width = min(max(0.0, ax2 - ax1), max(0.0, bx2 - bx1))

    if min_width <= 0:
        return 0.0

    return overlap / min_width


def overlap_ratio_y(box_a: list[float], box_b: list[float]) -> float:
    _, ay1, _, ay2 = box_a
    _, by1, _, by2 = box_b

    overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
    min_height = min(max(0.0, ay2 - ay1), max(0.0, by2 - by1))

    if min_height <= 0:
        return 0.0

    return overlap / min_height


def compute_geometry_features(
    subject_box: list[float],
    object_box: list[float],
    width: int,
    height: int,
) -> dict:
    subject_box = clip_box(subject_box, width, height)
    object_box = clip_box(object_box, width, height)

    s_cx, s_cy = box_center(subject_box)
    o_cx, o_cy = box_center(object_box)

    dx = s_cx - o_cx
    dy = s_cy - o_cy

    return {
        "subject_box_clipped": subject_box,
        "object_box_clipped": object_box,
        "subject_center": [s_cx, s_cy],
        "object_center": [o_cx, o_cy],
        "dx": dx,
        "dy": dy,
        "abs_dx": abs(dx),
        "abs_dy": abs(dy),
        "subject_area": box_area(subject_box),
        "object_area": box_area(object_box),
        "iou": iou(subject_box, object_box),
        "subject_in_object_ratio": containment_ratio(subject_box, object_box),
        "object_in_subject_ratio": containment_ratio(object_box, subject_box),
        "center_distance": center_distance(subject_box, object_box),
        "normalized_center_distance": normalized_center_distance(
            subject_box,
            object_box,
            width,
            height,
        ),
        "x_overlap_ratio": overlap_ratio_x(subject_box, object_box),
        "y_overlap_ratio": overlap_ratio_y(subject_box, object_box),
    }


def infer_relation_v2(
    subject_box: list[float],
    object_box: list[float],
    width: int,
    height: int,
    caption_relation: str | None = None,
) -> dict:
    features = compute_geometry_features(subject_box, object_box, width, height)

    dx = features["dx"]
    dy = features["dy"]
    abs_dx = features["abs_dx"]
    abs_dy = features["abs_dy"]

    norm_dist = features["normalized_center_distance"]
    subj_in_obj = features["subject_in_object_ratio"]
    obj_in_subj = features["object_in_subject_ratio"]

    caption_key = ""
    if caption_relation:
        caption_key = caption_relation.strip().lower().replace(" ", "_")

    # 优先为 caption 所在关系组计算对应关系
    if caption_key in {"inside", "outside"}:
        if subj_in_obj >= 0.65:
            relation = "inside"
        else:
            relation = "outside"

        return {
            "computed_relation": relation,
            "relation_family": "containment",
            "geometry_features": features,
        }

    if caption_key in {"near", "next_to", "far_from"}:
        if norm_dist <= 0.30:
            relation = "near"
        else:
            relation = "far_from"

        return {
            "computed_relation": relation,
            "relation_family": "distance",
            "geometry_features": features,
        }

    # 默认方向关系
    if abs_dx >= abs_dy:
        relation = "right_of" if dx > 0 else "left_of"
        family = "horizontal"
    else:
        relation = "below" if dy > 0 else "above"
        family = "vertical"

    return {
        "computed_relation": relation,
        "relation_family": family,
        "geometry_features": features,
    }
