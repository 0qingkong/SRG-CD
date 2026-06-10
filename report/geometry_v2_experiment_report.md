# SRG-CD Geometry v2 Experiment Report

## 1. Experiment Overview

This experiment evaluates whether explicit bounding-box-level Spatial Relation Graphs can provide visual evidence for diagnosing spatial relation statements in VSR-derived SRG-Bench v0.1.

The pipeline is:

```text
SRG-Bench v0.1
↓
OWL-ViT open-vocabulary object detection
↓
subject/object bounding box extraction
↓
Geometry v2 spatial relation inference
↓
BBox-level Spatial Relation Graph construction
↓
Caption-SRG and BBox-SRG diagnostic comparison

| Item                     |  Count |
| ------------------------ | -----: |
| SRG-Bench v0.1 samples   |   1408 |
| BBox-SRG success records |    976 |
| BBox-SRG failed records  |    432 |
| Detection success rate   | 69.32% |


| Relation family | Count |
| --------------- | ----: |
| horizontal      |   384 |
| vertical        |   303 |
| distance        |   173 |
| containment     |   116 |



| BBox relation | Count |
| ------------- | ----: |
| right_of      |   199 |
| left_of       |   185 |
| above         |   176 |
| below         |   127 |
| near          |    89 |
| far_from      |    84 |
| outside       |    72 |
| inside        |    44 |




| Metric                       |  Value |
| ---------------------------- | -----: |
| Supported diagnostic records |    976 |
| Diagnostic correct           |    694 |
| Diagnostic incorrect         |    282 |
| Diagnostic consistency rate  | 71.11% |



| Original label | Total | Diagnostic correct | Diagnostic rate |
| -------------- | ----: | -----------------: | --------------: |
| False          |   468 |                371 |          79.27% |
| True           |   508 |                323 |          63.58% |


| Relation group | Total | Diagnostic correct | Diagnostic rate |
| -------------- | ----: | -----------------: | --------------: |
| containment    |   116 |                 83 |          71.55% |
| distance       |   173 |                 98 |          56.65% |
| horizontal     |   168 |                138 |          82.14% |
| vertical       |   519 |                375 |          72.25% |



| Relation family | Total | Diagnostic correct | Diagnostic rate |
| --------------- | ----: | -----------------: | --------------: |
| containment     |   116 |                 83 |          71.55% |
| distance        |   173 |                 98 |          56.65% |
| horizontal      |   384 |                266 |          69.27% |
| vertical        |   303 |                247 |          81.52% |


9. Key Findings
OWL-ViT provides a usable open-vocabulary detection backbone for constructing BBox-level SRGs, reaching a 69.32% subject-object joint detection success rate on the full SRG-Bench v0.1 set.
Geometry v2 extends the spatial reasoning module from simple directional relations to four relation families: horizontal, vertical, distance, and containment.
The full Geometry v2 pipeline achieves 71.11% diagnostic consistency on successfully detected samples.
The method is stronger on false statements, reaching 79.27% diagnostic consistency, which supports the project goal of structured counterfactual diagnosis.
Distance relations remain the weakest category, with a diagnostic rate of 56.65%, suggesting the need for adaptive distance thresholds and scale-aware geometry modeling.
10. Limitations

Current limitations include:

OWL-ViT sometimes fails to detect subject or object.
Some detected boxes are too large or imprecise.
Distance relations rely on a fixed normalized center-distance threshold.
Containment relations rely on simple overlap ratios.
Spatial relations such as "on top of" and "under" may require contact, support, or scene-context reasoning beyond bounding-box geometry.


11. Next Steps

The next stage will focus on VLM-based diagnostic evaluation:

image + original caption
image + relation-flip counterfactual
image + object-swap counterfactual
image + wrong-SRG conflict prompt
↓
VLM response
↓
structured diagnostic evaluation

Candidate models include Qwen-VL, InternVL, LLaVA, and other open-source VLMs.

The final goal is to evaluate whether VLMs can detect conflicts between language-level spatial statements and image-level spatial evidence.


