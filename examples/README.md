# examples/

Eighteen clips for the app and for `python -m src.demo_live --video <file>`: three per source (real, LTX-Video, AnimateDiff, SVD, ModelScope, Wan2.1). All are **held-out test clips** (never used for training or model selection), already normalised (14 frames, 256 x 256 native crop, H.264 CRF 18). They were drawn at random with seed 643 by `tools/make_examples.py`, not hand-picked. ModelScope and Wan2.1 were never seen in training.

Verdicts below are from the shipped fusion MLP; wrong ones are kept in.

| file                                 | truth   | source      | scene              |   P(AI), MLP fusion | verdict   | correct   |
|:-------------------------------------|:--------|:------------|:-------------------|--------------------:|:----------|:----------|
| ai_animatediff__animatediff_0001.mp4 | AI      | animatediff | ocean waves        |               0.905 | AI        | yes       |
| ai_animatediff__animatediff_0099.mp4 | AI      | animatediff | desert sand        |               0.998 | AI        | yes       |
| ai_animatediff__animatediff_0101.mp4 | AI      | animatediff | snow ice           |               1     | AI        | yes       |
| ai_ltx__ltx_0001.mp4                 | AI      | ltx         | ocean waves        |               0.995 | AI        | yes       |
| ai_ltx__ltx_0080.mp4                 | AI      | ltx         | night sky stars    |               1     | AI        | yes       |
| ai_ltx__ltx_0140.mp4                 | AI      | ltx         | night sky stars    |               1     | AI        | yes       |
| ai_modelscope__modelscope_0017.mp4   | AI      | modelscope  | sunset sunrise     |               0     | real      | NO        |
| ai_modelscope__modelscope_0022.mp4   | AI      | modelscope  | night sky stars    |               0.99  | AI        | yes       |
| ai_modelscope__modelscope_0089.mp4   | AI      | modelscope  | forest trees       |               0     | real      | NO        |
| ai_svd__svd_10279641.mp4             | AI      | svd         | clouds sky         |               0.514 | AI        | yes       |
| ai_svd__svd_6016047.mp4              | AI      | svd         | snow ice           |               0.859 | AI        | yes       |
| ai_svd__svd_6841923.mp4              | AI      | svd         | fog mist           |               0.016 | real      | NO        |
| ai_wan__wan_0016.mp4                 | AI      | wan         | clouds sky         |               0.001 | real      | NO        |
| ai_wan__wan_0031.mp4                 | AI      | wan         | leaves plants      |               0.657 | AI        | yes       |
| ai_wan__wan_0044.mp4                 | AI      | wan         | snow ice           |               0.003 | real      | NO        |
| real__real_11072713.mp4              | real    | real        | lake water surface |               0     | real      | yes       |
| real__real_2865145.mp4               | real    | real        | clouds sky         |               0.999 | AI        | NO        |
| real__real_854332.mp4                | real    | real        | snow ice           |               0     | real      | yes       |
