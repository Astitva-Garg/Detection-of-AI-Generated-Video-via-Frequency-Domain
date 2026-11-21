# results/ : cross-validation of the resize variant

Written by the `ablation` stage of `run_pipeline.sh`: the same clips as the main pipeline, but every frame shrunk
to a short side of 256 px before the crop (`IE643_NORM=resize`) instead of a native-resolution crop. It holds the
grouped 5-fold cross-validation of that variant (`cv_summary.csv`, `cv_folds.csv`). The comparison with the main
(crop) results is `results_crop/resize_ablation.csv` and `results_crop/resize_ablation_cv.csv`. These are not the
main results; those are in `results_crop/`.
