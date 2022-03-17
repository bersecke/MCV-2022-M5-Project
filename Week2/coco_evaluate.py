from detectron2.evaluation import COCOEvaluator

evaluator = COCOEvaluator("balloon_val", output_dir="./output")