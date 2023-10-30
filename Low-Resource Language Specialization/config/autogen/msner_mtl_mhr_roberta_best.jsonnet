local t = import "DO_NOT_ERASE_pathfinder.libsonnet";

t.build_mtl_ner_ms("mhr", "bert", ["roberta", "best"], "mtl_mhr_roberta_best")
