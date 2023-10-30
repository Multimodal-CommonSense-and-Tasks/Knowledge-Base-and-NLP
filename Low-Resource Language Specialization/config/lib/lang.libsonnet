local data_root = std.extVar("MTL_DATA"); 
local archive_root = std.extVar("MTL_ALLENNLP_OUTPUTS");
local split_seed = std.extVar("SEED_SET");
local ext_train_size = std.parseInt(std.extVar("TRAIN_SIZE"));

// local data_root = "/m-pinotHD/echau18/lr-ssmba/data";
// local archive_root = "/m-pinotHD/echau18/lr-ssmba/allennlp_outputs";
{
    "be": {
        "ner": {
            "train_size": 15000,
            "train_data_path": data_root + "/be/panx/train",
            "validation_data_path": data_root + "/be/panx/dev",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ner_be_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/be/panx/ssmba/mbert.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/be/panx/ssmba/roberta.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "tva_base": {
                    "train_data_path": data_root + "/be/panx/ssmba/tva_base.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
            },
        },
        "ud": {
            "train_size": 319,
            "train_data_path": data_root + "/be/ud/train.conllu",
            "validation_data_path": data_root + "/be/ud/dev.conllu",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ud_be_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/be/ud/ssmba/mbert.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/be/ud/ssmba/roberta.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "tva_base": {
                    "train_data_path": data_root + "/be/ud/ssmba/tva_base.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
            },
        },
    },
    "bg": {
        "ner": {
            "train_size": 20000,
            "train_data_path": data_root + "/bg/panx/train",
            "validation_data_path": data_root + "/bg/panx/dev",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ner_bg_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/bg/panx/ssmba/mbert.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/bg/panx/ssmba/roberta.train",
                    "train_size": standard.train_size + 159992,
                },
                "tva_base": {
                    "train_data_path": data_root + "/bg/panx/ssmba/tva_base.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
            },
        },
        "ud": {
            "train_size": 8907,
            "train_data_path": data_root + "/bg/ud/train.conllu",
            "validation_data_path": data_root + "/bg/ud/dev.conllu",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ud_bg_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/bg/ud/ssmba/mbert.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/bg/ud/ssmba/roberta.train.conllu",
                    "train_size": standard.train_size + 70648,
                },
                "tva_base": {
                    "train_data_path": data_root + "/bg/ud/ssmba/tva_base.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
            },
        },
    },
    "ga": {
        "ner": {
            "train_size": 1000,
            "train_data_path": data_root + "/ga/panx/train",
            "validation_data_path": data_root + "/ga/panx/dev",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ner_ga_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/ga/panx/ssmba/mbert.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/ga/panx/ssmba/roberta.train",
                    "train_size": standard.train_size * 9,
                },
                "tva_base": {
                    "train_data_path": data_root + "/ga/panx/ssmba/tva_base.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
            },
        },
        "ud": {
            "train_size": 858,
            "train_data_path": data_root + "/ga/ud/train.conllu",
            "validation_data_path": data_root + "/ga/ud/dev.conllu",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ud_ga_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/ga/ud/ssmba/mbert.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/ga/ud/ssmba/roberta.train.conllu",
                    "train_size": standard.train_size + 6824,
                },
                "tva_base": {
                    "train_data_path": data_root + "/ga/ud/ssmba/tva_base.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
            },
        },
    },
    "mhr": {
        "ner": {
            "train_size": 100,
            "train_data_path": data_root + "/mhr/panx/train",
            "validation_data_path": data_root + "/mhr/panx/dev",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ner_mhr_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/mhr/panx/ssmba/mbert.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/mhr/panx/ssmba/roberta.train",
                    "train_size": standard.train_size + 688,
                },
                "tva_base": {
                    "train_data_path": data_root + "/mhr/panx/ssmba/tva_base.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
            },
        },
        "ud": null,
    },
    "mhrlatin": {
        "ner": {
            "train_size": 100,
            "train_data_path": data_root + "/mhrlatin/panx/train",
            "validation_data_path": data_root + "/mhrlatin/panx/dev",
        },
        "ud": null,
    },
    "mhrwiktralatin": {
        "ner": {
            "train_size": 100,
            "train_data_path": data_root + "/mhrwiktralatin/panx/train",
            "validation_data_path": data_root + "/mhrwiktralatin/panx/dev",
        },
        "ud": null,
    },
    "si": {
        "ner": {
            "train_size": 100,
            "train_data_path": data_root + "/si/panx/train",
            "validation_data_path": data_root + "/si/panx/dev",
        },
        "ud": null,
    },
    "silatin": {
        "ner": {
            "train_size": 100,
            "train_data_path": data_root + "/silatin/panx/train",
            "validation_data_path": data_root + "/silatin/panx/dev",
        },
        "ud": null,
    },
    "xmf": {
        "ner": {
            "train_size": 100,
            "train_data_path": data_root + "/xmf/panx/train",
            "validation_data_path": data_root + "/xmf/panx/dev",
        },
        "ud": null,
    },
    "xmflatin": {
        "ner": {
            "train_size": 100,
            "train_data_path": data_root + "/xmflatin/panx/train",
            "validation_data_path": data_root + "/xmflatin/panx/dev",
        },
        "ud": null,
    },
    "km": {
        "ner": {
            "train_size": 100,
            "train_data_path": data_root + "/km/panx/train",
            "validation_data_path": data_root + "/km/panx/dev",
        },
        "ud": null,
    },
    "kmlatin": {
        "ner": {
            "train_size": 100,
            "train_data_path": data_root + "/kmlatin/panx/train",
            "validation_data_path": data_root + "/kmlatin/panx/dev",
        },
        "ud": null,
    },
    "ckb": {
        "ner": {
            "train_size": 1000,
            "train_data_path": data_root + "/ckb/panx/train",
            "validation_data_path": data_root + "/ckb/panx/dev",
        },
        "ud": null,
    },
    "ckblatinnfc": {
        "ner": {
            "train_size": 1000,
            "train_data_path": data_root + "/ckblatinnfkc/panx/train",
            "validation_data_path": data_root + "/ckblatinnfkc/panx/dev",
        },
        "ud": null,
    },
    "ckblatinnfkc": {
        "ner": {
            "train_size": 1000,
            "train_data_path": data_root + "/ckblatinnfkc/panx/train",
            "validation_data_path": data_root + "/ckblatinnfkc/panx/dev",
        },
        "ud": null,
    },
    "cu": {
        "ner": null,
        "ud": {
            "train_size": 4124,
            "train_data_path": data_root + "/cu/ud/train.conllu",
            "validation_data_path": data_root + "/cu/ud/dev.conllu",
        },
    },
    "culatin": {
        "ner": null,
        "ud": {
            "train_size": 4124,
            "train_data_path": data_root + "/culatin/ud/train.conllu",
            "validation_data_path": data_root + "/culatin/ud/dev.conllu",
        },
    },
    "myv": {
        "ner": null,
        "ud": {
            "train_size": ext_train_size,
            "train_data_path": data_root + "/myv/ud" + split_seed + "/train.conllu",
            "validation_data_path": data_root + "/myv/ud" + split_seed + "/dev.conllu",
        },
    },
    "myvrulatin": {
        "ner": null,
        "ud": {
            "train_size": ext_train_size,
            "train_data_path": data_root + "/myv/ud" + split_seed + "/train.conllu",
            "validation_data_path": data_root + "/myv/ud" + split_seed + "/dev.conllu",
        },
    },
    "myvlatin": {
        "ner": null,
        "ud": {
            "train_size": ext_train_size,
            "train_data_path": data_root + "/myv/ud" + split_seed + "/train.conllu",
            "validation_data_path": data_root + "/myv/ud" + split_seed + "/dev.conllu",
        },
    },
    "mtarabbw": {
        "ner": {
            "train_size": 100,
            "train_data_path": data_root + "/mt/panx/train",
            "validation_data_path": data_root + "/mt/panx/dev",
        },
        "ud": {
            "train_size": 1123,
            "train_data_path": data_root + "/mt/ud/train.conllu",
            "validation_data_path": data_root + "/mt/ud/dev.conllu",
        },
    },
    "mt": {
        "ner": {
            "train_size": 100,
            "train_data_path": data_root + "/mt/panx/train",
            "validation_data_path": data_root + "/mt/panx/dev",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ner_mt_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/mt/panx/ssmba/mbert.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/mt/panx/ssmba/roberta.train",
                    "train_size": standard.train_size * 9,
                },
                "tva_base": {
                    "train_data_path": data_root + "/mt/panx/ssmba/tva_base.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
            },
        },
        "ud": {
            "train_size": 1123,
            "train_data_path": data_root + "/mt/ud/train.conllu",
            "validation_data_path": data_root + "/mt/ud/dev.conllu",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ud_mt_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/mt/ud/ssmba/mbert.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/mt/ud/ssmba/roberta.train.conllu",
                    "train_size": standard.train_size + 8920,
                },
                "tva_base": {
                    "train_data_path": data_root + "/mt/ud/ssmba/tva_base.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
            },
        },
    },
    "ug": {
        "ner": {
            "train_size": 100,
            "train_data_path": data_root + "/ug/panx/train",
            "validation_data_path": data_root + "/ug/panx/dev",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ner_ug_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/ug/panx/ssmba/mbert.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/ug/panx/ssmba/roberta.train",
                    "train_size": standard.train_size + 640,
                },
                "tva_base": {
                    "train_data_path": data_root + "/ug/panx/ssmba/tva_base.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
            },
        },
        "ud": {
            "train_size": 1656,
            "train_data_path": data_root + "/ug/ud/train.conllu",
            "validation_data_path": data_root + "/ug/ud/dev.conllu",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ud_ug_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/ug/ud/ssmba/mbert.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/ug/ud/ssmba/roberta.train.conllu",
                    "train_size": standard.train_size * 9,
                },
                "tva_base": {
                    "train_data_path": data_root + "/ug/ud/ssmba/tva_base.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
            },
        },
    },
    "uglatin": {
        "ner": {
            "train_size": 100,
            "train_data_path": data_root + "/uglatin/panx/train",
            "validation_data_path": data_root + "/uglatin/panx/dev",
        },
        "ud": {
            "train_size": 1656,
            "train_data_path": data_root + "/uglatin/ud/train.conllu",
            "validation_data_path": data_root + "/uglatin/ud/dev.conllu",
        },
    },
    "uglatinnfc": {
        "ner": {
            "train_size": 100,
            "train_data_path": data_root + "/uglatinnfc/panx/train",
            "validation_data_path": data_root + "/uglatinnfc/panx/dev",
        },
        "ud": {
            "train_size": 1656,
            "train_data_path": data_root + "/uglatinnfc/ud/train.conllu",
            "validation_data_path": data_root + "/uglatinnfc/ud/dev.conllu",
        },
    },
    "uglatinnfkc": {
        "ner": {
            "train_size": 100,
            "train_data_path": data_root + "/uglatinnfkc/panx/train",
            "validation_data_path": data_root + "/uglatinnfkc/panx/dev",
        },
        "ud": {
            "train_size": 1656,
            "train_data_path": data_root + "/uglatinnfkc/ud/train.conllu",
            "validation_data_path": data_root + "/uglatinnfkc/ud/dev.conllu",
        },
    },
    "ur": {
        "ner": {
            "train_size": 20000,
            "train_data_path": data_root + "/ur/panx/train",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ner_ur_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/ur/panx/ssmba/mbert.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/ur/panx/ssmba/roberta.train",
                    "train_size": standard.train_size * 9,
                },
                "tva_base": {
                    "train_data_path": data_root + "/ur/panx/ssmba/tva_base.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
            },
            "validation_data_path": data_root + "/ur/panx/dev",
        },
        "ud": {
            "train_size": 4043,
            "train_data_path": data_root + "/ur/ud/train.conllu",
            "validation_data_path": data_root + "/ur/ud/dev.conllu",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ud_ur_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/ur/ud/ssmba/mbert.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/ur/ud/ssmba/roberta.train.conllu",
                    "train_size": standard.train_size * 9,
                },
                "tva_base": {
                    "train_data_path": data_root + "/ur/ud/ssmba/tva_base.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
            },
        },
    },
    "vi": {
        "ner": {
            "train_size": 20000,
            "train_data_path": data_root + "/vi/panx/train",
            "validation_data_path": data_root + "/vi/panx/dev",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ner_vi_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/vi/panx/ssmba/mbert.train",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/vi/panx/ssmba/roberta.train",
                    "train_size": standard.train_size * 9,
                },
                "tva_base": {
                    "train_data_path": data_root + "/vi/panx/ssmba/tva_base.train",
                    // original +  8x augmented
                    "train_size": standard.train_size + 159999,
                },
            },
        },
        "ud": {
            "train_size": 1400,
            "train_data_path": data_root + "/vi/ud/train.conllu",
            "validation_data_path": data_root + "/vi/ud/dev.conllu",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ud_vi_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/vi/ud/ssmba/mbert.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/vi/ud/ssmba/roberta.train.conllu",
                    "train_size": standard.train_size + 11192,
                },
                "tva_base": {
                    "train_data_path": data_root + "/vi/ud/ssmba/tva_base.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
            },
        },
    },
    "wo": {
        "ner": null,
        "ud": {
            "train_size": 1188,
            "train_data_path": data_root + "/wo/ud/train.conllu",
            "validation_data_path": data_root + "/wo/ud/dev.conllu",
            local standard = self,
            "selfsup": {
                "teacher_model_archive": archive_root + "/ud_wo_tva_best/model.tar.gz",
                "mbert": {
                    "train_data_path": data_root + "/wo/ud/ssmba/mbert.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
                "roberta": {
                    "train_data_path": data_root + "/wo/ud/ssmba/roberta.train.conllu",
                    "train_size": standard.train_size + 9496,
                },
                "tva_base": {
                    "train_data_path": data_root + "/wo/ud/ssmba/tva_base.train.conllu",
                    // original +  8x augmented
                    "train_size": standard.train_size * 9,
                },
            },
        },
    },
}
