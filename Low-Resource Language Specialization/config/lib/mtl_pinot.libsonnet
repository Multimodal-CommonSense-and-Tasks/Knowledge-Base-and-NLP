local languages = import "lang.libsonnet";
local representations = import "rep.libsonnet";
local common = import "common.libsonnet";
local batch_sizes = {
    "be": {
        "bert": {
            "mbert": 16,
            "vambert": 16,
            "lapt": 16,
            "roberta": 64,
        },
        "fasttext": 64,
    },
    "bg": {
        "bert": {
            "mbert": 16,
            "vambert": 16,
            "lapt": 16,
            "roberta": 32,
        },
        "fasttext": 64,
    },
    "ga": {
        "bert": {
            "mbert": 4,
            "vambert": 4,
            "lapt": 4,
            "roberta": 8,
        },
        "fasttext": 8,
    },
    "mhr": {
        "bert": {
            "mbert": 64,
            "vambert": 64,
            "lapt": 64,
            "roberta": 64,
        },
        "fasttext": 64,
    },
    "mt": {
        "bert": {
            "mbert": 32,
            "vambert": 32,
            "lapt": 32,
            "roberta": 64,
        },
        "fasttext": 64,
    },
    "ug": {
        "bert": {
            "mbert": 32,
            "vambert": 32,
            "lapt": 32,
            "roberta": 64,
        },
        "fasttext": 64,
    },
    "ur": {
        "bert": {
            "mbert": 16,
            "vambert": 16,
            "lapt": 16,
            "roberta": 8,
        },
        "fasttext": 64,
    },
    "vi": {
        "bert": {
            "mbert": 64,
            "vambert": 64,
            "lapt": 64,
            "roberta": 64,
        },
        "fasttext": 64,
    },
    "wo": {
        "bert": {
            "mbert": 64,
            "vambert": 64,
            "lapt": 64,
            "roberta": 64,
        },
        "fasttext": 64,
    },
};

local ner_batch_sizes = {
    "be": {
        "bert": {
            "mbert": 32,
            "vambert": 32,
            "lapt": 32,
            "roberta": 64,
        },
        "fasttext": 128,
    },
    "bg": {
        "bert": {
            "mbert": 16,
            "vambert": 16,
            "lapt": 16,
            "roberta": 64,
        },
        "fasttext": 128,
    },
    "ga": {
        "bert": {
            "mbert": 32,
            "vambert": 32,
            "lapt": 32,
            "roberta": 128,
        },
        "fasttext": 128,
    },
    "mhr": {
        "bert": {
            "mbert": 128,
            "vambert": 128,
            "lapt": 128,
            "roberta": 128,
        },
        "fasttext": 128,
    },
    "mt": {
        "bert": {
            "mbert": 128,
            "vambert": 128,
            "lapt": 128,
            "ssmba_vambert": 128,
            "roberta": 128,
        },
        "fasttext": 128,
    },
    "ug": {
        "bert": {
            "mbert": 32,
            "vambert": 32,
            "lapt": 32,
            "roberta": 128,
        },
        "fasttext": 128,
    },
    "ur": {
        "bert": {
            "mbert": 16,
            "vambert": 16,
            "lapt": 16,
            "roberta": 8,
        },
        "fasttext": 128,
    },
    "vi": {
        "bert": {
            "mbert": 64,
            "vambert": 64,
            "lapt": 64,
            "roberta": 128,
        },
        "fasttext": 128,
    },
    // "wo": {
    //     "bert": {
    //         "mbert": 128,
    //         "vambert": 128,
    //         "lapt": 128,
    //         "roberta": 128,
    //     },
    //     "fasttext": 128,
    // },
};

local pos_batch_sizes = {
    "be": {
        "bert": {
            "mbert": 16,
            "vambert": 16,
            "lapt": 16,
            "roberta": 64,
        },
        "fasttext": 128,
    },
    "bg": {
        "bert": {
            "mbert": 64,
            "vambert": 64,
            "lapt": 64,
            "roberta": 128,
        },
        "fasttext": 128,
    },
    "ga": {
        "bert": {
            "mbert": 4,
            "vambert": 4,
            "lapt": 4,
            "roberta": 16,
        },
        "fasttext": 128,
    },
    // "mhr": {
    //     "bert": {
    //         "mbert": 128,
    //         "vambert": 128,
    //         "lapt": 128,
    //         "roberta": 128,
    //     },
    //     "fasttext": 128,
    // },
    "mt": {
        "bert": {
            "mbert": 32,
            "vambert": 32,
            "lapt": 32,
            "roberta": 64,
        },
        "fasttext": 128,
    },
    "ug": {
        "bert": {
            "mbert": 64,
            "vambert": 32,
            "lapt": 32,
            "roberta": 128,
        },
        "fasttext": 128,
    },
    "ur": {
        "bert": {
            "mbert": 16,
            "vambert": 16,
            "lapt": 16,
            "roberta": 64,
        },
        "fasttext": 128,
    },
    "vi": {
        "bert": {
            "mbert": 64,
            "vambert": 64,
            "lapt": 64,
            "roberta": 128,
        },
        "fasttext": 128,
    },
    "wo": {
        "bert": {
            "mbert": 64,
            "vambert": 64,
            "lapt": 64,
            "roberta": 128,
        },
        "fasttext": 128,
    },
};

local ud_batch_sizes = {
    "be": {
        "bert": {
            "mbert": 16,
            "vambert": 16,
            "lapt": 16,
            "roberta": 64,
        },
        "fasttext": 64,
    },
    "bg": {
        "bert": {
            "mbert": 64,
            "vambert": 64,
            "lapt": 64,
            "roberta": 64,
        },
        "fasttext": 64,
    },
    "ga": {
        "bert": {
            "mbert": 4,
            "vambert": 4,
            "lapt": 4,
            "roberta": 8,
        },
        "fasttext": 8,
    },
    // "mhr": {
    //     "bert": {
    //         "mbert": 64,
    //         "vambert": 64,
    //         "lapt": 64,
    //         "roberta": 64,
    //     },
    //     "fasttext": 64,
    // },
    "mt": {
        "bert": {
            "mbert": 32,
            "vambert": 32,
            "lapt": 32,
            "roberta": 64,
        },
        "fasttext": 64,
    },
    "ug": {
        "bert": {
            "mbert": 32,
            "vambert": 32,
            "lapt": 32,
            "roberta": 64,
        },
        "fasttext": 64,
    },
    "ur": {
        "bert": {
            "mbert": 16,
            "vambert": 16,
            "lapt": 16,
            "roberta": 64,
        },
        "fasttext": 64,
    },
    "vi": {
        "bert": {
            "mbert": 64,
            "vambert": 64,
            "lapt": 64,
            "roberta": 64,
        },
        "fasttext": 64,
    },
    "wo": {
        "bert": {
            "mbert": 64,
            "vambert": 64,
            "lapt": 64,
            "roberta": 64,
        },
        "fasttext": 64,
    },
};
{
    build_mtl(language, emb_type, params)::
    local batch_size = if emb_type == "fasttext" then batch_sizes[language]["fasttext"]
        else batch_sizes[language][emb_type][params[0]];
    local lang_ud = languages[language]["ud"];
    local lang_ner = languages[language]["ner"];
    local train_size_ud = lang_ud["train_size"];
    local train_size_ner = lang_ner["train_size"];
    local train_size = 2 * std.max(train_size_ud, train_size_ner);
    local representation_builder = representations[emb_type];
    local representation = representation_builder.build(language, batch_size, train_size, params);
    if (lang_ud == null || lang_ner == null || representation == null) then null else {
        "numpy_seed": common["numpy_seed"],
        "pytorch_seed": common["pytorch_seed"],
        "random_seed": common["random_seed"],
        "dataset_reader": {
            "type": "multitask",
            "readers": {
                "ner": {
                    "type": "wikiann",
                    "token_indexers": representation["indexers"],
                },
                "ud": {
                    "type": "universal_dependencies",
                    "token_indexers": representation["indexers"],
                },
                "pos": {
                    "type": "universal_dependencies",
                    "token_indexers": representation["indexers"],
                    // NOTE: this is supposed to make things harder
                    // but Belarusian has some sentences without xpos
                    "use_language_specific_pos": language != "be",
                },
            },
        },
        "train_data_path": {
            "ner": lang_ner["train_data_path"],
            "ud": lang_ud["train_data_path"],
            "pos": lang_ud["train_data_path"],
        },
        "validation_data_path": {
            "ner": lang_ner["validation_data_path"],
            "ud": lang_ud["validation_data_path"],
            "pos": lang_ud["validation_data_path"],
        },
        "model": {
            "type": "multitask",
            "arg_name_mapping": {
                "backbone": {
                    "tokens": "text",
                    "words": "text",
                },
            },
            "backbone": {
                "type": "embedder_and_mask",
                "text_field_embedder": representation["embedders"],
            },
            "heads": {
                "ud": {
                    "type": "biaffine_parser",
                    "arc_representation_dim": 100,
                    "tag_representation_dim": 100,
                    // NEW!
                    "use_mst_decoding_for_validation": true,
                    "dropout": 0.3,
                    "input_dropout": 0.3,
                    "encoder": representation["encoders"]["ud"],
                    "initializer": {
                        "regexes": [
                            [".*projection.*weight", {"type": "xavier_uniform"}],
                            [".*projection.*bias", {"type": "zero"}],
                            [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
                            [".*tag_bilinear.*bias", {"type": "zero"}],
                            [".*weight_ih.*", {"type": "xavier_uniform"}],
                            [".*weight_hh.*", {"type": "orthogonal"}],
                            [".*bias_ih.*", {"type": "zero"}],
                            [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
                        ]
                    },
                },
                "ner": {
                    "type": "crf_tagger",
                    "encoder": representation["encoders"]["ner"],
                    "include_start_end_transitions": false,
                    // following SciBERT
                    "dropout": 0.1,
                    "calculate_span_f1": true,
                    "constrain_crf_decoding": true,
                    "label_encoding": "BIO",
                },
                "pos": {
                    "type": "linear_tagger",
                    "encoder": representation["encoders"]["pos"],
                    "dropout": 0.4,
                    "initializer": {
                        "regexes": [
                            [".*projection.*weight", {"type": "xavier_uniform"}],
                            [".*projection.*bias", {"type": "zero"}],
                            [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
                            [".*tag_bilinear.*bias", {"type": "zero"}],
                            [".*weight_ih.*", {"type": "xavier_uniform"}],
                            [".*weight_hh.*", {"type": "orthogonal"}],
                            [".*bias_ih.*", {"type": "zero"}],
                            [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
                        ]
                    },
                },
            },
        },
        "data_loader": {
            "type": "multitask",
            "shuffle": true,
            "scheduler": {
                "type": "unbalanced_homogeneous_roundrobin",
                "batch_size": batch_size,
                "dataset_sizes": {
                    "ud": train_size_ud,
                    "ner": train_size_ner,
                    "pos": train_size_ud,
                },
            },
        },
        "validation_data_loader": {
            "type": "multitask",
            "shuffle": true,
            "scheduler": {
                "type": "homogeneous_roundrobin",
                "batch_size": batch_size,
            },
        },
        "trainer": {
            "num_epochs": 200,
            "grad_norm": 5.0,
            "patience": 20,
            "cuda_device": 0,
            "validation_metric": ["+ud_LAS", "+ner_f1-measure-overall", "+pos_accuracy"],
            "callbacks": [
                {
                    "type": "tensorboard",
                    "tensorboard_writer": {
                        "should_log_learning_rate": true,
                        "should_log_parameter_statistics": true,
                    },
                },
            ],
            "optimizer": representation["optimizer"],
            "learning_rate_scheduler": representation["lr_scheduler"],
        },
    },
    build_mtl_ud(language, emb_type, params)::
    local batch_size = if emb_type == "fasttext" then ud_batch_sizes[language]["fasttext"]
        else ud_batch_sizes[language][emb_type][params[0]];
    local lang_ud = languages[language]["ud"];
    local train_size_ud = lang_ud["train_size"];
    local representation_builder = representations[emb_type];
    local representation = representation_builder.build(language, batch_size, train_size_ud, params);
    if (lang_ud == null || representation == null) then null else {
        "numpy_seed": common["numpy_seed"],
        "pytorch_seed": common["pytorch_seed"],
        "random_seed": common["random_seed"],
        "dataset_reader": {
            "type": "multitask",
            "readers": {
                "ud": {
                    "type": "universal_dependencies",
                    "token_indexers": representation["indexers"],
                },
            },
        },
        "train_data_path": {
            "ud": lang_ud["train_data_path"],
        },
        "validation_data_path": {
            "ud": lang_ud["validation_data_path"],
        },
        "model": {
            "type": "multitask",
            "arg_name_mapping": {
                "backbone": {
                    "tokens": "text",
                    "words": "text",
                },
            },
            "backbone": {
                "type": "embedder_and_mask",
                "text_field_embedder": representation["embedders"],
            },
            "heads": {
                "ud": {
                    "type": "biaffine_parser",
                    "arc_representation_dim": 100,
                    "tag_representation_dim": 100,
                    // NEW!
                    "use_mst_decoding_for_validation": true,
                    "dropout": 0.3,
                    "input_dropout": 0.3,
                    "encoder": representation["encoders"]["ud"],
                    "initializer": {
                        "regexes": [
                            [".*projection.*weight", {"type": "xavier_uniform"}],
                            [".*projection.*bias", {"type": "zero"}],
                            [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
                            [".*tag_bilinear.*bias", {"type": "zero"}],
                            [".*weight_ih.*", {"type": "xavier_uniform"}],
                            [".*weight_hh.*", {"type": "orthogonal"}],
                            [".*bias_ih.*", {"type": "zero"}],
                            [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
                        ]
                    },
                },
            },
        },
        "data_loader": {
            "type": "multitask",
            "shuffle": true,
            "scheduler": {
                "type": "unbalanced_homogeneous_roundrobin",
                "batch_size": batch_size,
                "dataset_sizes": {
                    "ud": train_size_ud,
                },
            },
        },
        "validation_data_loader": {
            "type": "multitask",
            "shuffle": true,
            "scheduler": {
                "type": "homogeneous_roundrobin",
                "batch_size": batch_size,
            },
        },
        "trainer": {
            "num_epochs": 200,
            "grad_norm": 5.0,
            "patience": 20,
            "cuda_device": 0,
            "validation_metric": ["+ud_LAS"],
            "callbacks": [
                {
                    "type": "tensorboard",
                    "tensorboard_writer": {
                        "should_log_learning_rate": true,
                        "should_log_parameter_statistics": true,
                    },
                },
            ],
            "optimizer": representation["optimizer"],
            "learning_rate_scheduler": representation["lr_scheduler"],
        },
    },
    build_mtl_ner(language, emb_type, params)::
    local batch_size = if emb_type == "fasttext" then ner_batch_sizes[language]["fasttext"]
        else ner_batch_sizes[language][emb_type][params[0]];
    local lang_ner = languages[language]["ner"];
    local train_size_ner = lang_ner["train_size"];
    local representation_builder = representations[emb_type];
    local representation = representation_builder.build(language, batch_size, train_size_ner, params);
    if (lang_ner == null || representation == null) then null else {
        "numpy_seed": common["numpy_seed"],
        "pytorch_seed": common["pytorch_seed"],
        "random_seed": common["random_seed"],
        "dataset_reader": {
            "type": "multitask",
            "readers": {
                "ner": {
                    "type": "wikiann",
                    "token_indexers": representation["indexers"],
                },
            },
        },
        "train_data_path": {
            "ner": lang_ner["train_data_path"],
        },
        "validation_data_path": {
            "ner": lang_ner["validation_data_path"],
        },
        "model": {
            "type": "multitask",
            "arg_name_mapping": {
                "backbone": {
                    "tokens": "text",
                    "words": "text",
                },
            },
            "backbone": {
                "type": "embedder_and_mask",
                "text_field_embedder": representation["embedders"],
            },
            "heads": {
                "ner": {
                    "type": "crf_tagger",
                    "encoder": representation["encoders"]["ner"],
                    "include_start_end_transitions": false,
                    // following SciBERT
                    "dropout": 0.1,
                    "calculate_span_f1": true,
                    "constrain_crf_decoding": true,
                    "label_encoding": "BIO",
                },
            },
        },
        "data_loader": {
            "type": "multitask",
            "shuffle": true,
            "scheduler": {
                "type": "unbalanced_homogeneous_roundrobin",
                "batch_size": batch_size,
                "dataset_sizes": {
                    "ner": train_size_ner,
                },
            },
        },
        "validation_data_loader": {
            "type": "multitask",
            "shuffle": true,
            "scheduler": {
                "type": "homogeneous_roundrobin",
                "batch_size": batch_size,
            },
        },
        "trainer": {
            "num_epochs": 200,
            "grad_norm": 5.0,
            "patience": 20,
            "cuda_device": 0,
            "validation_metric": ["+ner_f1-measure-overall"],
            "callbacks": [
                {
                    "type": "tensorboard",
                    "tensorboard_writer": {
                        "should_log_learning_rate": true,
                        "should_log_parameter_statistics": true,
                    },
                },
            ],
            "optimizer": representation["optimizer"],
            "learning_rate_scheduler": representation["lr_scheduler"],
        },
    },
    build_mtl_pos(language, emb_type, params)::
    local batch_size = if emb_type == "fasttext" then pos_batch_sizes[language]["fasttext"]
        else pos_batch_sizes[language][emb_type][params[0]];
    local lang_ud = languages[language]["ud"];
    local train_size_ud = lang_ud["train_size"];
    local representation_builder = representations[emb_type];
    local representation = representation_builder.build(language, batch_size, train_size_ud, params);
    if (lang_ud == null || representation == null) then null else {
        "numpy_seed": common["numpy_seed"],
        "pytorch_seed": common["pytorch_seed"],
        "random_seed": common["random_seed"],
        "dataset_reader": {
            "type": "multitask",
            "readers": {
                "pos": {
                    "type": "universal_dependencies",
                    "token_indexers": representation["indexers"],
                    // NOTE: this is supposed to make things harder
                    // but Belarusian has some sentences without xpos
                    "use_language_specific_pos": language != "be",
                },
            },
        },
        "train_data_path": {
            "pos": lang_ud["train_data_path"],
        },
        "validation_data_path": {
            "pos": lang_ud["validation_data_path"],
        },
        "model": {
            "type": "multitask",
            "arg_name_mapping": {
                "backbone": {
                    "tokens": "text",
                    "words": "text",
                },
            },
            "backbone": {
                "type": "embedder_and_mask",
                "text_field_embedder": representation["embedders"],
            },
            "heads": {
                "pos": {
                    "type": "linear_tagger",
                    "encoder": representation["encoders"]["pos"],
                    "dropout": 0.4,
                    "initializer": {
                        "regexes": [
                            [".*projection.*weight", {"type": "xavier_uniform"}],
                            [".*projection.*bias", {"type": "zero"}],
                            [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
                            [".*tag_bilinear.*bias", {"type": "zero"}],
                            [".*weight_ih.*", {"type": "xavier_uniform"}],
                            [".*weight_hh.*", {"type": "orthogonal"}],
                            [".*bias_ih.*", {"type": "zero"}],
                            [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
                        ]
                    },
                },
            },
        },
        "data_loader": {
            "type": "multitask",
            "shuffle": true,
            "scheduler": {
                "type": "unbalanced_homogeneous_roundrobin",
                "batch_size": batch_size,
                "dataset_sizes": {
                    "pos": train_size_ud,
                },
            },
        },
        "validation_data_loader": {
            "type": "multitask",
            "shuffle": true,
            "scheduler": {
                "type": "homogeneous_roundrobin",
                "batch_size": batch_size,
            },
        },
        "trainer": {
            "num_epochs": 200,
            "grad_norm": 5.0,
            "patience": 20,
            "cuda_device": 0,
            "validation_metric": ["+pos_accuracy"],
            "callbacks": [
                {
                    "type": "tensorboard",
                    "tensorboard_writer": {
                        "should_log_learning_rate": true,
                        "should_log_parameter_statistics": true,
                    },
                },
            ],
            "optimizer": representation["optimizer"],
            "learning_rate_scheduler": representation["lr_scheduler"],
        },
    },
}