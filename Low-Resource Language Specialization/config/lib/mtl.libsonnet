local languages = import "lang.libsonnet";
local representations = import "rep.libsonnet";
local common = import "common.libsonnet";
local batch_sizes = import "batch_sizes.libsonnet";
local archive_root = std.extVar("MTL_ALLENNLP_OUTPUTS");
local fold = std.extVar("SEED_SET");
{
    build_mtl(language, emb_type, params, homogeneous)::
    local batch_size = if emb_type == "fasttext" then batch_sizes["mtl"][language]["fasttext"]
        else batch_sizes["mtl"][language][emb_type][params[0]];
    local lang_ud = languages[language]["ud"];
    local lang_ner = languages[language]["ner"];
    local train_size_ud = lang_ud["train_size"];
    local train_size_ner = if language != "wo" then lang_ner["train_size"] else 0;
    local rounded_up_train_size_ud = batch_size * std.ceil(train_size_ud / batch_size);
    local rounded_up_train_size_ner = batch_size * std.ceil(train_size_ner / batch_size);
    # if this is homogeneous, then it's 2 * # ud batches + # ner batches
    # if it is not, then it's 3 * # of batches in the bigger dataset
    # # of batches for a given dataset is 
    local train_size = if homogeneous
        then 2 * rounded_up_train_size_ud + rounded_up_train_size_ner
        else (if language != "wo" then 3 else 2) * std.max(rounded_up_train_size_ud, rounded_up_train_size_ner);
    local representation_builder = representations[emb_type];
    local representation = representation_builder.build(language, batch_size, train_size, params);
    if (lang_ud == null || (lang_ner == null && language != "wo") || representation == null) then null else {
        "numpy_seed": common["numpy_seed"],
        "pytorch_seed": common["pytorch_seed"],
        "random_seed": common["random_seed"],
        "dataset_reader": {
            "type": "multitask",
            "readers": {
                [if language != "wo" then "ner" else null]: {
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
            [if language != "wo" then "ner" else null]: lang_ner["train_data_path"],
            "ud": lang_ud["train_data_path"],
            "pos": lang_ud["train_data_path"],
        },
        "validation_data_path": {
            [if language != "wo" then "ner" else null]: lang_ner["validation_data_path"],
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
                [if language != "wo" then "ner" else null]: {
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
            "scheduler": if homogeneous then {
                "type": "homogeneous_roundrobin",
                "batch_size": batch_size,
            } else {
                "type": "unbalanced_homogeneous_roundrobin",
                "batch_size": batch_size,
                "dataset_sizes": {
                    "ud": train_size_ud,
                    [if language != "wo" then "ner" else null]: train_size_ner,
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
            "validation_metric": if language != "wo"
                then ["+ud_LAS", "+ner_f1-measure-overall", "+pos_accuracy"]
                else ["+ud_LAS", "+pos_accuracy"],
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
    local batch_size = if emb_type == "fasttext" then batch_sizes["mtlud"][language]["fasttext"]
        else batch_sizes["mtlud"][language][emb_type][params[0]];
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
    local batch_size = if emb_type == "fasttext" then batch_sizes["mtlner"][language]["fasttext"]
        else batch_sizes["mtlner"][language][emb_type][params[0]];
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
    local batch_size = if emb_type == "fasttext" then batch_sizes["mtlpos"][language]["fasttext"]
        else batch_sizes["mtlpos"][language][emb_type][params[0]];
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
    build_mtl_alt(language, emb_type, params, homogeneous)::
    local batch_size = if emb_type == "fasttext" then batch_sizes["mtlalt"][language]["fasttext"]
        else batch_sizes["mtlalt"][language][emb_type][params[0]];
    local lang_ud = languages[language]["ud"];
    local lang_ner = languages[language]["ner"];
    local train_size_ud = lang_ud["train_size"];
    local train_size_ner = if language != "wo" then lang_ner["train_size"] else 0;
    local rounded_up_train_size_ud = batch_size * std.ceil(train_size_ud / batch_size);
    local rounded_up_train_size_ner = batch_size * std.ceil(train_size_ner / batch_size);
    # if this is homogeneous, then it's 2 * # ud batches + # ner batches
    # if it is not, then it's 3 * # of batches in the bigger dataset
    # # of batches for a given dataset is 
    local train_size = if homogeneous
        then 2 * rounded_up_train_size_ud + rounded_up_train_size_ner
        else (if language != "wo" then 3 else 2) * std.max(rounded_up_train_size_ud, rounded_up_train_size_ner);
    local representation_builder = representations[emb_type];
    local representation = representation_builder.build(language, batch_size, train_size, params);
    if (lang_ud == null || (lang_ner == null && language != "wo") || representation == null) then null else {
        "numpy_seed": common["numpy_seed"],
        "pytorch_seed": common["pytorch_seed"],
        "random_seed": common["random_seed"],
        "dataset_reader": {
            "type": "multitask",
            "readers": {
                [if language != "wo" then "ner" else null]: {
                    "type": "wikiann",
                    "token_indexers": representation["indexers"],
                },
                "ud": {
                    "type": "universal_dependencies",
                    "token_indexers": representation["indexers"],
                    // NOTE: this is supposed to make things harder
                    // but Belarusian has some sentences without xpos
                    "use_language_specific_pos": language != "be",
                },
            },
        },
        "train_data_path": {
            [if language != "wo" then "ner" else null]: lang_ner["train_data_path"],
            "ud": lang_ud["train_data_path"],
        },
        "validation_data_path": {
            [if language != "wo" then "ner" else null]: lang_ner["validation_data_path"],
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
                    "type": "tagger_parser",
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
                    "tagger_encoder": representation["encoders"]["pos"],
                    "tagger_dropout": 0.4,
                },
                [if language != "wo" then "ner" else null]: {
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
            "scheduler": if homogeneous then {
                "type": "homogeneous_roundrobin",
                "batch_size": batch_size,
            } else {
                "type": "unbalanced_homogeneous_roundrobin",
                "batch_size": batch_size,
                "dataset_sizes": {
                    "ud": train_size_ud,
                    [if language != "wo" then "ner" else null]: train_size_ner,
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
            "validation_metric": if language != "wo"
                then ["+ud_LAS", "+ner_f1-measure-overall", "+ud_tagger_accuracy"]
                else ["+ud_LAS", "+ud_tagger_accuracy"],
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
    build_mtl_ud_ms(language, emb_type, params, base_config)::
    local is_alt = std.startsWith(base_config, "mtlalt");
    local batch_size = if emb_type == "fasttext" then batch_sizes["mtlud"][language]["fasttext"]
        else batch_sizes["mtlud"][language][emb_type][params[0]];
    local lang_ud = languages[language]["ud"];
    local train_size_ud = lang_ud["train_size"];
    local representation_builder = representations[emb_type];
    local representation = representation_builder.build(language, batch_size, train_size_ud, params);
    local base_model_archive = archive_root + "/" + base_config + "." + fold;
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
                    // NOTE: this is supposed to make things harder
                    // but Belarusian has some sentences without xpos;
                    // we need to keep this here for consistency with
                    // the original model, even though we don't train pos
                    // anymore
                    "use_language_specific_pos": language != "be",
                },
            },
        },
        "vocabulary": {
            "type": "extend",
            "directory": base_model_archive + "/vocabulary",
        },
        "train_data_path": {
            "ud": lang_ud["train_data_path"],
        },
        "validation_data_path": {
            "ud": lang_ud["validation_data_path"],
        },
        "model": {
            "type": "finetuned_multitask_from_archive",
            "archive_file": base_model_archive,
            "head_name": "ud",
            "head_overrides": if is_alt then {
                "tagger_weight": 0.0,
            } else {},
        },
        "data_loader": {
            "type": "multitask",
            "shuffle": true,
            "scheduler": {
                // balancing here doesn't matter since it's a single task
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
    build_mtl_ner_ms(language, emb_type, params, base_config)::
    local batch_size = if emb_type == "fasttext" then batch_sizes["mtlner"][language]["fasttext"]
        else batch_sizes["mtlner"][language][emb_type][params[0]];
    local lang_ner = languages[language]["ner"];
    local train_size_ner = lang_ner["train_size"];
    local representation_builder = representations[emb_type];
    local representation = representation_builder.build(language, batch_size, train_size_ner, params);
    local base_model_archive = archive_root + "/" + base_config + "." + fold;
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
        "vocabulary": {
            "type": "extend",
            "directory": base_model_archive + "/vocabulary",
        },
        "train_data_path": {
            "ner": lang_ner["train_data_path"],
        },
        "validation_data_path": {
            "ner": lang_ner["validation_data_path"],
        },
        "model": {
            "type": "finetuned_multitask_from_archive",
            "archive_file": base_model_archive,
            "head_name": "ner",
            "head_overrides": {},
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
    build_mtl_pos_ms(language, emb_type, params, base_config)::
    local is_alt = std.startsWith(base_config, "mtlalt");
    local batch_size = if emb_type == "fasttext" then batch_sizes["mspos"][language]["fasttext"]
        else batch_sizes["mspos"][language][emb_type][params[0]];
    local lang_ud = languages[language]["ud"];
    local train_size_ud = lang_ud["train_size"];
    local representation_builder = representations[emb_type];
    local representation = representation_builder.build(language, batch_size, train_size_ud, params);
    local base_model_archive = archive_root + "/" + base_config + "." + fold;
    local data_key = if is_alt then "ud" else "pos";
    if (lang_ud == null || representation == null) then null else {
        "numpy_seed": common["numpy_seed"],
        "pytorch_seed": common["pytorch_seed"],
        "random_seed": common["random_seed"],
        "dataset_reader": {
            "type": "multitask",
            "readers": {
                [data_key]: {
                    "type": "universal_dependencies",
                    "token_indexers": representation["indexers"],
                    // NOTE: this is supposed to make things harder
                    // but Belarusian has some sentences without xpos
                    "use_language_specific_pos": language != "be",
                },
            },
        },
        "vocabulary": {
            "type": "extend",
            "directory": base_model_archive + "/vocabulary",
        },
        "train_data_path": {
            [data_key]: lang_ud["train_data_path"],
        },
        "validation_data_path": {
            [data_key]: lang_ud["validation_data_path"],
        },
        "model": {
            "type": "finetuned_multitask_from_archive",
            "archive_file": base_model_archive,
            "head_name": if is_alt then "ud" else "pos",
            "head_overrides": if is_alt then {
                "parser_weight": 0.0,
            } else {},
        },
        "data_loader": {
            "type": "multitask",
            "shuffle": true,
            "scheduler": {
                "type": "unbalanced_homogeneous_roundrobin",
                "batch_size": batch_size,
                "dataset_sizes": {
                    [data_key]: train_size_ud,
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
            "validation_metric": if is_alt then ["+ud_tagger_accuracy"]
                else ["+pos_accuracy"],
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
