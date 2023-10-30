local languages = import "lang.libsonnet";
// local trained_path = "/m-pinotHD/echau18/lr-ssmba/bert_outputs";
// local data_root = "/m-pinotHD/echau18/lr-ssmba/data";
local trained_path = std.extVar("MTL_BERT_OUTPUTS");
local data_root = std.extVar("MTL_DATA"); 
local mbert_path = data_root + "/mbert";
local pretrained_path = std.extVar("PRETRAIN_PATH");

// this is just for reference; we actually store the best models
// as epoch_best
local best_berts = {
    "vambert": {
        "be": 2,
        "bg": 4,
        "ga": 5,
        "mhr": 3,
        "mt": 4,
        "ug": 5,
        "ur": 6,
        "vi": 2,
        "wo": 5,
    },
    "roberta": {
        "be": 20,
        "bg": 20,
        "ga": 20,
        "mhr": 20,
        "mt": 20,
        "ug": 20,
        "ur": 20,
        "vi": 20,
        "wo": 20,
    },
};

{
    "fasttext": {
        build(language, batch_size, train_size, params)::
        local dim = 100;
        local path = data_root + "/" + language + 
            "/unlabeled/bert_cleaned/fasttext.vec";
        {
            "indexers": {
                "tokens": {
                    "type": "single_id",
                },
            },
            "embedders": {
                "token_embedders": {
                    "tokens": {
                        "type": "embedding",
                        "embedding_dim": dim,
                        "pretrained_file": path,
                        "sparse": true,
                        "trainable": true,
                    },
                },
            },
            "encoders": {
                "ner": {
                    "type": "lstm",
                    "input_size": dim,
                    "hidden_size": 200,
                    "num_layers": 2,
                    "dropout": 0.5,
                    "bidirectional": true,
                },
                "ud": {
                    "type": "stacked_bidirectional_lstm",
                    "hidden_size": 400,
                    "input_size": dim,
                    "num_layers": 3,
                    "recurrent_dropout_probability": 0.3,
                    "use_highway": true,
                },
                "pos": {
                    "type": "lstm",
                    "input_size": dim,
                    "hidden_size": 200,
                    "num_layers": 2,
                    "dropout": 0.5,
                    "bidirectional": true,
                },
            },
            "optimizer": {
                "type": "dense_sparse_adam",
                "lr": 1e-3,
                "betas": [0.9, 0.999],
            },
            "lr_scheduler": null,
        }
    },
    "bert": {
        build(language, batch_size, train_size, params)::
        // either mbert, vambert, or roberta
        local model_name = params[0];
        local epoch = if model_name == "mbert" then -1 else params[1];
        // local epoch = if model_name == "mbert" then -1 
        //     else if params[1] == "best" then best_berts[model_name][language]
        //     else params[1];
        local dim = 768;
        local is_bert = (model_name != "roberta");
        local model_path = pretrained_path;
//            else if model_name == "vambert" then pretrained_path
//            else if model_name == "ssmba_vambert" then trained_path + "/" + language + "/tva_base_ssmba_tva_base/epoch_" + epoch
//            else if model_name == "lapt" then trained_path + "/" + language + "/" + pretrained_path + "/epoch_" + epoch
//            else trained_path + "/" + language + "/roberta/epoch_" + epoch;
        local max_length = 512;
        {
            "indexers": {
                "transformer": {
                    "type": "pretrained_transformer_mismatched",
                    "model_name": model_path,
                    "max_length": max_length,
                    "tokenizer_kwargs": {
                        // these should be no-op for RoBERTa?
                        "do_lower_case": false,
                        "tokenize_chinese_chars": true,
                        "strip_accents": false,
                        "clean_text": true,
                    },
                },
            },
            "embedders": {
                "token_embedders": {
                    "transformer": {
                        "type": "pretrained_transformer_mismatched_with_dropout",
                        "model_name": model_path,
                        "max_length": max_length,
                        "last_layer_only": false,
                        "train_parameters": true,
                        "layer_dropout": 0.1,
                        "tokenizer_kwargs": {
                            "do_lower_case": false,
                            "tokenize_chinese_chars": true,
                            "strip_accents": false,
                            "clean_text": true,
                        },
                    },
                },
            },
            "encoders": {
                "ner": {
                    "type": "pass_through",
                    "input_dim": dim,
                },
                "ud": {
                    "type": "pass_through",
                    "input_dim": dim,
                },
                "pos": {
                    "type": "pass_through",
                    "input_dim": dim,
                },
            },
            "optimizer": {
                "type": "huggingface_adamw",
                // faster LR; slower one computed via decay_factor in the
                // scheduler
                "lr": 1e-3,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
                // HF says that BERT doesn't use this
                "correct_bias": !is_bert,
                // eps kept default
                "parameter_groups": [
                    [
                        [
                            "text_field_embedder.*transformer_model.embeddings.*_embeddings.*",
                            "text_field_embedder.*transformer_model.encoder.*.(key|query|value|dense).weight",
                        ],
                        {}
                    ],
                    [
                        // adapted from SciBERT
                        [
                            "text_field_embedder.*transformer_model.embeddings.LayerNorm.*",
                            "text_field_embedder.*transformer_model.encoder.*.output.LayerNorm.*",
                            "text_field_embedder.*transformer_model.encoder.*.(key|query|value|dense).bias",
                            "text_field_embedder.*transformer_model.pooler.dense.bias",
                        ], 
                        {"weight_decay": 0.0}
                    ],
                    [
                        [
                            "text_field_embedder.*._scalar_mix.*",
                            "text_field_embedder.*transformer_model.pooler.dense.weight",
                            "_head_sentinel",
                            "head_arc_feedforward._linear_layers.*.weight",
                            "child_arc_feedforward._linear_layers.*.weight",
                            "head_tag_feedforward._linear_layers.*.weight",
                            "child_tag_feedforward._linear_layers.*.weight",
                            "arc_attention._weight_matrix",
                            "tag_bilinear.weight",
                            "tag_projection_layer._module.weight",
                            "crf",
                            "linear.weight",
                            "tagger_linear.weight",
                        ],
                        {}
                    ],
                    [
                        [
                            "head_arc_feedforward._linear_layers.*.bias",
                            "child_arc_feedforward._linear_layers.*.bias",
                            "head_tag_feedforward._linear_layers.*.bias",
                            "child_tag_feedforward._linear_layers.*.bias",
                            "arc_attention._bias",
                            "tag_bilinear.bias",
                            "tag_projection_layer._module.bias",
                            "linear.bias",
                            "tagger_linear.bias",
                        ],
                        {"weight_decay": 0.0},
                    ],
                ],
            },
            "lr_scheduler": {
                "type": "ulmfit_sqrt",
                "model_size": 1, // UDify did this so...?
                "affected_group_count": 2,
                // language-specific one epoch
                "warmup_steps": std.ceil(train_size / batch_size),
                // language-specific one epoch, by suggestion of UDify
                // https://github.com/Hyperparticle/udify/issues/6
                "start_step": std.ceil(train_size / batch_size),
                "factor": 5.0, // following UDify
                "gradual_unfreezing": true,
                "discriminative_fine_tuning": true,
                "decay_factor": 0.05, // yields a slow LR of 5e-5
                // steepness kept to 0.5 (sqrt)
            },
        }
    },
}