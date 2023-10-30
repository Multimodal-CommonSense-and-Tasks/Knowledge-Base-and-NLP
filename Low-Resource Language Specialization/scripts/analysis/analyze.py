from vocab_data import get_vocab_data, IS_UNLABELED
from task_data import get_task_data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.plotting as pdp
import scipy.stats as scp

from matplotlib.lines import Line2D

SPLIT_FILTER = "train"
GRAPH_PREFIX = "unlabeledvocab_" if IS_UNLABELED else ""
# TODO: Uyghur is an outlier
GRAPHS_INCLUDE_UYGHUR = True

FINAL_GRAPH_ATTRIBUTES = {
    "type0": "d",
    "type1": "o",
    "type2": "X",
    "cyrillic": "tab:blue",
    "latin": "tab:purple",
    "arabic": "tab:orange",
}

vocab_data = get_vocab_data()
task_data = get_task_data()

for df in ("unlabeled_df", "ud_df", "panx_df"):
    vocab_data[df] = vocab_data[df].loc[
        vocab_data[df]["split"] == SPLIT_FILTER
    ]

transliterated_df_map = {
    "unlabeled_df": ["unlabeled"],
    "ud_df": ["ud", "pos"],
    "panx_df": ["ner"],
}
transliterated_data = {}
transliterated_df = pd.DataFrame({})
for df, tasks in transliterated_df_map.items():
    transliterated_data[df] = vocab_data[df].loc[
        (vocab_data[df].index == "mhrlatin")
        | (vocab_data[df].index == "uglatinnfkc")
        | (vocab_data[df].index == "ug")
        | (vocab_data[df].index == "mhr")
    ]

    for task in tasks:
        task_df = transliterated_data[df].rename(
            index={
                "mhr": f"mhr_{task}",
                "mhrlatin": f"mhrlatin_{task}",
                "ug": f"ug_{task}",
                "uglatinnfkc": f"uglatin_{task}",
            }
        )
        transliterated_df = pd.concat([transliterated_df, task_df])


for df in ("unlabeled_df", "ud_df", "panx_df"):
    vocab_data[df] = vocab_data[df].loc[
        (vocab_data[df].index != "mhrlatin")
        & (vocab_data[df].index != "uglatinnfkc")
    ]

    if not GRAPHS_INCLUDE_UYGHUR:
        vocab_data[df] = vocab_data[df].loc[vocab_data[df].index != "ug"]


def deltify(task_df):
    transpose_df = task_df.T
    # get rid of average row
    transpose_df = transpose_df.loc[transpose_df.index != "avg"]
    transpose_df["tva-mbert"] = transpose_df["tva"] - transpose_df["mbert"]
    transpose_df["tva/mbert"] = transpose_df["tva"] / transpose_df["mbert"]
    transpose_df["tva-lapt"] = transpose_df["tva"] - transpose_df["lapt"]
    transpose_df["tva/lapt"] = transpose_df["tva"] / transpose_df["lapt"]

    return transpose_df


def fertility_ratio(vocab_df):
    fert_df = vocab_df.loc[:, ("tokenizer", "tfertility")]
    roberta_fert = fert_df.loc[fert_df["tokenizer"] == "roberta"]
    roberta_fert = roberta_fert["tfertility"]
    mbert_fert = fert_df.loc[fert_df["tokenizer"] == "mbert"]
    mbert_fert = mbert_fert["tfertility"]
    tva_fert = fert_df.loc[fert_df["tokenizer"] == "tva"]
    tva_fert = tva_fert["tfertility"]

    # want to compare the distances from 1, and having a decrease is better
    # so abs(mbert/roberta - 1) - abs(tva/roberta - 1)

    ratio_df = (mbert_fert / roberta_fert - 1).abs() - (
        tva_fert / roberta_fert - 1
    ).abs()
    return ratio_df


def unk_metrics(vocab_df):
    unk_df = vocab_df.loc[:, ("tokenizer", "tunkprop", "tokwithunkprop")]
    mbert_df = unk_df.loc[unk_df["tokenizer"] == "mbert"].loc[
        :, ("tunkprop", "tokwithunkprop")
    ]
    tva_df = unk_df.loc[unk_df["tokenizer"] == "tva"].loc[
        :, ("tunkprop", "tokwithunkprop")
    ]
    abs_df = mbert_df - tva_df
    abs_df.columns = ["mbert-tva:tunkprop", "mbert-tva:tokwithunkprop"]

    # this doesn't work because often one or more of the values is a 0
    # rel_df = mbert_df / tva_df

    return abs_df


def unk_rate_computations(vocab_df, unk_metrics):
    unk_df = vocab_df.loc[
        vocab_df["tokenizer"] == "mbert", ("tunkprop", "tokwithunkprop")
    ].rename(
        columns={
            "tunkprop": "up_m",
            "tokwithunkprop": "tup_m",
        }
    )
    tva_unk_df = vocab_df.loc[
        vocab_df["tokenizer"] == "tva", ("tunkprop", "tokwithunkprop")
    ].rename(
        columns={
            "tunkprop": "up_t",
            "tokwithunkprop": "tup_t",
        }
    )
    unk_df = pd.concat([unk_df, tva_unk_df, unk_metrics], axis=1)
    results = {}
    for attr, attr_set in task_data["attributes"].items():
        subset_df = unk_df.loc[unk_df.index.isin(attr_set)]
        results[attr] = subset_df.mean()

    # original mbert values give a sense of how bad things are originally,
    # and the delta values aggregated by class show improvement potential
    # (except not really, because most of the time TVA drives to zero anyways)
    results = pd.DataFrame(results).T
    return results


def translit_unk_metrics(vocab_df):
    mbert_df = vocab_df.loc[
        vocab_df["tokenizer"] == "mbert",
        ("tunkprop", "tokwithunkprop"),
    ].rename(
        columns={
            "tunkprop": "tunkprop_mbert",
            "tokwithunkprop": "tokwithunkprop_mbert",
        }
    )
    tva_df = vocab_df.loc[
        vocab_df["tokenizer"] == "tva",
        ("tunkprop", "tokwithunkprop"),
    ].rename(
        columns={
            "tunkprop": "tunkprop_tva",
            "tokwithunkprop": "tokwithunkprop_tva",
        }
    )
    return mbert_df.join(tva_df)


def delta_computations(delta_df):
    delta_df = delta_df.loc[
        :, ("tva-mbert", "tva/mbert", "tva-lapt", "tva/lapt")
    ]
    results = {}
    for attr, attr_set in task_data["attributes"].items():
        subset_df = delta_df.loc[delta_df.index.isin(attr_set)]
        results[attr] = subset_df.mean()

    results = pd.DataFrame(results).T
    return results


ud_fert = fertility_ratio(vocab_data["ud_df"])
panx_fert = fertility_ratio(vocab_data["panx_df"])

ud_unk = unk_metrics(vocab_data["ud_df"])
panx_unk = unk_metrics(vocab_data["panx_df"])
unlabeled_unk = unk_metrics(vocab_data["unlabeled_df"])

ud_unkrate = unk_rate_computations(vocab_data["ud_df"], ud_unk)
unlabeled_unkrate = unk_rate_computations(
    vocab_data["unlabeled_df"], unlabeled_unk
)
panx_unkrate = unk_rate_computations(vocab_data["panx_df"], panx_unk)

ud_delt = deltify(task_data["ud"])
ner_delt = deltify(task_data["ner"])
pos_delt = deltify(task_data["pos"])

ud_delt_metrics = delta_computations(ud_delt)
ner_delt_metrics = delta_computations(ner_delt)
pos_delt_metrics = delta_computations(pos_delt)

translit_fert = fertility_ratio(transliterated_df)
translit_unk = translit_unk_metrics(transliterated_df)
translit_delt = deltify(task_data["translit"])
# unkrate?
# delt_metrics?

translit_unk_comparison = (
    translit_delt.join(translit_unk)
    .loc[
        :,
        (
            "lapt",
            "tunkprop_mbert",
            "tokwithunkprop_mbert",
            "tva",
            "tunkprop_tva",
            "tokwithunkprop_tva",
        ),
    ]
    .T
)


def plot_table(df, filename):
    df = df.round(5)
    plt.cla()
    plt.clf()
    plt.close()
    plt.figure()
    ax = plt.gca()
    ax.set_axis_off()
    pdp.table(ax, df, loc="center")
    plt.savefig(filename, bbox_inches="tight", dpi=600)


plot_table(ud_unkrate, f"./{GRAPH_PREFIX}ud_unkrate.png")
plot_table(unlabeled_unkrate, f"./{GRAPH_PREFIX}unlabeled_unkrate.png")
plot_table(panx_unkrate, f"./{GRAPH_PREFIX}panx_unkrate.png")

plot_table(ud_delt_metrics, f"./{GRAPH_PREFIX}ud_delt_metrics.png")
plot_table(pos_delt_metrics, f"./{GRAPH_PREFIX}pos_delt_metrics.png")
plot_table(ner_delt_metrics, f"./{GRAPH_PREFIX}ner_delt_metrics.png")

plot_table(
    translit_unk_comparison, f"./{GRAPH_PREFIX}translit_unk_comparison.png"
)


##### BEGIN DATA ANALYSIS #####

# Fertility ratios
# one scatter plot per {ud, ner, pos} of:
# fertility ratio vs. {tva/mbert, tva/lapt}
for task, vocab, data in [
    ("ud", ud_fert, ud_delt),
    ("pos", ud_fert, pos_delt),
    ("ner", panx_fert, ner_delt),
]:
    plt.cla()
    plt.clf()
    plt.close()

    plt.figure()
    join_table = data.join(vocab)

    if not GRAPHS_INCLUDE_UYGHUR:
        join_table = join_table.loc[join_table.index != "ug"]

    rho, pval = scp.spearmanr(join_table["tfertility"], join_table["tva/lapt"])

    # ax = join_table.plot(
    #     x="tfertility",
    #     y="tva/mbert",
    #     kind="scatter",
    #     label="TVA over mBERT",
    #     color="blue",
    # )

    join_table.plot(
        x="tfertility",
        y="tva/lapt",
        kind="scatter",
        label="TVA over LAPT",
        color="green",
        # ax=ax,
    )
    plt.xlabel("Fertility alignment")
    plt.ylabel("TVA relative improvement")
    plt.title(f"Fertility (Spearman's rho = {rho:.4f}, p = {pval:.4f})")
    plt.savefig(f"./{GRAPH_PREFIX}fertility_{task}.png")

# Unk proportion
# one scatter plot per {ud, ner, pos} of
# mbert-tva:tunkprop vs. {tva-mbert, tva-lapt}
# (or maybe one joint scatter)
for task, vocab, data in [
    ("ud", ud_unk, ud_delt),
    ("pos", ud_unk, pos_delt),
    ("ner", panx_unk, ner_delt),
]:
    plt.cla()
    plt.clf()
    plt.close()

    plt.figure()
    join_table = data.join(vocab)

    if not GRAPHS_INCLUDE_UYGHUR:
        join_table = join_table.loc[join_table.index != "ug"]

    rho, pval = scp.spearmanr(
        join_table["mbert-tva:tunkprop"], join_table["tva-lapt"]
    )

    # ax = join_table.plot(
    #     x="mbert-tva:tunkprop",
    #     y="tva-mbert",
    #     kind="scatter",
    #     label="TVA over mBERT",
    #     color="blue",
    # )
    join_table.plot(
        x="mbert-tva:tunkprop",
        y="tva-lapt",
        kind="scatter",
        label="TVA over LAPT",
        color="green",
        # ax=ax,
    )
    plt.xlabel("Decrease in proportion of subwords with unknown")
    plt.ylabel("TVA absolute improvement")
    plt.title(
        f"Prop. of subwords with unknown (Spearman's rho = {rho:.4f}, p={pval:.4f})"
    )
    plt.savefig(f"./{GRAPH_PREFIX}unkprop_{task}.png")

final_graph_dfs = {}

# Tokens with unk
# one scatter plot per {ud, ner, pos} of
# mbert-tva:tokwithunkprop vs. {tva-mbert, tva-lapt}
# (or maybe one joint scatter)
for task, vocab, data in [
    ("ud", ud_unk, ud_delt),
    ("pos", ud_unk, pos_delt),
    ("ner", panx_unk, ner_delt),
]:
    plt.cla()
    plt.clf()
    plt.close()

    plt.figure()
    join_table = data.join(vocab)

    if not GRAPHS_INCLUDE_UYGHUR:
        join_table = join_table.loc[join_table.index != "ug"]

    rho, pval = scp.spearmanr(
        join_table["mbert-tva:tokwithunkprop"], join_table["tva-lapt"]
    )

    # ax = join_table.plot(
    #     x="mbert-tva:tokwithunkprop",
    #     y="tva-mbert",
    #     kind="scatter",
    #     label="TVA over mBERT",
    #     color="blue",
    # )
    join_table.plot(
        x="mbert-tva:tokwithunkprop",
        y="tva-lapt",
        kind="scatter",
        label="TVA over LAPT",
        color="green",
        # ax=ax,
    )
    plt.xlabel("Decrease in proportion of words yielding unknown subword")
    plt.ylabel("TVA absolute improvement")
    plt.title(
        f"Prop. of words yielding UNK (Spearman's rho = {rho:.4f}, p={pval:.4f})"
    )
    plt.savefig(f"./{GRAPH_PREFIX}tokwithunkprop_{task}.png")
    final_graph_dfs[task] = join_table

# Final plots
for task, metric, dataset, ticks in [
    (
        "ud",
        "UD LAS",
        "Unlabeled" if IS_UNLABELED else "UD",
        np.arange(-2, 20, 2),
    ),
    (
        "pos",
        "POS Accuracy",
        "Unlabeled" if IS_UNLABELED else "UD",
        np.arange(-2, 14, 2),
    ),
    (
        "ner",
        "NER Macro F1",
        "Unlabeled" if IS_UNLABELED else "WikiAnn",
        np.arange(-2, 14, 2),
    ),
]:
    join_table = final_graph_dfs[task]

    plt.cla()
    plt.clf()
    plt.close()
    plt.figure()
    plt.rcParams["font.size"] = 14
    plt.rcParams["legend.fontsize"] = 16
    # LOBF
    x_series = join_table["mbert-tva:tokwithunkprop"]
    y_series = join_table["tva-lapt"]
    x_series, y_series = zip(*sorted(zip(x_series, y_series)))
    x_series = np.array(x_series)
    y_series = np.array(y_series)
    m, b = np.polyfit(x_series, y_series, 1)
    plt.plot(
        x_series, m * x_series + b, ":", c="tab:gray", linewidth=5, zorder=0
    )

    final_plot_data = {}
    for index, row in join_table.iterrows():
        x_val = row["mbert-tva:tokwithunkprop"]
        y_val = row["tva-lapt"]
        color = None
        shape = None
        for script in ("latin", "cyrillic", "arabic"):
            if index in task_data["attributes"][script]:
                color = FINAL_GRAPH_ATTRIBUTES[script]
                break
        assert color != None
        for langtype in ("type0", "type1", "type2"):
            if index in task_data["attributes"][langtype]:
                shape = FINAL_GRAPH_ATTRIBUTES[langtype]
                break
        assert shape != None
        final_plot_data[index] = (x_val, y_val, color, shape)
    plt.xlabel(
        f"Change in UNK Token %, {dataset} Data (mBERT - TVA)", fontsize=15
    )
    plt.ylabel(f"Change in {metric} (TVA - LAPT)", fontsize=15)
    plt.yticks(ticks)
    if IS_UNLABELED:
        plt.xscale("log")
    else:
        plt.xscale("symlog", linthresh=1e-5)
    plt.grid(True)

    for x, y, color, shape in final_plot_data.values():
        plt.scatter(x, y, c=color, marker=shape, s=180, zorder=100)

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="w",
            marker="d",
            markerfacecolor="tab:gray",
            markersize=17,
            label="Type 0",
        ),
        Line2D(
            [0],
            [0],
            color="w",
            marker="o",
            markerfacecolor="tab:gray",
            markersize=17,
            label="Type 1",
        ),
        Line2D(
            [0],
            [0],
            color="w",
            marker="X",
            markerfacecolor="tab:gray",
            markersize=17,
            label="Type 2",
        ),
        Line2D(
            [0],
            [0],
            color="tab:purple",
            lw=4,
            label="Latin",
        ),
        Line2D(
            [0],
            [0],
            color="tab:blue",
            lw=4,
            label="Cyrillic",
        ),
        Line2D(
            [0],
            [0],
            color="tab:orange",
            lw=4,
            label="Arabic",
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper left")

    plt.subplots_adjust(top=0.95, bottom=0.15, right=0.95)
    plt.savefig(f"./{GRAPH_PREFIX}FINAL_{task}.png")
