import matplotlib.pyplot as plt

from summarize_mtl_results import results as mtl_results, langs
from summarize_probing_results import (
    results as probe_pos,
    controls as probe_controls,
    sensitivities as probe_sensitivities,
)

probe_values = (probe_pos, probe_controls, probe_sensitivities)

ud = mtl_results["ud"]
ner = mtl_results["ner"]

ud_pairs = [("tva_best", "tva_best/mtlud"), ("tva_best_mtl", "tva_best/mtl")]
ner_pairs = [("tva_best", "tva_best/mtlner"), ("tva_best_mtl", "tva_best/mtl")]

fig, axs = plt.subplots(len(probe_values), 2)
fig.suptitle(
    "UD (left) and NER (right) performance vs. POS (top), Control (mid), Sensitivity (bottom)"
)

for i, probe_val in enumerate(probe_values):
    # UD
    ax = axs[i, 0]
    for ud_key, probe_key in ud_pairs:
        pairs = [
            (ud[lang][ud_key], probe_val[lang][probe_key]) for lang in langs
        ]
        pairs = [pair for pair in pairs if pair[0] != 0.0 and pair[1] != 0.0]
        pairs.sort(key=lambda x: x[0])
        x, y = zip(*pairs)
        ax.plot(x, y, label=f"UD: {ud_key} vs. {probe_key}")
    # ax.legend()

    # NER
    ax = axs[i, 1]
    for ner_key, probe_key in ner_pairs:
        pairs = [
            (ner[lang][ner_key], probe_val[lang][probe_key]) for lang in langs
        ]
        pairs = [pair for pair in pairs if pair[0] != 0.0 and pair[1] != 0.0]
        pairs.sort(key=lambda x: x[0])
        x, y = zip(*pairs)
        ax.plot(x, y, label=f"NER: {ner_key} vs. {probe_key}")
    # ax.legend()

plt.show()