import pickle

import torch
from typing import Dict
def vis(embed_datas: Dict[str, list]):
    from sklearn.decomposition import PCA
    fitter = PCA(n_components=2)

    datas = fitter.fit_transform(torch.cat(sum(embed_datas.values(), [])).detach().cpu().numpy())

    import random
    import plotly.graph_objects as graph_obj
    fig = graph_obj.Figure()

    prev_end = 0
    for embed_name, dat in embed_datas.items():
        new_end = prev_end + len(dat)
        x, y = zip(*datas[prev_end:new_end])
        if 'hrl' in embed_name:
            rand_color = f"rgb(31, 119, 180)"
        else:
            rand_color = f"rgb(255, 127, 14)"
        # rand_r = random.randint(0, 255)
        # rand_g = random.randint(0, 255)
        # rand_b = random.randint(0, 255)
        # rand_color = f'rgb({rand_r},{rand_g},{rand_b})'

        w = embed_name
        fig.add_trace(
            graph_obj.Scatter(mode="markers", x=x, y=y,
                              marker_line_color=rand_color, marker_color=rand_color,
                              legendgroup=w, name=w,
                              marker_line_width=2, marker_size=3))
        prev_end = new_end

    return fig, datas

prefix = "mlmed"
embed_datas = {
    "baseline": pickle.load(open("bert/scripts/cross_script_align/mt_script_mix/visualize/tr/va_mlm/orig_embs", 'rb'))[f'12'],
    "cs+cl": pickle.load(open("bert/scripts/cross_script_align/mt_script_mix/visualize/tr/cs_cl_mlm_reset/orig_embs", 'rb'))[f'12'],
    "baseline_hrl": pickle.load(open("bert/scripts/cross_script_align/mt_script_mix/visualize/tr/va_mlm/tl_embs", 'rb'))[f'12'],
    "cs+cl_hrl": pickle.load(open("bert/scripts/cross_script_align/mt_script_mix/visualize/tr/cs_cl_mlm_reset/tl_embs", 'rb'))[f'12'],
}

prefix = "nomlm"
embed_datas = {
    "nomlm_baseline": pickle.load(open("bert/scripts/cross_script_align/mt_script_mix/visualize/tr/va/orig_embs", 'rb'))[f'12'],
    "nomlm_cs+cl": pickle.load(open("bert/scripts/cross_script_align/mt_script_mix/visualize/tr/cs_cl/orig_embs", 'rb'))[f'12'],
    "nomlm_baseline_hrl": pickle.load(open("bert/scripts/cross_script_align/mt_script_mix/visualize/tr/va/tl_embs", 'rb'))[f'12'],
    "nomlm_cs+cl_hrl": pickle.load(open("bert/scripts/cross_script_align/mt_script_mix/visualize/tr/cs_cl/tl_embs", 'rb'))[f'12'],
}

fig, _ = vis(embed_datas)
fig.write_html(f"bert/scripts/cross_script_align/mt_script_mix/visualize/tr/{prefix}.html")
