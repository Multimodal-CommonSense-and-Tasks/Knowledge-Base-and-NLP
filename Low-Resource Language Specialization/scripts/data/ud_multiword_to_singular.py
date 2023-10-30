"""
The Wolof treebank makes use of multiword expressions, which we can't predict.
This causes the LAS to be artificially deflated.  If we ignore multiword
expressions/treat them as singulars, we get similar results to past work for
our baselines.
"""

wolof_path = "/m-pinotHD/echau18/lr-ssmba/data/wo/ud/test.conllu.orig"
new_wolof_path = "/m-pinotHD/echau18/lr-ssmba/data/wo/ud/test.conllu"

with open(wolof_path) as inf, open(new_wolof_path, "w") as ouf:
    for line in inf:
        fields = line.strip().split("\t")
        if fields[0].startswith("#") or "-" not in fields[0]:
            ouf.write(line)
