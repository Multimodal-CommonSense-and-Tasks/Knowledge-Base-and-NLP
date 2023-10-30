local path = std.extVar("MTL_REPO_SHORTNAME");

local thunks = {
    hyak()::
    import "/gscratch/ark/echau18/research-lr-ssmba/config/lib/mtl.libsonnet",
    pinot()::
    import "/homes/gws/echau18/research-lr-ssmba/config/lib/mtl.libsonnet",

};

thunks[path]()