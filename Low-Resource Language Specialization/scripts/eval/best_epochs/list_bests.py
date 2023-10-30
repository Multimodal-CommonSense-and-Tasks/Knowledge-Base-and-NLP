import os, json
for fn in os.listdir("."):
    if "_bert" in fn or "lapt" in fn:
        with open(fn) as f:
            min_loss = 10000000
            best_epoch = None
            jstr = f.readlines()[0]
            jstr = jstr.replace("'", '')
            jstr = jstr.replace("{", '{"')
            jstr = jstr.replace(":", '":')
            jstr = jstr.replace(", ", ', "')
            j = json.loads(jstr)
            for key, val in j.items():
                if val["eval_loss"] < min_loss:
                    min_loss = val["eval_loss"]
                    best_epoch = key
            print(best_epoch)
