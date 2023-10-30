I ran
```bash
cd ..
cp -r bert bert_v2
tf_upgrade_v2 --intree bert_v2 --outtree bert_v3 --reportfile report.txt
cp bert_v3/* bert/
```
and
- changed tf.flags to absl.flags
- changed the layer_norm following https://stackoverflow.com/a/62357941
