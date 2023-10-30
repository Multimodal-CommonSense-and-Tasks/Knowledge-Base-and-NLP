Code Instruction
This software includes code for temporal description retrieval and dense video captioning on youcook2.
As we use lxmert encoding for video frames, we first explain how to use lxmert for youcook2.
To use lxmert encodings, 
1. yc2 videos should be downloaded from youtube including transcript.
2. extract image features with faster rcnn to be fed to lxmert (we recommend to visit lxmert official github) 
3. clone official lxmert code and place official lxmert pretrained weight as official github says.

For temporal description retrieval, use dual_encoder-lxmert_sequence_encoder_lstm-outsegs__framecontrast.ipynb code.
For efficient training we dumped image features from faster rcnn tokenized result to be used for pytorch dataset code in above mentioned ipynb file.
Run each cells in dual_encoder-lxmert_sequence_encoder_lstm-outsegs__framecontrast.ipynb

For dense video captioning, clone official github repo of EMT or PDVC under EMT_yc2 or PDVC_yc2.
first dump lxmert encoding similar to dual_encoder-lxmert_sequence_encoder_lstm-outsegs__framecontrast.ipynb.
To use EMT for dense video captioning, use densecap_ours_lxmert_seq_contrast.ipynb.
For PDVC, run python train.py --cfg_path ${config_path} --gpu_id ${GPU_ID} --data_rescale 0 with config_path=cfgs/yc2_tsn_pdvc_lxmert+contrast_recipe_768_no_base_encoder_5e-5.yml