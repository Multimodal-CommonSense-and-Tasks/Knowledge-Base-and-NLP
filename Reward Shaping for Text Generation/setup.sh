pip install -r requirements.txt

# Download NMT dataset via huggingface
python dataset_downloader_nmt.py

# Hotfix diaparser module for compatibility with torch==1.13.1
python hotfix_diaparser.py 

# # Download comet small model
# wget https://unbabel-experimental-models.s3.amazonaws.com/comet/eamt22/eamt22-cometinho-da.tar.gz
# tar -xf eamt22-cometinho-da.tar.gz
# # Move to cache directory
# mkdir ~/.comet
# mv eamt22-cometinho-da ~/.comet/eamt22-cometinho-da
# # Remove tarball
# rm eamt22-cometinho-da.tar.gz