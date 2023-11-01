# curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
# sudo apt-get install git-lfs
# git lfs install

# huggingface-cli login
# [token] taken from https://huggingface.co/settings/tokens

from transformers import AutoModel, AutoTokenizer
generator_path="RaMDA-G"
gen_model = AutoModel.from_pretrained(generator_path)
gen_tokenizer = AutoTokenizer.from_pretrained(generator_path)
gen_model.push_to_hub("JihyukKim/RaMDA-G")
gen_tokenizer.push_to_hub("JihyukKim/RaMDA-G")

from sentence_transformers import SentenceTransformer
retriever_path="./RaMDA-R-nfcorpus"
retriever = SentenceTransformer(retriever_path)
retriever.save_to_hub("JihyukKim/RaMDA-R-nfcorpus")

retriever_path="./RaMDA-R-scidocs"
retriever = SentenceTransformer(retriever_path)
retriever.save_to_hub("JihyukKim/RaMDA-R-scidocs")

retriever_path="./RaMDA-R-trec-covid-v2"
retriever = SentenceTransformer(retriever_path)
retriever.save_to_hub("JihyukKim/RaMDA-R-trec-covid-v2")

retriever_path="./RaMDA-R-climate-fever"
retriever = SentenceTransformer(retriever_path)
retriever.save_to_hub("JihyukKim/RaMDA-R-climate-fever")

from sentence_transformers import SentenceTransformer
retriever = SentenceTransformer("JihyukKim/RaMDA-R-nfcorpus")