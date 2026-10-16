# Knowledge-Base-and-NLP

#### This repository organizes researches related to AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data, especially Knowledge-Base-and-NLP task.
#### This repository summarizes following researches.

## Research list
    
* CoTEVer: Chain of Thought Prompting Annotation Toolkit for Explanation Verification (EACL 2023) - Seungone Kim, Se June Joo, Yul Jang, Hyungjoo Chae, and Jinyoung Yeo.

  * The proposed Chain of Thought Prompting Annotation Toolkit for Explanation Verification (CoTEVer), is a tool-kit for annotating the factual correctness of generated explanations and collecting revision data of wrong explanations.

* COMMIT: Code-Mixing English-Centric Large Language Model for Multilingual Instruction Tuning (Findings of NAACL 2024) - Jaeseong Lee, YeonJoon Jung, and Seung-won Hwang.

  * The proposed code-mixed continual causal language modeling to align the decoder improves the exact match score of low-resourced language QA task by up to 32x.

* Mind the Gap! Injecting Commonsense Knowledge for Abstractive Dialogue Summarization (COLING 2022) - Seungone Kim, Se June Joo, Hyungjoo Chae, Chaehyeong Kim, Seung-won Hwang, and Jinyoung Yeo.

  * The proposed Summarizing with Injected Commonsense Knowledge (SICK), is a framework that uses commonsense inferences as additional context. SICK leverages the unique characteristics of dialogues sharing commonsense knowledge across participants, to resolve the difficulties in summarizing them.

* ContrastiveMix: Overcoming Code-Mixing Dilemma in Cross-Lingual Transfer for Information Retrieval (NAACL 2024) - Junggeun Do, Jaeseong Lee, and Seung-won Hwang.

  * The proposed ContrastiveMix balances the tension between the positive effect of code-mixing on aligning representations across languages and the negative impact it has on IR-specific objective of matching representations between queries and relevant passages.

* Dialogue Chain-of-Thought Distillation for Commonsense-aware Conversational Agents (EMNLP 2023) - Hyungjoo Chae, Yongho Song, Kai Tzu-iunn Ong, Taeyoon Kwon, Minjin Kim, Youngjae Yu, Dongha Lee, Dongyeop Kang, and Jinyoung Yeo.

  * The proposed DialOgue Chain-of-ThOught Reasoner (DOCTOR), is a knowledge distillation framework that leverages LLMs as unreliable teachers and selectively distills consistent and helpful rationales via alignment filters. DOCTOR provides reliable CoT rationales for response generation.

* DADA: Distribution-Aware Domain Adaptation of PLMs for Information Retrieval (Findings of ACL 2024) - Dohyeon Lee, Jongyoon Kim, Seung-won Hwang and Joonsuk Park.

  * The proposed DADA tackles the failure of pseudo-query generation for domain adaptation of informration retrieval in resembling real queries in the target domain, by incorporating term distirbution feedback.

* On Complementarity Objectives for Hybrid Retrieval (ACL 2023) - Dohyeon Lee, Seung-won Hwang, Kyungjae Lee, Seungtaek Choi, and Sunghyun Park.

  * The proposed Ratio of Complementarity (RoC), is a new objective which captures a fuller notion of complementarity. Improving RoC of model improves the performance of hybrid retrieval.

* Script-mix: Mixing Scripts for Low-resource Language Parsing (NAACL 2024) - Jaeseong Lee, Dohyeon Lee, and Seung-won Hwang.

  * The proposed ScriptMix, combines the complementary strengths and overcomes the hurdle in realizing the integration of the two, transliteration and vocabulary augmentation, for low-resource language adaptation of multilinugal pretrained language models.

* Script, Language, and Labels: Overcoming Three Discrepancies for Low-Resource Language Specialization (AAAI 2023) - Jaeseong Lee, Dohyeon Lee, and Seung-won Hwang.

  * The three discrepancies from Masked Language Modeling (MLM) pretraining, Script, Language, and Labels, lead into a naive specialization as such can be suboptimal. Script and linguistic discrepancy of the target language from the related seen languages, hinder a positive transfer, for which authors propose to maximize representation similarity, unlike existing approaches maximizing overlaps. In addition, label space for MLM prediction can vary across languages, for which authors propose to reinitialize top layers for a more effective adaptation.

* Retrieval-augmented Video Encoding for Instructional Captioning (ACL 2023) - Yeonjoon Jung, Minsoo Kim, Seungtaek Choi, Jihyuk Kim, Minji Seo, and Seung-won Hwang.

  * The proposed retrieval-based framework augments the model representations in the presence of key-object degeneracy. This framework repairs key-object degeneracy, where any single modality fails to sufficiently capture the key objects reffered to in the procedure, in the instructional video.

* Learning to Rank Generation with Pairwise Partial Rewards (EMNLP 2023) - Youngwon Lee, Jinu Lee, and Seung-won Hwang.

  * The proposed reward shaping method provides partial rewards for intermediate actions taken on partial sequences. This method enables the model to promptly prioritize actions that lead to the generation of more desirable sequences.

* Relevance-assisted Generation for Robust Zero-shot Retrieval (EMNLP 2023) - Jihyuk Kim, Minsoo Kim, Joonsuk Park, and Seung-won Hwang.

  * The proposed relevance-guided generation, is divided in two simple subtasks, generating relevance explanations and guiding the generation to avoid negative generalization. Relevance-guided generation method is more robust to domain shifts when key biases cause sampled Psuedo Queries (PQ) to be irrelevant, negatively contributing to generalization. 

* Chaining Event Spans for Temporal Relation Grounding (EACL 2024) - Jongho Kim, Dohyeon Lee, Minsoo Kim, and Seung-won Hwang.

  * The proposed TRN (timeline reasoning network) outperforms previous methods for temporal reading comprehension and temporal relation extraction tasks, by effectively resolving the spurious overlaps in answers using the predicted timeline.

* Train-Attention: Meta-Learning Where to Focus in Continual Knowledge Learning (NeurIPS 2024) - Yeongbin Seo, Dongha Lee, and Jinyoung Yeo.

  * The proposed Train-Attention, is a meta-learning framework that adaptively adjusts the attention mechanism during training to focus on relevant tokens, thereby improving the model's ability to learn new information without forgetting previously acquired knowledge.

* Large Language Models Are Clinical Reasoners: Reasoning-Aware Diagnosis Framework with Prompt-Generated Rationales (AAAI 2024) - Taeyoon Kwon, Kai Tzu-iunn Ong, Dongjin Kang, Seungjun Moon, Jeong Ryong Lee, Dosik Hwang, Beomseok Sohn, Yongsik Sim, Dongha Lee, and Jinyoung Yeo.

  * The proposed reasoning-aware diagnosis framework, is a novel framework that leverages LLMs as unreliable teachers and selectively distills consistent and helpful rationales via alignment filters.

* Language Models as Compilers (EMNLP 2024) - Hyungjoo Chae, Yeonghyeon Kim, Seungone Kim, Kai Tzu-iunn Ong, Beong-woo Kwak, Moohyeon Kim, Seonghwan Kim, Taeyoon Kwon, Jiwan Chung, Youngjae Yu, and Jinyoung Yeo.

  * The proposed language models as compilers, is a novel framework that uses psuedocode to have a better control over LLMs' reasoning process.

* Can Large Language Models be Good Emotional Supporter? Mitigating Preference Bias on Emotional Support Conversation (ACL 2024) - Dongjin Kang, Sunghwan Kim, Taeyoon Kwon, Seungjun Moon, Hyunsouk Cho, Youngjae Yu, Dongha Lee, and Jinyoung Yeo.

  * This paper first investigates the ability of LLMs as emotional supporters, and then proposes a method to mitigate the preference bias of LLMs on emotional support conversations.

* Cactus: Towards Psychological Counseling Conversations using Cognitive Behavioral Theory (EMNLP 2024) - Suyeon Lee, Sunghwan Kim, Minju Kim, Dongjin Kang, Dongil Yang, Harim Kim, Minseok Kang, Dayi Jung, Min Hee Kim, Seungbeen Lee, Kyoung-Mee Chung, Youngjae Yu, Dongha Lee, and Jinyoung Yeo.

  * This paper proposes a psychological counseling conversation framework using cognitive behavioral theory. In addition, it provides a new dataset for psychological counseling conversation.

* Commonsense-augmented Memory Construction and Management in Long-term Conversations via Context-aware Persona Refinement (EACL 2024) - Hana Kim, Kai Tzu-iunn Ong, Seoyeon Kim, Dongha Lee, and Jinyoung Yeo.

  * This paper introduces a novel framework that leverages commonsense-based persona expansion to refine uninformative persona sentences in long-term conversations, enhancing response quality through human-like persona refinement.

* Multitask Deep Learning for Joint Detection of Necrotizing Viral and Noninfectious Retinitis From Common Blood and Serology Test Data (IOVS 2024) - Kai Tzu-iunn Ong, Taeyoon Kwon, Harok Jang, Min Kim, Christopher Seungkyu Lee, Suk Ho Byeon, Sung Soo Kim, Jinyoung Yeo, Eun Young Choi.

  * This paper proposes a multitask deep learning framework for joint detection of necrotizing viral and noninfectious retinitis from common blood and serology test data.

* Coffee-Gym: An Environment for Learning Feedback for Code Editing (EMNLP 2024) - Hyungjoo Chae, Taeyoon Kwon, Seungjun Moon, Yongho Song, Dongjin Kang, Kai Tzu-iunn Ong, Beong-woo Kwak, Seonghyeon Bae, Seung-won Hwang, and Jinyoung Yeo.

  * This paper presents Coffee-Gym, a comprehensive RL environment for training models that provide feedback on code editing.

* Intended Target Identification for Anomia Patients with Gradient-based Selective Augmentation (EMNLP Findings 2024) - Jongho Kim, Romain Storaï and Seung-won Hwang.

  * This paper improves LM-based intended target identification for anomia patients by using gradient-based selective augmentation to handle semantic paraphasia and unseen relevant terms.

* Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models (ACL 2025) - Jongho Kim and Seung-won Hwang.

  * This paper uses counterfactual questions and collective constraints to improve the consistency of large language models in predicting temporal relations between events.

* Adaptive Retrieval for Reasoning (ACL 2026) - Jongho Kim, Jaeyoung Kim, Jihyuk Kim, Yu Jin Kim, Seung-won Hwang and Moontae Lee.

  * This paper proposes REPAIR, which uses reasoning plans as dense feedback signals to selectively guide adaptive retrieval and recover missing bridge documents for reasoning-intensive retrieval.

* Query Variant Detection Using Retriever as Environment (NAACL 2025 industry) - Minji Seo, Youngwon Lee, Seung-won Hwang, Seoho Song, Hee-Cheol Seo and Young-In Song.

  * This paper leverages the retriever as an environment, using retrieval outcomes as feedback to identify semantically equivalent query variants.

* Interventional Speech Noise Injection for ASR Generalizable Spoken Language Understanding (EMNLP 2024) - YeonJoon Jung, Jaeseong Lee, Seungtaek Choi, Dohyeon Lee, Minsoo Kim and Seung-won Hwang.

  * This paper proposes an interventional speech noise augmentation method that reduces ASR-specific bias by removing non-causal noise effects, improving the robustness of spoken language understanding to unseen ASR systems.

* HARP: Hesitation-Aware Reframing in Transformer Inference Pass (NAACL 2025) - Romain Storaï and Seung-won Hwang.

  * This paper presents HARP, a simple modification to the "off-the-shelf" Transformer forward pass that selectively applies additional computation when the model hesitates.

* PROM: Pivoted and Regulated Optimization for Multilingual Instruction Learning (NAACL 2025) - Jaeseong Lee, Seung-won Hwang, Hojin Lee, Yunju Bak and Changmin Lee.

  * This paper disentangles the roles of English and target-language data in multilingual instruction tuning to mitigate their negative interference.

* HIL: Hybrid Isotropy Learning for Zero-shot Performance in Dense retrieval (NAACL 2024) - Jaeyoung Kim, Dohyeon Lee and Seung-won Hwang.

  * This paper proposes a Hybrid Isotropy Learning (HIL) architecture that balances isotropic and anisotropic representations for improved zero-shot dense retrieval.

* Beyond Markovian Forgetfulness: Episodic Memory for Reasoning-Intensive Retrieval (ACL 2026) - Dohyeon Lee, Yeonseok Jeong and Seung-won Hwang.

  * This paper proposes Episodic Memory for Retrieval (EMR), which augments state-based retrieval with episodic memory to avoid repetitive reasoning cycles.

* Relevance to Utility: Process-Supervised Rewrite for RAG (ACL Findings 2026) - Jaeyoung Kim, Jongho Kim, Seung-won Hwang, Seoho Song and Young-In Song.

  * This paper proposes R2U, which learns document rewriting from downstream answer utility rather than retrieval relevance through process supervision.

* UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval (ACL Findings 2026) - Jongyoon Kim, Minseong Hwang and Seung-won Hwang.

  * This paper proposes UnIte, which filters inherently noisy documents using aleatoric uncertainty and prioritizes informative ones using epistemic uncertainty for retrieval domain adaptation.

* RaDA: Retrieval-augmented Web Agent Planning with LLMs (ACL Findings 2024) - Minsoo Kim, Victor Bursztyn, Eunyee Koh, Shunan Guo and Seung-won Hwang.

  * This paper proposes RaDA, which dynamically retrieves exemplars for both task decomposition and action generation, enabling Web agent planning without manually crafted exemplars.

* Smarter, Not Harder: Training-Free Adaptive Computation for Transformers (ACL Findings 2025) - Romain Storaï, Jaeseong Lee and Seung-won Hwang.

  * This paper proposes IMPACT, a PACT method that perturbs network weights rather than input embeddings, enabling efficient and deterministic adaptive computation with KV-cache reuse.

* tRAG: Term-level Retrieval-Augmented Generation for Domain-Adaptive Retrieval (NAACL 2025) - Dohyeon Lee, Jongyoon Kim, Jihyuk Kim, Seung-won Hwang and Joonsuk Park.

  * This paper proposes tRAG, which retrieves domain-specific terms from a corpus-level term pool to generate pseudo-queries with unseen terms for domain-adaptive retrieval.

* ECoRAG: Evidentiality-guided Compression for Long Context RAG (ACL Findings 2025) - Yeonseok Jeong, Jinsu Kim, Dohyeon Lee and Seung-won Hwang.

  * This paper proposes ECoRAG, which compresses retrieved documents based on evidentiality and adaptively retrieves additional evidence when the compressed context is insufficient for answering.

* Age-Related Scattered Hypofluorescent Spots as an Adverse Prognostic Factor for Polypoidal Choroidal Vasculopathy (Ophthalmology Science 2025) - Kai Tzu-iunn Ong, Seo Hee Kim, Seonghee Choi, Eun Jee Chung, Min Kim, Christopher Seungkyu Lee, Jinyoung Yeo, and Eun Young Choi.

  * This paper investigates the prognostic significance of age-related scattered hypofluorescent spots on late-phase indocyanine green angiography (ASHS-LIA) in polypoidal choroidal vasculopathy, using an AdaBoost model to predict disease stability, injection demand, and time to first remission.

* PRINCIPLES: Synthetic Strategy Memory for Proactive Dialogue Agents (Findings of EMNLP 2025) - Namyoung Kim, Kai Tzu-iunn Ong, Yeonjun Hwang, Minseok Kang, Iiseo Jihn, Gayoung Kim, Minju Kim, and Jinyoung Yeo.

  * The proposed PRINCIPLES is a synthetic strategy memory for proactive dialogue agents, derived through offline self-play simulations. It serves as reusable knowledge that guides strategy planning at inference time, expanding strategy coverage and mitigating preference bias without additional training or data annotation.

* EMBGuard: Constructing Hazard-Aware Guardrails for Safe Planning in Embodied Agents (ICML 2026) - Dongwook Choi, Taeyoon Kwon, Bogyung Jeong, Minju Kim, Yeonjun Hwang, Hyojun Kim, Byungchul Kim, Young Kyun Jang, and Jinyoung Yeo.

  * The proposed EMBGuard is the first MLLM-based safety guardrail for embodied agents, decoupling physical risk reasoning from agent policy. Given a (visual observation, action) pair, it identifies hazardous configurations and explains potential risks in natural language, accompanied by the EMBHazard training set and the EMBGuardTest benchmark.

* PAC-Bench: Evaluating Multi-Agent Collaboration under Privacy Constraints (Findings of ACL 2026) - Minjun Park, Donghyun Kim, Hyeonjong Ju, Seungwon Lim, Dongwook Choi, Taeyoon Kwon, Minju Kim, and Jinyoung Yeo.

  * The proposed PAC-Bench is a benchmark for systematically evaluating multi-agent collaboration under privacy constraints. Experiments reveal that privacy constraints substantially degrade collaboration performance through recurring coordination breakdowns, including early-stage privacy violations, overly conservative abstraction, and privacy-induced hallucinations.

* Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity (ICLR 2026) - Yeongbin Seo, Gayoung Kim, Jaehyung Kim, and Jinyoung Yeo.

  * The proposed prior-based filter estimates token priors from corpus-level term frequency statistics and filters documents by the mean and standard deviation of those priors, serving as a fast proxy to perplexity while requiring no model inference. It achieves the highest average performance across 20 downstream benchmarks while reducing time cost by over 1000x.

* Evidentiality-aware Retrieval for Overcoming Abstractiveness in Open-Domain Question Answering (Findings of EACL 2024) - Yongho Song, Dahyun Lee, Myungha Jang, Seung-won Hwang, Kyungjae Lee, Dongha Lee, and Jinyoung Yeo.

  * The proposed Evidentiality-Aware Dense Passage Retriever (EADPR) takes a data-centric approach to the misalignment between relevance and answerability in abstractive ODQA, synthesizing distractor samples by removing evidence spans from gold passages. Treating these distractors as both hard negatives and pseudo-positives, EADPR learns to rank evidence passages above distractors and distractors above irrelevant contexts.

* Predicting branch retinal vein occlusion development using multimodal deep learning and pre-onset fundus hemisection images (Scientific Reports 2025) - Eun Young Choi, Dongyoung Kim, Jinyeong Kim, Eunjin Kim, Hyunseo Lee, Jinyoung Yeo, Tae Keun Yoo, and Min Kim.

  * This paper predicts future branch retinal vein occlusion from pre-onset, metadata-matched fundus hemisection images, segmenting the optic disc and retinal blood vessels with a U-net and combining fundus and vessel images in a BV-enhanced multimodal EfficientNetB0 model whose predictions focus on arteriovenous crossing regions in the retinal vascular arcade.

* ToolHaystack: Stress-Testing Tool-Augmented Language Models in Realistic Long-Term Interactions (arXiv 2025) - Beong-woo Kwak, Minju Kim, Dongha Lim, Hyungjoo Chae, Dongjin Kang, Sunghwan Kim, Dongil Yang, and Jinyoung Yeo.

  * The proposed ToolHaystack is a benchmark that evaluates tool-augmented language models in long-term interactions, where the information needed for a tool call is scattered across many sessions and buried among unrelated tasks, testing Context Recall, Information Shift, and Missing Context.

* ANI: Adaptive Numerical Injection for Unifying Semantic and Arithmetic Representations in Numerical Reasoning (EMNLP 2026) - Jinsung Jeon and Seung-won Hwang.

  * This paper proposes ANI (Adaptive Numerical Injection), which uses context-aware gating to selectively inject numerical embeddings into LLM representations, improving arithmetic reasoning while preserving the semantic roles of numbers and general language capabilities.
