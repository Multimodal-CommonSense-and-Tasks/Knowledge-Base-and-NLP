# Knowledge-Base-and-NLP

### This repository organizes researches related to AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data, especially Knowledge-Base-and-NLP task. This repository summarizes following researches.

## Research list
* CoTEVer: Chain of Thought Prompting Annotation Toolkit for Explanation Verification (EACL 2023) - Seungone Kim, Se June Joo, Yul Jang, Hyungjoo Chae, and Jinyoung Yeo.

  * To improve the correctness of the explanations, fine-tuning language models with explanation data is needed. However, there exists only a few datasets that can be used for such approaches, and no data collection tool for building them. Thus, we introduce CoTEVer, a tool-kit for annotating the factual correctness of generated explanations and collecting revision data of wrong explanations.

* Mind the Gap! Injecting Commonsense Knowledge for Abstractive Dialogue Summarization (COLING 2022) - Seungone Kim, Se June Joo, Hyungjoo Chae, Chaehyeong Kim, Seung-won Hwang, and Jinyoung Yeo.

  * In this paper, we propose to leverage the unique characteristics of dialogues sharing commonsense knowledge across participants, to resolve the difficulties in summarizing them. We present SICK, a framework that uses commonsense inferences as additional context.

* Dialogue Chain-of-Thought Distillation for Commonsense-aware Conversational Agents (EMNLP 2023) - Hyungjoo Chae, Yongho Song, Kai Tzu-iunn Ong, Taeyoon Kwon, Minjin Kim, Youngjae Yu, Dongha Lee, Dongyeop Kang, and Jinyoung Yeo.

  * Our focus is to facilitate such multi-hop reasoning over a dialogue context, namely dialogue chain-of-thought (CoT) reasoning. To this end, we propose a knowledge distillation framework that leverages LLMs as unreliable teachers and selectively distills consistent and helpful rationales via alignment filters. We further present DOCTOR, a DialOgue Chain-of-ThOught Reasoner that provides reliable CoT rationales for response generation.

* On Complementarity Objectives for Hybrid Retrieval (ACL 2023) - Dohyeon Lee, Seung-won Hwang, Kyungjae Lee, Seungtaek Choi, and Sunghyun Park.

  * Our key distinction is to show how this notion of residual complementarity is limited, and propose a new objective, denoted as RoC (Ratio of Complementarity), which captures a fuller notion of complementarity. We propose a two-level orthogonality designed to improve RoC, then show that the improved RoC of our model, in turn, improves the performance of hybrid retrieval. Our method outperforms all state-of-the-art methods on three representative IR benchmarks: MSMARCO-Passage, Natural Questions, and TREC Robust04, with statistical significance. Our finding is also consistent in various adversarial settings.

* Script, Language, and Labels: Overcoming Three Discrepancies for Low-Resource Language Specialization (AAAI 2023) - Jaeseong Lee, Dohyeon Lee, and Seung-won Hwang.

  * Although multilingual pretrained models (mPLMs) enabled support of various natural language processing in diverse languages, its limited coverage of 100+ languages lets 6500+ languages remain ‘unseen’. One common approach for an unseen language is specializing the model for it as target, by performing additional masked language modeling (MLM) with the target language corpus. However, we argue that, due to the discrepancy from multilingual MLM pretraining, a naive specialization as such can be suboptimal. Specifically, we pose three discrepancies to overcome. Script and linguistic discrepancy of the target language from the related seen languages, hinder a positive transfer, for which we propose to maximize representation similarity, unlike existing approaches maximizing overlaps. In addition, label space for MLM prediction can vary across languages, for which we propose to reinitialize top layers for a more effective adaptation. Experiments over four different language families and three tasks shows that our method improves the task performance of unseen languages with statistical significance, while previous approach fails to.

* Retrieval-augmented Video Encoding for Instructional Captioning (ACL 2023) - Yeonjoon Jung, Minsoo Kim, Seungtaek Choi, Jihyuk Kim, Minji Seo, and Seung-won Hwang.

  * Instructional videos make learning knowledge more efficient, by providing a detailed multimodal context of each procedure in instruction. A unique challenge posed by instructional videos is key-object degeneracy, where any single modality fails to sufficiently capture the key objects referred to in the procedure. For machine systems, such degeneracy can disturb the performance of a downstream task such as dense video captioning, leading to the generation of incorrect captions omitting key objects. To repair degeneracy, we propose a retrieval-based framework to augment the model representations in the presence of such key-object degeneracy. We validate the effectiveness and generalizability of our proposed framework over baselines using modalities with key-object degeneracy.

* Learning to Rank Generation with Pairwise Partial Rewards (EMNLP 2023) - Youngwon Lee, Jinu Lee, and Seung-won Hwang.

  * This paper studies the use of reinforcement learning for conditional text generation, which overcomes the limitation of the prevalent supervised maximum likelihood estimation approach. However, it still suffers from challenges including the large action space and the delayed reward, as the reward can be computed only af- ter an entire sequence is generated. To address these challenges, we propose a method that provides partial rewards for intermediate actions taken on partial sequences. This enables the model to promptly prioritize actions that lead to the generation of more desirable sequences. Our method’s key contribution lies in its focus on distinguishing relatively more desirable actions rather than striving to precisely estimate pointwise values for arbitrary partial sequences. Instead, our reward shaping method learns to discern the relative desirability between pairs of actions, or rank actions in a pairwise manner, only when necessary and feasible. This is materialized in an efficient way by leveraging the prefix tree constructed from the sampled sequences. Experimental results on paraphrase generation and lexically constrained machine translation tasks showcase the effectiveness of our method.

* Relevance-assisted Generation for Robust Zero-shot Retrieval (RaMDA) (EMNLP 2023) - Jihyuk Kim, Minsoo Kim, Joonsuk Park, and Seung-won Hwang.

  * Our contribution is showing that key biases can cause sampled PQ to be irrelevant, negatively contributing to generalization. We propose to preempt their generation, by dividing the generation into simpler subtasks, of generating relevance explanations and guiding the generation to avoid negative generalization.

## Acknowledgements
These works were supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT). Also, these works were supported by supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government(MSIT).
