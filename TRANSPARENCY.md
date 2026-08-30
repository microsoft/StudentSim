# StudentSim

## Overview

StudentSim is a research framework for training and evaluating AI models that simulate how students respond to learning material. This release contains source code only: the Python code used to train and evaluate the models described in the StudentSim paper [link to be added once the paper is public], to prepare the chess training data, and to produce the chess auxiliary models (the chess tutor model and the chess style classifier). While the release focuses on chess, the underlying code and framework are general and have also been tried with other tasks such as L2 English writing and mathematics.

### What can StudentSim do?

StudentSim was developed to give AI tutoring researchers a realistic stand-in for a student against which to test their tutoring methods, without recruiting human students per experiment. The released training scripts let external users build their own student-simulator models on their own data; each model takes a domain problem as input, for example a chess board position, and produces a candidate student response, such as a chess move. It is not agentic, so it uses no tools, does no browsing, and takes no actions in an external environment.

A detailed discussion of StudentSim, including how it was developed and tested, can be found in the paper at: [link to be added once the paper is public]

### Intended uses

StudentSim is best suited for research use, and supports three goals. Reproducibility: external researchers can rerun the training and evaluation pipeline and verify the paper's chess results. Extensibility: the framework is a starting point for adapting the approach to other learning domains. Tutor research: researchers developing AI tutoring methods can use the released code to train their own student-simulator models, in chess or other domains, as stand-in students for their experiments.

StudentSim is being shared with the research community to facilitate reproduction of our results and foster further research in this area.

StudentSim is intended to be used by domain experts who are independently capable of evaluating the quality of outputs before acting on them.

### Out-of-scope uses

StudentSim is not well suited for anything beyond research. No student-facing deployment artifact is part of the release, and its outputs are simulated student responses rather than measurements of any real person's ability, so it should not be used to assess, rank, or make decisions about an identifiable learner. It is also not ready to be pointed at a new task as it stands. The released recipe was developed and tested most extensively on chess, and the paper reports more limited experiments on second-language English writing and mathematics, so how well it works on any other task, domain, or student population is unknown. Anyone applying the code to a new target task should evaluate it thoroughly on that task first, and should not rely on its outputs until they have done so.

We do not recommend using StudentSim in commercial or real-world applications without further testing and development. It is being released for research purposes.

StudentSim was not designed or evaluated for all possible downstream purposes. Developers should consider its inherent limitations as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness concerns specific to each intended downstream use.

Without further testing and development, StudentSim should not be used in sensitive domains where inaccurate outputs could suggest actions that lead to injury or negatively impact an individual's legal, financial, or life opportunities.

We do not recommend using StudentSim in the context of high-risk decision making (e.g. in law enforcement, legal, finance, or healthcare).

## How to get started

To begin using StudentSim, see <https://github.com/microsoft/StudentSim>, which hosts all of the source code under the MIT license. The training data can be reconstructed locally using the download and processing scripts documented in the GitHub README; no training or evaluation data is redistributed. The released training scripts target Qwen3-4B-Instruct-2507 as the base model, which users download from Qwen directly.

## Evaluation

StudentSim was evaluated on its ability to predict a specific chess player's next move and their response to natural-language coach guidance, using records drawn from Lichess (May 2025), on which each record pairs a board position with the player's actual move.

A detailed discussion of our evaluation methods and results can be found in the paper at: [link to be added once the paper is public]

### Evaluation methods

We used behavioral fidelity (F), which captures how well a simulator matches a student's own responses (in chess, top-1 move accuracy), and guidance responsiveness (R), which captures how readily it updates that response under a tutor's guidance (in chess, corrected-move rate), to measure StudentSim's performance.

We compared the performance of StudentSim against two kinds of baseline: the closed-source LLMs GPT-4o and GPT-5.4, prompted in-context with each player's profile, problem, and guidance; and Maia2, a chess-specific human-style move predictor conditioned on the board position (FEN) and the player's rating (ELO), using StudentSimEval, a standardized per-student protocol. It fixes a roster of 30 chess players in advance, independently of any method, together with per-player held-out records that are disjoint from all training data, so that every method is fit on the same records and scored on the same held-out set.

The model used for evaluation was Qwen3-4B-Instruct-2507. For more on this specific model, please see its model card at https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507.

Results may vary if StudentSim is used with a different model based on its unique design, configuration and training.

No formal RAI red-team exercise or Deployment Safety Board (DSB) review was carried out. The Responsible AI review for this record was scoped as a self-attest, of which this Transparency Note and the accompanying Impact Assessment are the deliverables, and that scope reflects how narrow the release is. DSB review is aimed at foundation models, and StudentSim is not one: it is a training framework, and the only model it produces is a small LoRA adapter over the public Qwen3-4B-Instruct-2507 base. The release ships code only, with no model weights and no data. The task is prediction rather than open-ended generation, and it is confined to chess: the model reads a chess board position and emits a single move in UCI notation, a few tokens with no free-form natural-language output. It is also not agentic, so it uses no tools, does no browsing, and takes no actions in an external environment.

### Evaluation results

At a high level, we found that StudentSim performed strongly on both axes. In chess, StudentSim reaches F = 0.51 and R = 0.91, compared with 0.23 and 0.72 for GPT-5.4 and 0.45 and 0.27 for Maia2, a skill-conditioned chess move prediction model. Baselines are strong on at most one axis: prompted LLMs follow tutor guidance fluently but do not reproduce a particular player's competence and mistakes, while Maia2 tracks a player's move distribution but has no input pathway for natural-language guidance.

## Limitations

StudentSim was developed for research and experimental purposes. Further testing and validation are needed before considering its application in commercial or real-world scenarios.

StudentSim was designed and tested using the English language. Performance in other languages may vary and should be assessed by someone who is both an expert in the expected outputs and a native speaker of that language.

Outputs generated by AI may include factual errors, fabrication, or speculation. Users are responsible for assessing the accuracy of generated content. All decisions leveraging outputs of the system should be made with human oversight and not be based solely on system outputs.

StudentSim inherits any biases, errors, or omissions produced by its base model. Developers are advised to choose an appropriate base LLM/MLLM carefully, depending on the intended use case.

StudentSim uses the Qwen3-4B-Instruct-2507 model. See https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 to understand the capabilities and limitations of this model.

StudentSim inherits any biases, errors, or omissions characteristic of its training data, which may be amplified by any AI-generated interpretations.

There has not been a systematic effort to ensure that systems using StudentSim are protected from security vulnerabilities such as indirect prompt injection attacks. Any systems using it should take proactive measures to harden their systems as appropriate.

## Best practices

Better performance can be achieved by supplying more records for the student being modeled. The two-stage design first trains one domain-specific adapter on records pooled across a larger set of students, then continues training that adapter on an individual student's own records, so each student's adapter refines a well-initialized shared foundation rather than training from scratch on sparse individual data.

We strongly encourage users to use LLMs/MLLMs that support robust Responsible AI mitigations, such as Azure Open AI (AOAI) services. Such services continually update their safety and RAI mitigations with the latest industry standards for responsible use. For more on AOAI's best practices when employing foundations models for scripts and applications:

- [What is Azure AI Content Safety?](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview)
- [Overview of Responsible AI practices for Azure OpenAI models](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/overview)
- [Azure OpenAI Transparency Note](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/transparency-note)
- [OpenAI's Usage policies](https://openai.com/policies/usage-policies)
- [Azure OpenAI's Code of Conduct](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/code-of-conduct)

Users are responsible for sourcing their datasets legally and ethically. This could include securing appropriate rights, ensuring consent for use of audio/images, and/or the anonymization of data prior to use in research.

Users are reminded to be mindful of data privacy concerns and are encouraged to review the privacy policies associated with any models and data storage solutions interfacing with StudentSim.

It is the user's responsibility to ensure that the use of StudentSim complies with relevant data protection regulations and organizational guidelines.

Developers should follow transparency best practices and inform end-users they are interacting with an AI system.

## License

MIT License

Nothing disclosed here, including the Out of Scope Uses section, should be interpreted as or deemed a restriction or modification to the license the code is released under.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

## Contact

This research was conducted by members of Microsoft Research. We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected/offensive behavior in our technology, please contact us at mgalley@microsoft.com.

If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations.
