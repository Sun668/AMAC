# Paper context

Title: AMAC: Risk-Aware Commitment under Asynchronous Modality Arrival for Multimodal Affective Agents.

Research question: how should an affective agent decide whether to wait, commit,
or revise when text, audio, and vision arrive asynchronously, without changing
the final prediction produced from all three modalities?

Method: AMAC estimates whether the current prefix state is correct and applies a
frozen threshold and revision margin. The third modality always forces the full
TAV state. LR is the deployment estimator; H0 is retained as the condition fixed in the archived local protocol before sealed-test evaluation and the
primary-test estimator.

Evidence: CH-SIMS v2 one-shot test, EmotionTalk contract-level external test,
competitive scalar-calibration controls, and real Go Agent replay.

Boundary: the asynchronous commitment protocol is author-defined. Dataset labels
and splits remain official where available. EmotionTalk does not provide an
end-to-end raw-media replication in this study.

