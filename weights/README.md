# Released weights

- chsimsv2_correctness_ensemble.pt: five frozen CH-SIMS v2 correctness MLPs.
- chsimsv2_masked_fusion.pt: frozen full masked-fusion backbone, fold metadata,
  modality scalers, and five correctness MLPs.
- emotiontalk_correctness_ensemble.pt: five frozen annotation-derived
  EmotionTalk correctness MLPs.

The primary Ridge backbones have no original serialized checkpoint. They are
deterministically refit from the restricted official CH-SIMS v2 feature
container by models.training.fit_ridge_backbones. No reconstructed or post-hoc
Ridge weight is presented as an original artifact.

PyTorch checkpoints can execute Python during deserialization. Load only files
whose hashes match SHA256SUMS.
