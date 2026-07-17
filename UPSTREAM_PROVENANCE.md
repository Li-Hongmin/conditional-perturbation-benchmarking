# Upstream provenance

This record identifies the external software and inputs used to generate the
frozen predictions. It does not redistribute those materials.

## PerturBench

- Repository: `https://github.com/altoslabs/perturbench`
- Commit: `4825e392294768da4b35561a76502c7006d6453e`
- Licence: BSD-3-Clause with additional model-specific notices
- Configuration root:
  `src/perturbench/configs/experiment/neurips2025/norman19/`

| Public model identifier | Configuration file | SHA-256 |
| --- | --- | --- |
| `pb_linear_additive` | `linear_best_params_norman19.yaml` | `4a29cc54ae910fb05db2ebd5d663f6a64dc95e3ab6249c66fcadee5d8e2504f3` |
| `pb_decoder_only` | `decoder_best_params_norman19.yaml` | `20611fdf9bb3e1d2fd10d3f6b114d6fbf19dc9b19712fd54ed867d54a3644a78` |
| `pb_latent_additive` | `latent_best_params_norman19.yaml` | `d282060062f8425ac9f8601fddb793a8b0154860572274ba3c458977ef80d25f` |
| `pb_cpa` | `cpa_best_params_norman19.yaml` | `28c060985f71ffc1309f0c4749f49a8624c403fe43bd7f25cd475bf213bdcc7c` |
| `pb_biolord_star` | `biolord_best_params_norman19.yaml` | `38518ddbc7616d1c402fe18a08c203861a06cb9cfbd59cc283e4290817aba2a2` |
| `pb_sparse_additive_vae` | `sams_best_params_norman19.yaml` | `2b629e7827d821ee3057c7487d04e21a71cfa26825f7bf0a8896f3a0bc7f54b6` |

## Norman inputs

- Study: Norman et al., Science (2019), DOI
  `10.1126/science.aax4438`, GEO `GSE133344`
- Processed Norman matrix identity:
  `5e36bfa440c8ad1107a2559dcf3c760b596ccd6783902937f0b41722f061ef9b`
- Frozen split identity:
  `1c99040b7d4b753bf00b6da83430b9936824b27cc238793e827b7dd9e641bcc9`
- Common-feature count: 1,931

The source matrix, split file, model implementations and checkpoints are not
part of this public post-result replay. The confidential reviewer package
contains the author-generated orchestration source and the exact PerturBench
configuration files under their upstream licence.
