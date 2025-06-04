# Handling Missing Modalities

This repository provides a small example on how missing input modalities can be addressed. The
`MissingModalitiesPreprocessor` fills absent image channels with zeros during preprocessing. A
corresponding trainer (`nnUNetTrainerModalityPrompt`) wraps the default architecture with simple
per‑modality prompt encoders. These encoders process each input channel separately before passing
the result to the standard nnU‑Net network.

Use the new components as follows:

```bash
