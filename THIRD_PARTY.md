# Third-Party Components

This repository vendors a small amount of third-party source code.

## External dataset source: `Bjonze/Public-Cardiac-CT-Dataset`

- Source: <https://github.com/Bjonze/Public-Cardiac-CT-Dataset>
- Role here: external dataset source used by this project
- Associated publication: "A Public Cardiac CT Dataset Featuring the Left Atrial Appendage" (STACOM 2025 MICCAI workshop)

This repository uses data derived from that public cardiac CT dataset.
Please acknowledge and cite the original dataset paper and repository when describing or redistributing work based on those data.

## `HINTLab/NeAR`

- Source: <https://github.com/HINTLab/NeAR>
- Role here: upstream project this repository was adapted from
- Upstream license: Apache License 2.0

This repository reorganizes and extends the original NeAR codebase for cardiac CT annotation repair.
The upstream project should be acknowledged when redistributing or describing the origin of this codebase.

## `surface_distance/`

- Source: <https://github.com/deepmind/surface-distance>
- Upstream copyright: Google / DeepMind
- License: Apache License 2.0

The files in `surface_distance/` retain their original license headers.
They are included to avoid requiring an extra runtime dependency for surface-distance utilities.
