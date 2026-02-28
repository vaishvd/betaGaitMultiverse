# Multiverse analysis of beta activity during gait
This repository works on an analysis pipeline for preprocessing and analyzing EEG data recorded during split-belt treadmill walking. The goal is to study neural dynamics across the gait cycle, with a focus on beta activity between the swing and stance phase.

The project is currently under active development. Code structure, preprocessing choices, and analysis steps may change as the methodological approach is refined.

## Overview

This repository implements a full analysis workflow from raw EEG to gait-locked neural activity:
    - Preprocessing (filtering, rereferencing, artifact handling)= - Event detection and epoching around gait events
    - Automated artifact rejection
    - Independent Component Analysis (ICA)
    - Time-domain and time–frequency analyses
    - Exploration of alternative preprocessing strategies using a Multiverse analysis

The emphasis is on transparency, reproducibility, and a quantitative evaluation of how preprocessing decisions affect neural measures.

## Repository structure
The workflow is currently notebook-driven. The inital analysis pipeline can be found in [`initialanalysis.ipynb`](initialanalysis.ipynb)

