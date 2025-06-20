<div align="center">

# LoomUI — Automated UI Generation for ML Tasks
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) 
[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-380/) 
[![CI](https://github.com/langchain-ai/react-agent/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/react-agent/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/react-agent/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/react-agent/actions/workflows/integration-tests.yml)
[![Open in - LangGraph Studio](https://img.shields.io/badge/Open_in-LangGraph_Studio-00324d.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4NS4zMzMiIGhlaWdodD0iODUuMzMzIiB2ZXJzaW9uPSIxLjAiIHZpZXdCb3g9IjAgMCA2NCA2NCI+PHBhdGggZD0iTTEzIDcuOGMtNi4zIDMuMS03LjEgNi4zLTYuOCAyNS43LjQgMjQuNi4zIDI0LjUgMjUuOSAyNC41QzU3LjUgNTggNTggNTcuNSA1OCAzMi4zIDU4IDcuMyA1Ni43IDYgMzIgNmMtMTIuOCAwLTE2LjEuMy0xOSAxLjhtMzcuNiAxNi42YzIuOCAyLjggMy40IDQuMiAzLjQgNy42cy0uNiA0LjgtMy40IDcuNkw0Ny4yIDQzSDE2LjhsLTMuNC0zLjRjLTQuOC00LjgtNC44LTEwLjQgMC0xNS4ybDMuNC0zLjRoMzAuNHoiLz48cGF0aCBkPSJNMTguOSAyNS42Yy0xLjEgMS4zLTEgMS43LjQgMi41LjkuNiAxLjcgMS44IDEuNyAyLjcgMCAxIC43IDIuOCAxLjYgNC4xIDEuNCAxLjkgMS40IDIuNS4zIDMuMi0xIC42LS42LjkgMS40LjkgMS41IDAgMi43LS41IDIuNy0xIDAtLjYgMS4xLS44IDIuNi0uNGwyLjYuNy0xLjgtMi45Yy01LjktOS4zLTkuNC0xMi4zLTExLjUtOS44TTM5IDI2YzAgMS4xLS45IDIuNS0yIDMuMi0yLjQgMS41LTIuNiAzLjQtLjUgNC4yLjguMyAyIDEuNyAyLjUgMy4xLjYgMS41IDEuNCAyLjMgMiAyIDEuNS0uOSAxLjItMy41LS40LTMuNS0yLjEgMC0yLjgtMi44LS44LTMuMyAxLjYtLjQgMS42LS41IDAtLjYtMS4xLS4xLTEuNS0uNi0xLjItMS42LjctMS43IDMuMy0yLjEgMy41LS41LjEuNS4yIDEuNi4zIDIuMiAwIC43LjkgMS40IDEuOSAxLjYgMi4xLjQgMi4zLTIuMy4yLTMuMi0uOC0uMy0yLTEuNy0yLjUtMy4xLTEuMS0zLTMtMy4zLTMtLjUiLz48L3N2Zz4=)](https://langgraph-studio.vercel.app/templates/open?githubUrl=https://github.com/langchain-ai/react-agent)
</div>

## Introduction
**LoomUI** is an end-to-end framework that automatically generates full-featured, visually consistent, and task-aware user interfaces for machine learning applications. By leveraging large language models (LLMs), it translates structured task specifications (`task.yaml`) into interactive HTML interfaces powered by Tailwind CSS and vanilla JavaScript.

## Key Features

- **Task-Aware UI Synthesis**: Generates custom layouts based on input/output modality, model interaction, and data structure.
- **Tailwind-Based Styling**: Ensures modern, accessible, and responsive UI with consistent color palettes.
- **Single & Batch Processing Support**: Handles both manual input and dataset-based inference workflows.
- **LLM-Orchestrated Pipeline**: Modular agents for layout planning, component generation, JS injection, and visual refinement.
- **Bug Detection & Fixing**: Post-generation validation and repair of UI behavior.

## How It Works
LoomUI uses a modular pipeline orchestrated by LLMs to generate user interfaces for ML tasks:

- Task Analysis: Parses a task specification (e.g., task.yaml) describing ML model inputs, outputs, and workflow.
- Layout Planning: LLM agents determine the optimal arrangement of UI components.
- Component Generation: UI elements are created and styled using Tailwind CSS.
- Workflow Integration: Connects frontend components to backend ML APIs for real-time or batch processing.
- Validation & Repair: Automated testing and repair ensure the generated UI is functional and accessible.

## Getstart

## Sample `task.yaml`

```yaml
task_description:
  type: Text Classification
  description: Classify the emotion expressed in user-generated text.
  input: A sentence containing emotional content.
  output: One of the predefined emotion categories.

model_information:
  api_url: http://localhost:8000/api/emotion
  input_format:
    text: string
  output_format:
    label: string
    probability: float

dataset_description:
  data_source: ./data/emotion.csv
  data_format:
    - id: string
    - text: string
    - label: string
```

## License

**LoomUI** is released under the [MIT License](./LICENSE).


## Acknowledgements

This project was initiated during the **ISE AutoCode Challenge 1**, aiming to explore the frontier of automatic interface generation using LLMs.

