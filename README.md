# Deep Learning Model Robustness Evaluation Report Generator

This Python script generates comprehensive HTML reports from JSON data containing robustness evaluation results of deep learning models.

## Features

- **Three Sections**:
  1. Model and experiment configuration overview
  2. Complete results table
  3. Data visualizations using Matplotlib
- **Details**:
  - Metrics without `_` at the end are treated as user metrics and put to the end
  - Metrics with names starting with `Clean_` (case-insensitive) are considered to be calculated on unperturbed samples and should have equal values across experiments; a warning is printed otherwise
  - Metrics with names `X` and `Clean_X` (case-insensitive) are grouped in plots and in the table

## Requirements

- Python 3.10.18 or compatible
- matplotlib 3.10.7 or compatible

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python generate_report.py
```

### Custom Input/Output Files
```bash
python generate_report.py -i input_file.json -o my_report.html -p ./plots/ -PTp problem_types.json
```

### Command Line Arguments
- `-i, --input`: Path to input JSON file (default: report_dict.json)
- `-o, --output`: Output HTML file path (default: robustness_report.html)
- `-p, --plots-path`: Path to output plots directory (default: .)
- `-PTp, --problem-types-path`: Path to problem types JSON file (default: ./problem_types.json)

## JSON Input Format

The input JSON should have the following structure:

```json
{
    "desc": {
        "problem_type": "Problem type name",
        "model_name": "model_name",
        "model_parameters": { // all values are dictionaries
            "parameter1": { // values of a parameter cannot be dictionaries
                "subparameter1": 4,
                "subparameter2": 0.0001,
                // ...
                "subparameterS": "weights.pth"
            },
            "parameter2": {}, // can also be empty
            // ...
            "parameterM": {}
        },
        "dataset_loader_name": "dataset_loader_name",
        "dataloader_parameters": { // same format as "model_parameters"
            "parameter1": {
                "subparameter1": 42,
                // ...
                "subparameterP": 3.14,
            },
            // ...
            "parameterD": {}
        }
    },
    "experiments": {
        "attack": "attack_name",
        "variable_param_name": "variable_param_name",
        "fixed_attack_params": { // values can be ints, floats, or strings
            "parameter1": 0.0392156862745098,
            // ...
            "parameterA": 100
        },
        "metrics": { // keys represent variable parameter values
            "variable_param_value1": { // keys inside each variable_param_value must be the same
                "Metric_1_": 0.8333333333333334, // can be int or float
                // ...
                "Metric_N-1": 0.0784313753247261, // treated as user metric
                "Metric_N_": 83.3333333333333334 // treated as non-user metric
            },
            // ...
            "variable_param_valueV": {
                "Metric_1_": 0.6333333333333333,
                // ...
                "Metric_N-1": 0.3921568691730499,
                "Metric_N_": 63.3333333333333334
            }
        }
    }
}
```
