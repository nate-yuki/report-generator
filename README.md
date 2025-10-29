# Deep Learning Model Robustness Evaluation Report Generator

This Python script generates comprehensive HTML reports from JSON data containing robustness evaluation results of deep learning models.

## Features

- **Clear style**:
  - Clear color palette without colors being too vibrant
  - Drop-down lists to reduce cluttering
  - Visual grouping
- **Three Main Sections**:
  1. Model and experiment configuration overview
  2. Complete results table
  3. Data visualizations using Matplotlib
- **Mobile Responsive**: Works on all device sizes

## Requirements

- Python 3.6+
- matplotlib

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
python generate_report.py -i input_file.json -o my_report.html
```

### Command Line Arguments
- `-i, --input`: Path to input JSON file (default: report_dict.json)
- `-o, --output`: Output HTML file path (default: robustness_report.html)

## JSON Input Format

The input JSON should have the following structure:

```json
{
    "desc": {
        "problem_type": "Problem Type Name",
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
                "Metric 1": 0.8333333333333334, // can be int or float
                // ...
                "Metric N": 0.0784313753247261
            },
            // ...
            "variable_param_valueV": {
                "Metric 1": 0.6333333333333333,
                // ...
                "Metric N": 0.3921568691730499
            }
        }
    }
}
```
