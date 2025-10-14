# Deep Learning Model Robustness Evaluation Report Generator

This Python script generates comprehensive HTML reports from JSON data containing robustness evaluation results of deep learning models.

## Features

- **Professional Styling**: Clean, modern design with gradient backgrounds and responsive layout
- **Three Main Sections**:
  1. Model and experiment configuration overview
  2. Interactive data visualizations using Plotly
  3. Detailed results tables with multiple groupings
- **Interactive Visualizations**:
  - Metrics comparison bar charts
  - Epsilon vs metrics heatmaps
  - Radar charts for attack comparison
- **PDF Export Ready**: Optimized for printing and PDF export
- **Mobile Responsive**: Works on all device sizes

## Requirements

- Python 3.6+
- plotly>=5.0.0
- pandas>=1.3.0

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python generate_report.py input_file.json
```

### Custom Output File
```bash
python generate_report.py input_file.json -o my_report.html
```

### Command Line Arguments
- `input_file`: Path to input JSON file (required)
- `-o, --output`: Output HTML file path (default: robustness_report.html)

## JSON Input Format

The input JSON should have the following structure:

```json
{
    "desc": {
        "model": "model_name",
        "model_parameters": {
            "parameter1": "value1",
            "parameter2": "value2"
        }
    },
    "experiments": [
        {
            "attack": "attack_name",
            "eps": epsilon_value,
            "metrics": {
                "acc": accuracy_value,
                "asr": attack_success_rate,
                "f1": f1_score
            }
        }
    ]
}
```

### Field Descriptions

- **desc.model**: Name of the deep learning model
- **desc.model_parameters**: Dictionary of model configuration parameters
- **experiments**: Array of experiment results
- **attack**: Name of the attack method (e.g., "pgd", "ifgsm", "no_attack")
- **eps**: Epsilon value used in the attack
- **metrics**: Performance metrics
  - **acc**: Accuracy (0.0 to 1.0)
  - **asr**: Attack Success Rate (0.0 to 1.0)
  - **f1**: F1 Score (0.0 to 1.0)

## Report Sections

### 1. Model and Experiment Configuration
- Displays model information and parameters
- Lists all attack methods and their epsilon values
- Shows total number of experiments

### 2. Data Visualizations
- **Metrics Comparison**: Bar charts showing average metrics by attack type
- **Epsilon Heatmap**: Heatmaps showing relationships between epsilon values and metrics
- **Radar Chart**: Multi-dimensional comparison of attack methods

### 3. Detailed Results Tables
- **Complete Results**: All experiments with all metrics
- **Metrics Grouped**: Separate tables for each metric, sorted by performance
- **Attack Grouped**: Results organized by attack method

## Exporting to PDF

1. Open the generated HTML file in your web browser
2. Use the browser's Print function (Ctrl+P or Cmd+P)
3. Select "Save as PDF" as the destination
4. Adjust print settings as needed (margins, scale, etc.)
5. Save the PDF

## Customization

The script can be easily customized by modifying:

- **CSS Styles**: Edit the `_get_css_styles()` method for different styling
- **Visualizations**: Modify the plotting methods to add new chart types
- **Tables**: Adjust table generation methods for different data presentations
- **Layout**: Change the HTML structure in the generation methods

## Example Output

The generated report includes:
- Responsive design that works on desktop and mobile
- Interactive charts powered by Plotly
- Professional color scheme with gradient backgrounds
- Clean typography optimized for readability
- Print-friendly styling for PDF export

## Troubleshooting

### Common Issues

1. **JSON Parse Error**: Ensure your JSON file is properly formatted with correct syntax
2. **Missing Dependencies**: Install required packages using `pip install -r requirements.txt`
3. **Empty Visualizations**: Check that your JSON contains valid experiment data
4. **File Not Found**: Verify the input file path is correct

### Error Messages

- "Invalid JSON": Check JSON syntax, especially commas and brackets
- "File not found": Verify the input file path exists
- "Import error": Install missing dependencies

## Support

For issues or feature requests, please check:
1. JSON file format matches the expected structure
2. All required dependencies are installed
3. Python version is 3.6 or higher

## License

This script is provided as-is for robustness evaluation reporting purposes.