import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "dark_blue": "#2b4f66",
    "muted_blue": "#4f6d7a",
    "gray": "#6e7580",
    "light_gray": "#d9dbe0",
}


def read_json(path: str) -> dict[str, Any]:
    """Load dictionary from JSON file at path."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {path}: {e}")
        return None


def safe_int(s: str):
    try:
        return int(s)
    except Exception:
        return None


def safe_float(s: str):
    try:
        return float(s)
    except Exception:
        return None

def try_make_num(values: dict[str, dict[str, Any]], sort: bool = False) -> dict[int | float, dict[str, Any]]:
    """If possible, converts all elements in values to ints, otherwise converts every convertible element to float"""
    all_numeric = True
    converted = [safe_int(v) for v in values]
    if any(v is None for v in converted):
        converted = [safe_float(v) for v in values]
        if any(v is None for v in converted):
            converted = [c if c is not None else v for v, c in zip(values, converted)]
            all_numeric = False
    if sort and all_numeric:
        converted.sort()
    return converted


def format_value(v: Any, max_decimals: int = 5) -> str:
    if isinstance(v, float):
        if abs(v) >= 10 ** -max_decimals and abs(v) < 10 ** max_decimals or v == 0:
            return f"{v:.{max_decimals + 1 - len(str(int(v)))}f}"
        return f"{v:.{max(1, max_decimals - 4)}e}"
    if isinstance(v, int):
        if abs(v) <= 10 ** (max_decimals + 2):
            return f"{v}"
        return f"{v:.{max(0, max_decimals - 4)}e}"
    return str(v)


def flatten_param_dict(d: dict[str, Any], indent: int = 0) -> list[str]:
    """Flatten the 1-level dict where values are dicts possibly empty.

    Replace empty dicts with the string 'default'.
    For nested dicts, print sub-items on new lines with indentation.
    """
    out = []
    pad = "    " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            if not v:  # Empty dict
                out.append(f"{pad}{k}: по умолчанию")
            else:
                out.append(f"{pad}{k}:")
                for ik, iv in v.items():
                    out.extend(flatten_param_dict({ik: iv}, indent + 1))
        else:
            # Format all floats, keep ints as is
            if isinstance(v, (int, float)):
                out.append(f"{pad}{k}: {format_value(v)}")
            else:
                out.append(f"{pad}{k}: {v if v else 'по умолчанию'}")
    return out


def ensure_metric_floats(metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    out = OrderedDict()
    for param, m in metrics.items():
        out[param] = {k: float(v) for k, v in m.items()}
    return out


def make_tables(
    report: dict[str, Any],
    metrics_meta: dict[str, dict[str, Any]],
    sort_param_values: bool = False
) -> str:
    """Return HTML for the single metrics table, with Clean_X after X (case-insensitive)."""
    experiments = report["experiments"]
    var_name = experiments["variable_param_name"]
    metrics = ensure_metric_floats(experiments["metrics"])

    # columns: var_name, then metric keys, with Clean_X after X (case-insensitive)
    sample = next(iter(metrics.values()))
    metric_keys = list(sample.keys())

    # Move user metrics to the end
    user_metrics = [k for k in metric_keys if not k.endswith("_")]
    non_user_metrics = [k for k in metric_keys if k.endswith("_")]

    # Build mapping: for each X, if Clean_X (case-insensitive) exists, place after X
    used = set()
    ordered_keys = []
    for k in non_user_metrics:
        kl = k.lower()
        if kl.startswith("clean_"):
            continue
        ordered_keys.append(k)
        used.add(k)
        clean_key = None
        clean_name = f"clean_{k}".lower()
        for mk in non_user_metrics:
            if mk.lower() == clean_name:
                clean_key = mk
                break
        if clean_key:
            ordered_keys.append(clean_key)
            used.add(clean_key)
    # Add any remaining keys (e.g., only Clean_X present)
    for k in non_user_metrics:
        if k not in used:
            ordered_keys.append(k)
    # Add user metrics at the end
    for k in user_metrics:
        ordered_keys.append(k)

    # Build HTML table
    html = []
    html.append('<table class="metrics-table">')
    # header
    header_cells = [f"<th>{var_name}</th>"]
    for k in ordered_keys:
        # Try to find metric name in metrics_meta
        display_name = k
        if k in metrics_meta:
            display_name = metrics_meta[k].get("name", k)
        header_cells.append(f"<th>{display_name}</th>")
    html.append("<tr>" + "".join(header_cells) + "</tr>")

    param_values = list(metrics.keys())
    param_values = try_make_num(param_values, sort=sort_param_values)
    for param_value, m in zip(param_values, metrics.values()):
        row = [f"<td>{format_value(param_value)}</td>"]
        for k in ordered_keys:
            row.append(f'<td>{format_value(m.get(k, ""))}</td>')
        html.append("<tr>" + "".join(row) + "</tr>")

    html.append("</table>")
    return "\n".join(html)


def plot_metrics(
    report: dict[str, Any],
    out_dir: str,
    metrics_meta: dict[str, dict[str, Any]],
    sort_param_values: bool = False
) -> list[str]:
    """Plot metrics from report and store plots in out_dir."""
    experiments = report["experiments"]
    var_name = experiments["variable_param_name"]
    metrics = ensure_metric_floats(experiments["metrics"])

    # Determine x values: try to convert keys to ints or floats
    raw_x = list(metrics.keys())
    x = try_make_num(raw_x, sort=sort_param_values)
    numeric_ticks = all(safe_float(v) is not None for v in x)
    if not numeric_ticks:
        # For string values, use evenly spaced x positions
        x = list(range(len(raw_x)))
    int_ticks = all(safe_int(v) is not None for v in raw_x)

    sample = next(iter(metrics.values()))
    metric_keys = list(sample.keys())

    # Move user metrics to the end
    user_metrics = [k for k in metric_keys if not k.endswith("_")]
    non_user_metrics = [k for k in metric_keys if k.endswith("_")]

    # detect Clean_{X} patterns (case-insensitive)
    clean_map = {}
    for k in metric_keys:
        if k.lower().startswith("clean_"):
            base = k[6:].strip()
            clean_map[base.lower()] = k

    generated = []
    # For each base metric to plot: either k or base if clean exists
    to_plot_bases = []
    for k in non_user_metrics:
        if k.lower().startswith("clean_"):
            continue
        to_plot_bases.append(k)
    for k in user_metrics:
        if k.lower().startswith("clean_"):
            continue
        to_plot_bases.append(k)

    for metric_name in to_plot_bases:
        ys = [metrics[p].get(metric_name, float("nan")) for p in raw_x]

        plt.figure(figsize=(18.72 / 1.5, 12.48 / 1.5))
        plt.rcParams.update(
            {
                "font.size": 26,
                "axes.labelsize": 26,
                "axes.titlesize": 28,
                "xtick.labelsize": 24,
                "ytick.labelsize": 24,
                "legend.fontsize": 24,
                "lines.linewidth": 5,
                "lines.markersize": 20,
                "axes.titlepad": 20,  # Increase space between title and plot
            }
        )
        plt.plot(x, ys, marker="o" if len(x) < 7 else None, color="#5a8bb0", label="Attacked")

        if numeric_ticks:
            # Force integer ticks for integer parameters
            plt.locator_params(axis="x", integer=int_ticks)
        else:
            # For string values, set the x-ticks to the formatted values
            formatted_ticks = [format_value(x) for x in try_make_num(raw_x)]
            rotate_ticks = len(formatted_ticks) * max(len(str(val)) for val in formatted_ticks) >= 49
            plt.xticks(x, formatted_ticks, rotation=45 if rotate_ticks else 0, ha="right" if rotate_ticks else "center")

        clean_key = clean_map.get(metric_name.lower())
        legend_needed = False
        if clean_key:
            ys_clean = [metrics[p].get(clean_key, float("nan")) for p in raw_x]
            all_same = all(abs(y - ys_clean[0]) < 1e-9 for y in ys_clean)
            if not all_same:
                print(f"Warning: Clean_{metric_name} values differ across attack params")
            avg_clean = float(ys_clean[0])
            if numeric_ticks:
                plt.axhline(avg_clean, color="#bcd4e6", linestyle="--", label="Clean")
            else:
                plt.plot(x, [avg_clean] * len(x), color="#bcd4e6", linestyle="--", label="Clean")
            legend_needed = True

        display_name = metric_name
        if metric_name in metrics_meta:
            display_name = metrics_meta[metric_name].get("name", metric_name)
        plt.title(display_name)

        plt.xlabel(var_name)
        plt.ylabel("значение")
        plt.grid(color="#c5d6e6", linestyle="-", linewidth=0.8)
        if legend_needed:
            plt.legend()

        arrow_drawn = False
        meta = metrics_meta.get(metric_name, None)
        if meta is not None:
            higher_better = meta.get("higher_better")
            if higher_better is not None:
                arrow_drawn = True
                max_y_tick_len = max(len(label.get_text()) for label in plt.yticks()[1])
                plt.subplots_adjust(left=0.1 + 0.1 / 6 * max_y_tick_len)

                # Fixed arrow position in figure fraction
                arrow_x = 0.04
                text_x = 0.015
                bottom_y = 0.12
                top_y = 1 - bottom_y
                if higher_better:
                    top_y, bottom_y = bottom_y, top_y

                plt.annotate(
                    "",
                    xy=(arrow_x, bottom_y), xycoords="figure fraction",
                    xytext=(arrow_x, top_y), textcoords="figure fraction",
                    arrowprops=dict(
                        color="#5a8bb0",
                        lw=plt.rcParams["lines.linewidth"],
                        headwidth=plt.rcParams["lines.markersize"],
                        headlength=plt.rcParams["lines.markersize"],
                    ),
                    clip_on=False,
                )
                plt.text(
                    text_x, 0.5, f"{'больше' if higher_better else 'меньше'} — лучше",
                    ha="center", va="center", rotation=90,
                    fontsize=plt.rcParams["legend.fontsize"], color="#5a8bb0",
                    transform=plt.gcf().transFigure,
                    clip_on=False,
                )
        if not arrow_drawn:
            plt.tight_layout()

        plot_path = os.path.join(out_dir, f"{metric_name}.svg")
        plt.savefig(plot_path, format="svg", dpi=150)
        plt.close()
        generated.append(plot_path)

    return generated


def build_report_html(
    report: dict[str, Any],
    plots: list[str],
    problem_types: dict[str, str],
    metrics_meta: dict[str, dict[str, Any]],
    sort_param_values: bool = False
) -> str:
    """Build HTML report string from the given report dictionary and list of plot paths."""
    desc = report["desc"]
    experiments = report["experiments"]

    # Section 1: grouped info, 3 columns
    model_params = desc.get("model_parameters", {})
    dataloader_params = desc.get("dataloader_parameters", {})

    s1 = []
    s1.append('<div class="global-card">')
    s1.append(f"<h1>Отчет устойчивости {desc.get('model_name')} к атаке {experiments.get('attack')}</h1>")
    s1.append('<h2 class="section-title">Обзор</h2>')

    s1.append('<div class="section-row">')
    # Model info
    s1.append('<div class="section-card"><h3 class="section-title">Атакуемая модель</h3><ul class="pretty-list">')
    problem_type = desc.get("problem_type").replace("_", " ").capitalize()
    problem_type_ru = problem_types[problem_type]
    s1.append(f"<li><span class='param-key'>Задача:</span> {problem_type_ru} ({problem_type})</li><hr>")
    s1.append(f"<li><span class='param-key'>Модель:</span> {desc.get('model_name')}</li><hr>")
    s1.append('<li><details><summary><span class="param-key">Параметры</span></summary><ul class="param-list">')
    for line in flatten_param_dict(model_params, 0):
        indent_level = (len(line) - len(line.lstrip())) // 4
        style = f"margin-left:{indent_level*18}px;"
        if ":" in line:
            k, v = line.split(":", 1)
            if v.strip() == "":
                s1.append(f'<li style="{style}"><span class="param-key">{k.strip()}:</span></li><hr>')
            else:
                s1.append(
                    f'<li style="{style}"><span class="param-key">{k.strip()}:</span> <span class="param-val">{v.strip()}</span></li><hr>'
                )
        else:
            s1.append(f'<li style="{style}">{line}</li><hr>')
    s1.append("</ul></details></li></ul></div>")

    # Data info
    s1.append('<div class="section-card"><h3 class="section-title">Данные</h3><ul class="pretty-list">')
    s1.append(f"<li><span class='param-key'>Загрузчик данных:</span> {desc.get('dataset_loader_name')}</li><hr>")
    s1.append(
        '<li><details><summary><span class="param-key">Параметры загрузчика данных</span></summary><ul class="param-list">'
    )
    for line in flatten_param_dict(dataloader_params, 0):
        indent_level = (len(line) - len(line.lstrip())) // 4
        style = f"margin-left:{indent_level*18}px;"
        if ":" in line:
            k, v = line.split(":", 1)
            if v.strip() == "":
                s1.append(f'<li style="{style}"><span class="param-key">{k.strip()}:</span></li><hr>')
            else:
                s1.append(
                    f'<li style="{style}"><span class="param-key">{k.strip()}:</span> <span class="param-val">{v.strip()}</span></li><hr>'
                )
        else:
            s1.append(f'<li style="{style}">{line}</li><hr>')
    s1.append("</ul></details></li></ul></div>")

    # Attack info
    s1.append('<div class="section-card"><h3 class="section-title">Информация об атаке</h3><ul class="pretty-list">')
    s1.append(f"<li><span class='param-key'>Атака:</span> {experiments.get('attack')}</li><hr>")
    s1.append(
        '<li><details><summary><span class="param-key">Зафиксированные параметры</span></summary><ul class="param-list">'
    )
    for line in flatten_param_dict(experiments.get("fixed_attack_params", {}), 0):
        indent_level = (len(line) - len(line.lstrip())) // 4
        style = f"margin-left:{indent_level*18}px;"
        if ":" in line:
            k, v = line.split(":", 1)
            if v.strip() == "":
                s1.append(f'<li style="{style}"><span class="param-key">{k.strip()}:</span></li><hr>')
            else:
                s1.append(
                    f'<li style="{style}"><span class="param-key">{k.strip()}:</span> <span class="param-val">{v.strip()}</span></li><hr>'
                )
        else:
            s1.append(f'<li style="{style}">{line}</li><hr>')
    s1.append("</ul></details></li>")
    s1.append(
        f"<li><span class='param-key'>Переменный параметр:</span> {experiments.get('variable_param_name')}</li><hr>"
    )
    s1.append(
        '<li><details><summary><span class="param-key">Значения переменного параметра</span></summary><div class="param-values">'
    )
    param_values = list(experiments.get("metrics", {}).keys())
    param_values = try_make_num(param_values, sort=sort_param_values)
    for param_value in param_values:
        s1.append(f'<div class="param-item">{format_value(param_value)}</div>')
    s1.append("</div></details></li></ul></div>")
    s1.append("</div>")

    # Section 2: Metrics table
    s2 = []
    s2.append('<h2 class="section-title">Метрики</h2>')
    s2.append('<div class="section-card section-metrics section-wide">')
    s2.append(make_tables(report, metrics_meta))
    s2.append("</div>")

    # Section 3: plots, responsive columns
    s3 = []
    s3.append('<h2 class="section-title">Графики</h2>')
    s3.append('<div class="section-card section-plots section-wide"><div class="plots-grid">')
    for plot_path in plots:
        metric_name = os.path.splitext(os.path.basename(plot_path))[0]
        s3.append(
            f'<div class="plot-img"><img src="{plot_path}" alt="{metric_name}" style="max-width:100%; border-radius:12px;"></div>'
        )
    s3.append("</div></div>")
    s3.append("</div>")  # close global-card

    # CSS for layout, cards, grid, lists, etc.
    css = """
    <style>
    body { background: #f7f8fa; color: #243b44; font-family: Arial, sans-serif; margin: 0; padding: 0; }
    .global-card { background: #fff; border-radius: 32px; box-shadow: 0 4px 32px #0002; padding: 32px; max-width: 1140px; margin: 40px auto; box-sizing: border-box; overflow-x: hidden; }
    h1 { text-align: center; }
    .section-title { text-align: left; margin-left: 8px; }
    .section-row { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 24px; margin-bottom: 32px; }
    .section-card { background: #fafdff; border-radius: 18px; box-shadow: 0 2px 12px #0001; padding: 24px 18px 18px 18px; flex: 1 1 0; min-width: 300px; max-width: 360px; box-sizing: border-box; }
    .section-wide { max-width: 100%; width: 100%; margin: 0 auto; box-sizing: border-box; }
    .section-card h3 { margin-top: 0; }
    .param-list { margin: 0 0 0 12px; padding: 0; list-style: none; }
    .param-key { font-weight: 600; color: #2b4f66; }
    .param-val { color: #4f6d7a; }
    .pretty-list { margin: 0; padding: 0; list-style: none; }
    .pretty-list li { padding: 8px 0 8px 0; margin: 0; }
    .pretty-list hr { border: none; border-top: 1px solid #e6eef5; margin: 0; }
    .metrics-table { width: 100%; border-collapse: collapse; margin: 0 auto; background: #fafdff; border-radius: 12px; overflow: hidden; box-sizing: border-box; }
    .metrics-table th, .metrics-table td { border: 1px solid #d9dbe0; padding: 12px 18px; text-align: center; }
    .metrics-table th { background: #e6eef5; color: #2b4f66; font-weight: 700; }
    .metrics-table tr:nth-child(even) { background: #f3f6fa; }
    .metrics-table tr:nth-child(odd) { background: #fafdff; }
    .section-metrics, .section-plots { margin: 0 auto; max-width: 100%; box-sizing: border-box; }
    .plots-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 32px; justify-items: center; width: 100%; box-sizing: border-box; }
    .plot-img { background: #ffffff; border-radius: 12px; box-shadow: 0 1px 6px #0001; padding: 18px; display: flex; align-items: center; justify-content: center; width: 100%; box-sizing: border-box; }
    .param-values { margin-left: 18px; }
    .param-item { padding: 4px 0; color: #4f6d7a; }
    details { margin-bottom: 8px; }
    summary { cursor: pointer; font-weight: 600; color: #2b4f66; padding: 4px 0; }
    @media (max-width: 1200px) {
        .global-card { max-width: 98vw; }
        .section-card { max-width: 98vw; }
        .section-metrics, .section-plots { max-width: 98vw; }
        .plots-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 900px) {
        .section-row { flex-direction: column; align-items: center; }
        .section-card { max-width: 98vw; }
        .section-metrics, .section-plots { max-width: 98vw; }
        .plots-grid { grid-template-columns: 1fr; }
    }
    </style>
    """

    html = f'<html><head><meta charset="utf-8"><link rel="icon" href="https://mlm.intra.ispras.ru/assets/favicon.ico"><title>Отчет устойчивости</title>{css}</head><body>'
    html += "\n".join(s1)
    html += "\n".join(s2)
    html += "\n".join(s3)
    html += "</body></html>"
    return html


def main():
    """Main function to run the report generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate HTML report from robustness evaluation JSON data")
    parser.add_argument("-i", "--input", default="report_dict.json", help="Path to input JSON file")
    parser.add_argument("-o", "--output", default="robustness_report.html", help="Output HTML file path")
    parser.add_argument("-p", "--plots-path", default=".", help="Path to output plots directory")
    parser.add_argument(
        "-PTp",
        "--problem-types-path",
        default="./problem_types.json",
        help="Path to problem types JSON file",
    )
    parser.add_argument(
        "-m",
        "--metrics-path",
        default="./metrics.json",
        help="Path to metrics metadata JSON file",
    )

    args = parser.parse_args()

    report = read_json(args.input)
    if report is None:
        return

    problem_types = read_json(args.problem_types_path)
    if problem_types is None:
        return

    metrics_meta = read_json(args.metrics_path)
    if metrics_meta is None:
        return
    desc = report.get("desc", {})
    problem_type = desc.get("problem_type", "")
    metrics_meta = metrics_meta[problem_type]

    if not os.path.exists(args.plots_path):
        os.mkdir(args.plots_path)

    # generate plots
    plots = plot_metrics(report, args.plots_path, metrics_meta)

    # make plot paths relative
    for idx in range(len(plots)):
        plots[idx] = os.path.relpath(plots[idx], start=os.path.dirname(args.output))

    # build html
    html = build_report_html(report, plots, problem_types, metrics_meta)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report generated: {args.output}")
    for p in plots:
        print(f"Plot: {p}")


if __name__ == "__main__":
    main()
