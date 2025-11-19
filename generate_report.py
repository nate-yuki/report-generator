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
    plots: list[str] | list[list[str]],
    problem_types: dict[str, str],
    metrics_meta: dict[str, dict[str, Any]],
    attacks_type: str | None = None,
    sort_param_values: bool = False
) -> str:
    """Build HTML report string from the given report dictionary and list of plot paths."""
    desc = report["desc"]
    experiments = report["experiments"]

    # If experiments is a list -> verbose format
    verbose = isinstance(experiments, list)

    # Section 1: grouped info, 3 columns
    model_params = desc.get("model_parameters", {})
    dataloader_params = desc.get("dataloader_parameters", {})

    s1 = []
    s1.append('<div class="global-card">')

    if verbose:
        goal_ru = "уклонения" if attacks_type is None or attacks_type == "evasion" else "на приватность"
        s1.append(f"<h1>Отчет устойчивости {desc.get('model_name')} к атакам {goal_ru}</h1>")
    else:
        s1.append(f"<h1>Отчет устойчивости {desc.get('model_name')} к атаке {experiments.get('attack')}</h1>")
    s1.append('<details open><summary class="section-summary">Обзор</summary>')
    s1.append('<div class="card-grid">')
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

    # Experiments block(s)
    if not verbose:
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
    else:
        s1.append('</div>')
        s1.append('<div class="experiments-grid">')
        for idx, exp in enumerate(experiments, start=1):
            s1.append(f'<div class="section-card experiment-card"><h3 class="section-title">Эксперимент {idx}</h3><ul class="pretty-list">')
            s1.append(f"<li><span class='param-key'>Атака:</span> {exp.get('attack')}</li><hr>")
            s1.append('<li><details><summary><span class="param-key">Зафиксированные параметры</span></summary><ul class="param-list">')
            for line in flatten_param_dict(exp.get("fixed_attack_params", {}), 0):
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
            s1.append('</ul></details></li>')
            s1.append(f"<li><span class='param-key'>Переменный параметр:</span> {exp.get('variable_param_name')}</li><hr>")
            s1.append('<li><details><summary><span class="param-key">Значения переменного параметра</span></summary><div class="param-values">')
            param_values = list(exp.get("metrics", {}).keys())
            param_values = try_make_num(param_values, sort=sort_param_values)
            for param_value in param_values:
                s1.append(f'<div class="param-item">{format_value(param_value)}</div>')
            s1.append('</div></details></li></ul></div>')
        s1.append('</div>')

    # Close section 1
    s1.append("</details>")

    # Section 2: Metrics table(s)
    s2 = []
    s2.append('<details open><summary class="section-summary">Метрики</summary>')
    if verbose:
        for idx, exp in enumerate(experiments, start=1):
            s2.append(f'<div class="section-card section-metrics"><h3 class="section-title">Эксперимент {idx}: {exp.get("attack")}</h3>')
            s2.append(make_tables({"desc": desc, "experiments": exp}, metrics_meta))
            s2.append('</div>')
    else:
        s2.append('<div class="section-card section-metrics section-wide">')
        s2.append(make_tables(report, metrics_meta))
        s2.append("</div>")

    # Close section 2
    s2.append('</details>')

    # Section 3: plots, grouped by experiment
    s3 = []
    s3.append('<details open><summary class="section-summary">Графики</summary>')
    if verbose:
        for idx, (exp, exp_plots) in enumerate(zip(experiments, plots), start=1):
            s3.append(f'<div class="section-card section-plots"><h3 class="section-title">Эксперимент {idx}: {exp.get("attack")}</h3><div class="plots-grid">')
            for plot_path in exp_plots:
                metric_name = os.path.splitext(os.path.basename(plot_path))[0]
                s3.append(
                    f'<div class="plot-img"><img src="{plot_path}" alt="{metric_name}" style="max-width:100%; border-radius:12px;"></div>'
                )
            s3.append('</div></div>')
    else:
        s3.append('<div class="section-card section-plots section-wide"><div class="plots-grid">')
        for plot_path in plots:
            metric_name = os.path.splitext(os.path.basename(plot_path))[0]
            s3.append(
                f'<div class="plot-img"><img src="{plot_path}" alt="{metric_name}" style="max-width:100%; border-radius:12px;"></div>'
            )
        s3.append("</div></div>")

    # Close section 3
    s3.append('</details>')

    # Section 4: Comments (auto-generated quantitative remarks)
    s4 = []
    if verbose:
        s4.append('<details open><summary class="section-summary">Комментарии</summary>')
        s4.append('<div class="section-card section-comments section-wide">')
        for idx, exp in enumerate(experiments, start=1):
            if idx > 1:
                s4.append('<hr class="section-separator">')
            s4.append(f'<h3 class="section-subtitle">Эксперимент {idx}: {exp.get("attack")}</h3>')
            s4.append('<ul class="pretty-list">')
            exp_metrics = ensure_metric_floats(exp.get("metrics", {}))
            comments_added = False
            # last param value
            if exp_metrics:
                last_key = list(exp_metrics.keys())[-1]
                last_vals = exp_metrics[last_key]
                for mname, meta in (metrics_meta.items() if isinstance(metrics_meta, dict) else []):
                    # Skip if this is a Clean_ metric
                    if mname.lower().startswith("clean_"):
                        continue

                    higher_better = meta.get("higher_better")
                    if higher_better is None:
                        continue

                    metric_key = mname
                    clean_key = f"Clean_{metric_key}"

                    # find exact key in last_vals ignoring case
                    found_metric = None
                    found_clean = None
                    for k in last_vals.keys():
                        if k.lower() == metric_key.lower():
                            found_metric = k
                        if k.lower() == clean_key.lower():
                            found_clean = k

                    if found_metric and found_clean:
                        clean_val = exp_metrics[list(exp_metrics.keys())[0]].get(found_clean)
                        last_val = last_vals.get(found_metric)
                        if clean_val is None or last_val is None:
                            continue

                        comments_added = True
                        # percent change
                        try:
                            pct = ((last_val - clean_val) / (abs(clean_val) + 1e-12)) * 100.0
                        except Exception:
                            pct = 0.0
                        action = "увеличилась" if last_val > clean_val else "уменьшилась"
                        direction = "увеличение" if last_val > clean_val else "уменьшение"
                        rng = meta.get("range")
                        rng_str = ""
                        if isinstance(rng, list):
                            left = "-∞" if rng[0] is None else str(rng[0])
                            right = "+∞" if rng[1] is None else str(rng[1])
                            better_direction = "больше" if higher_better else "меньше"
                            rng_str = f"(диапазон: {left} – {right}, {better_direction} — лучше)"
                        s4.append(f"<li>Метрика <strong>{meta.get('name', metric_key)}</strong> {rng_str} {action} с {format_value(clean_val, 2)} до {format_value(last_val, 2)} ({direction} на {abs(pct):.2f}%)</li><hr>")

            if not comments_added:
                s4.append("<li>Нет метрик для анализа</li><hr>")
            s4.append('</ul>')
        s4.append('</div>')

        # Close section 4
        s4.append('</details>')

    # Section 5: Notations (metric descriptions)
    s5 = []
    if verbose:
        s5.append('<details open><summary class="section-summary">Обозначения</summary>')
        s5.append('<div class="section-card section-notations section-wide"><ul class="pretty-list">')
        for mname, meta in (metrics_meta.items() if isinstance(metrics_meta, dict) else []):
            desc_text = meta.get("description", "").rstrip(".")
            rng = meta.get("range")
            if isinstance(rng, list):
                left = "−∞" if rng[0] is None else str(rng[0])
                right = "+∞" if rng[1] is None else str(rng[1])
                higher_better = meta.get("higher_better")
                higher_better_comment = ""
                if higher_better is not None:
                    higher_better_comment = f", {'больше' if higher_better else 'меньше'} — лучше"
                rng_text = f"Диапазон: {left} – {right}{higher_better_comment}."
            else:
                rng_text = ""
            s5.append(f"<li><strong>{meta.get('name', mname)}:</strong> {desc_text}. {rng_text}</li><hr>")
        s5.append('</ul></div>')

        # Close section 5
        s5.append('</details>')

    # Close global-card
    s5.append('</div>')  

    base_css = """
    body { background: #f7f8fa; color: #243b44; font-family: Arial, sans-serif; margin: 0; padding: 0; }
    .global-card { background: #fff; border-radius: 32px; box-shadow: 0 4px 32px #0002; padding: 32px; max-width: 1140px; margin: 40px auto; box-sizing: border-box; overflow-x: hidden; }
    h1 { text-align: center; }
    .section-title { text-align: left; margin: 0 0 16px 8px; }

    .section-card { background: #fafdff; border-radius: 18px; box-shadow: 0 2px 12px #0001; padding: 24px 18px 18px 18px; flex: 1 1 0; min-width: 300px; box-sizing: border-box; }
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
    .metrics-table th:first-child { width: 200px; }
    .metrics-table tr:nth-child(even) { background: #f3f6fa; }
    .metrics-table tr:nth-child(odd) { background: #fafdff; }
    .section-metrics, .section-plots { margin: 0 auto; max-width: 100%; box-sizing: border-box; }
    .section-metrics + .section-metrics,
    .section-plots + .section-plots { margin-top: 24px; }
    h2.section-title { margin-top: 48px; margin-bottom: 24px; }
    .section-subtitle, 
    .section-card h3.section-title { 
        font-size: 1.3em;
        margin: 16px 0;
        color: #2b4f66;
    }
    hr.section-separator {
        border: none;
        border-top: 2px solid #e6eef5;
        margin: 24px 0;
    }
    .section-summary {
        font-size: 1.5em;
        font-weight: bold;
        color: #2b4f66;
        margin: 24px 0;
        cursor: pointer;
        user-select: none;
    }
    details {
        margin: 0;
    }
    details > summary {
        list-style: none;
    }
    details > summary::-webkit-details-marker {
        display: none;
    }
    .section-summary::before {
        content: "▶";
        display: inline-block;
        width: 20px;
        color: #2b4f66;
        transform: rotate(0);
        transition: transform 0.2s;
        margin-right: 8px;
    }
    details > summary:not(.section-summary)::before {
        content: "▶";
        display: inline-block;
        width: 20px;
        color: #2b4f66;
        transform: rotate(0);
        transition: transform 0.2s;
    }
    details[open] > summary::before {
        transform: rotate(90deg);
    }
    .plots-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 32px; justify-items: center; width: 100%; box-sizing: border-box; }
    .experiments-grid { 
        display: grid; 
        grid-template-columns: repeat(3, minmax(240px, 1fr)); 
        gap: 24px; 
        margin: 24px 0;
        justify-content: center;
    }
    .experiments-grid > .section-card {
        margin: 0 auto;
        width: 100%;
    }
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
    """

    # Brief format rules: 3 columns when wide, 2 columns for first two cards and full width for the third when medium,
    # and 1 column on small screens
    brief_css = """
    .card-grid {
        display: grid;
        gap: 24px;
        margin-bottom: 32px;
        grid-template-columns: repeat(3, minmax(300px, 1fr));
        align-items: start;
    }
    @media (max-width: 1200px) and (min-width: 901px) {
        .card-grid { grid-template-columns: repeat(2, minmax(300px, 1fr)); }
        .card-grid > .section-card:nth-child(3) { grid-column: 1 / -1; }
    }
    @media (max-width: 900px) {
        .card-grid { grid-template-columns: 1fr; }
    }
    """

    # Verbose format rules: top row contains two cards (model + data) that split the width evenly when possible
    # experiments-grid remains a separate block below and keeps its 3-column behavior
    verbose_css = """
    .card-grid {
        display: grid;
        gap: 24px;
        margin-bottom: 32px;
        grid-template-columns: repeat(2, minmax(300px, 1fr));
        align-items: start;
    }
    @media (max-width: 900px) {
        .card-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 1200px) {
        .experiments-grid { grid-template-columns: repeat(2, minmax(240px, 1fr)); }
    }
    @media (max-width: 900px) {
        .experiments-grid { grid-template-columns: 1fr; }
    }
    """

    css = f"<style>\n{base_css}\n{verbose_css if verbose else brief_css}\n</style>"

    html = f'<html><head><meta charset="utf-8"><link rel="icon" href="https://mlm.intra.ispras.ru/assets/favicon.ico"><title>Отчет устойчивости</title>{css}</head><body>'
    html += "\n".join(s1)
    html += "\n".join(s2)
    html += "\n".join(s3)
    html += "\n".join(s4)
    html += "\n".join(s5)
    html += "</body></html>"
    return html


def make_plots(
    report_dict: dict[str, Any],
    metrics_meta: dict[str, dict[str, Any]],
    plots_dir: Path | str,
    html_dir: Path | str
) -> list[str] | list[list[str]]:
    """Generate plots for the report and return list of plot paths."""
    experiments = report_dict.get("experiments")
    if isinstance(experiments, list):
        # Verbose report
        plots_list = []
        for experiment in experiments:
            experiment_report = {"desc": report_dict.get("desc", {}), "experiments": experiment}
            plot_path = os.path.join(plots_dir, f"{experiment['attack']}")
            if not os.path.exists(plot_path):
                os.mkdir(plot_path)
            experiment_plots = plot_metrics(experiment_report, plot_path, metrics_meta)
            for idx in range(len(experiment_plots)):
                experiment_plots[idx] = os.path.relpath(experiment_plots[idx], start=html_dir)
            plots_list.append(experiment_plots)
        # check Clean_ consistency across experiments
        clean_values = {}
        for experiment in experiments:
            for param_val, metrics in experiment.get("metrics", {}).items():
                for metric_name, metric_value in metrics.items():
                    if metric_name.lower().startswith("clean_"):
                        if metric_name not in clean_values:
                            clean_values[metric_name] = metric_value
                        else:
                            if abs(clean_values[metric_name] - metric_value) > 1e-9:
                                print(f"Warning: Clean_{metric_name} values differ across experiments/attack params: "
                                      f"{clean_values[metric_name]} != {metric_value}")
    else:
        # Brief report
        plots_list = plot_metrics(report_dict, plots_dir, metrics_meta)
        for idx, plot in enumerate(plots_list):
            plots_list[idx] = os.path.relpath(plot, start=html_dir)
    return plots_list


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
    plots = make_plots(report, metrics_meta, args.plots_path, os.path.dirname(args.output))

    # build html
    html = build_report_html(report, plots, problem_types, metrics_meta)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report generated: {args.output}")
    for p in plots:
        print(f"Plot: {p}")


if __name__ == "__main__":
    main()
