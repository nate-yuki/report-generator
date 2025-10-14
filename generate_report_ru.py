#!/usr/bin/env python3
"""
HTML Report Generator for Deep Learning Model Robustness Evaluation (Russian Version)

Этот скрипт генерирует комплексный HTML отчет из JSON данных, содержащих
результаты оценки робастности моделей глубокого обучения.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


class RobustnessReportGenerator:
    def __init__(self, json_data):
        """Инициализация генератора отчетов с JSON данными."""
        self.data = json_data
        self.experiments_df = pd.DataFrame(json_data['experiments'])
        
    def generate_html_report(self, output_file="robustness_report_ru.html"):
        """Генерация полного HTML отчета."""
        html_content = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Отчет об оценке робастности модели глубокого обучения</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        {self._generate_header()}
        {self._generate_model_info_section()}
        {self._generate_visualizations_section()}
        {self._generate_results_tables_section()}
        {self._generate_footer()}
    </div>
</body>
</html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Отчет успешно сгенерирован: {output_file}")
        return output_file
    
    def _get_css_styles(self):
        """Возвращает CSS стили для отчета."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            border-radius: 10px;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        .header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 3px solid #007bff;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #007bff;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            color: #6c757d;
            font-size: 1.2em;
        }
        
        .section {
            margin-bottom: 40px;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .section-title {
            color: #007bff;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
        }
        
        .model-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .model-info .section-title {
            color: white;
            border-bottom-color: rgba(255,255,255,0.3);
        }
        
        .visualizations {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        
        .visualizations .section-title {
            color: white;
            border-bottom-color: rgba(255,255,255,0.3);
        }
        
        .results {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }
        
        .results .section-title {
            color: white;
            border-bottom-color: rgba(255,255,255,0.3);
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .info-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 8px;
            backdrop-filter: blur(10px);
        }
        
        .info-card h3 {
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .info-card ul {
            list-style: none;
        }
        
        .info-card li {
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        
        .info-card li:last-child {
            border-bottom: none;
        }
        
        .chart-container {
            background: rgba(255,255,255,0.9);
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: rgba(255,255,255,0.9);
            border-radius: 8px;
            overflow: hidden;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(0,0,0,0.1);
            color: #007bff;
            font-weight: 500;
        }
        
        th {
            background-color: rgba(0,123,255,0.8);
            color: white;
            font-weight: 600;
        }
        
        tr:nth-child(even) {
            background-color: rgba(255,255,255,0.5);
        }
        
        tr:hover {
            background-color: rgba(255,255,255,0.8);
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
            margin-top: 30px;
        }
        
        .metric-highlight {
            font-weight: bold;
            color: #007bff;
        }
        
        .clear-data-row {
            border-top: 4px solid #2ECC71 !important;
            border-bottom: 4px solid #2ECC71 !important;
            background-color: rgba(46, 204, 113, 0.1) !important;
            font-weight: 600;
        }
        
        .clear-data-row td {
            color: #27AE60 !important;
            font-weight: 600;
        }
        
        @media print {
            body {
                background: white;
            }
            .container {
                box-shadow: none;
                margin: 0;
            }
            .section {
                page-break-inside: avoid;
            }
        }
        """
    
    def _generate_header(self):
        """Генерация заголовка отчета."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
        <div class="header">
            <h1>Отчет об оценке робастности модели глубокого обучения</h1>
            <p class="subtitle">Создан {current_time}</p>
        </div>
        """
    
    def _generate_model_info_section(self):
        """Генерация секции с информацией о модели."""
        desc = self.data.get('desc', {})
        model = desc.get('model', 'Неизвестно')
        model_params = desc.get('model_parameters', {})
        
        # Получение уникальных атак и их параметров
        attacks = {}
        for exp in self.data['experiments']:
            attack = exp['attack']
            eps = exp.get('eps', 'Н/Д')
            if attack not in attacks:
                attacks[attack] = []
            if eps not in attacks[attack]:
                attacks[attack].append(eps)
        
        model_params_html = ""
        if model_params:
            model_params_html = "<ul>"
            for key, value in model_params.items():
                model_params_html += f"<li><strong>{key}:</strong> {value}</li>"
            model_params_html += "</ul>"
        else:
            model_params_html = "<p>Дополнительные параметры не указаны</p>"
        
        attacks_html = "<ul>"
        for attack, eps_values in attacks.items():
            attacks_html += f"<li><strong>{attack}:</strong> значения ε: {', '.join(map(str, eps_values))}</li>"
        attacks_html += "</ul>"
        
        return f"""
        <div class="section model-info">
            <h2 class="section-title">1. Конфигурация модели и экспериментов</h2>
            <div class="info-grid">
                <div class="info-card">
                    <h3>Информация о модели</h3>
                    <ul>
                        <li><strong>Модель:</strong> {model}</li>
                    </ul>
                    <h4 style="margin-top: 15px;">Параметры модели</h4>
                    {model_params_html}
                </div>
                <div class="info-card">
                    <h3>Методы атак и параметры</h3>
                    {attacks_html}
                    <p style="margin-top: 15px;"><em>Всего экспериментов: {len(self.data['experiments'])}</em></p>
                </div>
            </div>
        </div>
        """
    
    def _generate_visualizations_section(self):
        """Генерация секции с визуализациями данных."""
        # Создание интерактивных графиков
        plots_html = []
        
        # График 1: Сравнение метрик по атакам
        fig1 = self._create_metrics_comparison_plot()
        plots_html.append(f'<div class="chart-container">{fig1.to_html(include_plotlyjs=False, div_id="plot1")}</div>')
        
        # График 2: Линейные графики эпсилон против точности/ASR
        fig2 = self._create_epsilon_lineplot()
        plots_html.append(f'<div class="chart-container">{fig2.to_html(include_plotlyjs=False, div_id="plot2")}</div>')
        
        # График 3: Радарная диаграмма для сравнения атак
        fig3 = self._create_radar_chart()
        plots_html.append(f'<div class="chart-container">{fig3.to_html(include_plotlyjs=False, div_id="plot3")}</div>')
        
        return f"""
        <div class="section visualizations">
            <h2 class="section-title">2. Визуализация данных</h2>
            {''.join(plots_html)}
        </div>
        """
    
    def _create_metrics_comparison_plot(self):
        """Создание сравнительного графика метрик по различным атакам."""
        # Группировка по атакам и вычисление средних метрик
        attack_groups = self.experiments_df.groupby('attack').agg({
            'metrics': lambda x: {
                'acc': [m['acc'] for m in x],
                'asr': [m['asr'] for m in x],
                'f1': [m['f1'] for m in x]
            }
        })
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Точность', 'Успешность атаки', 'F1-мера'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Разделение чистых данных и атак
        attacks = []
        acc_values = []
        asr_values = []
        f1_values = []
        colors_acc = []
        colors_asr = []
        colors_f1 = []
        
        # Сначала добавляем чистые данные, если они есть
        clear_data = None
        if 'no_attack' in attack_groups.index:
            clear_data = attack_groups.loc['no_attack', 'metrics']
            attacks.append('Чистые данные')
            acc_values.append(sum(clear_data['acc']) / len(clear_data['acc']))
            asr_values.append(sum(clear_data['asr']) / len(clear_data['asr']))
            f1_values.append(sum(clear_data['f1']) / len(clear_data['f1']))
            colors_acc.append('rgba(46, 204, 113, 0.8)')  # Зеленый для чистых данных
            colors_asr.append('rgba(46, 204, 113, 0.8)')
            colors_f1.append('rgba(46, 204, 113, 0.8)')
        
        # Затем добавляем другие атаки
        for attack in attack_groups.index:
            if attack != 'no_attack':
                attacks.append(attack)
                metrics_data = attack_groups.loc[attack, 'metrics']
                acc_values.append(sum(metrics_data['acc']) / len(metrics_data['acc']))
                asr_values.append(sum(metrics_data['asr']) / len(metrics_data['asr']))
                f1_values.append(sum(metrics_data['f1']) / len(metrics_data['f1']))
                colors_acc.append('rgba(55, 128, 191, 0.7)')
                colors_asr.append('rgba(219, 64, 82, 0.7)')
                colors_f1.append('rgba(128, 177, 211, 0.7)')
        
        fig.add_trace(
            go.Bar(x=attacks, y=acc_values, name='Точность', marker_color=colors_acc),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=attacks, y=asr_values, name='Успешность атаки', marker_color=colors_asr),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=attacks, y=f1_values, name='F1-мера', marker_color=colors_f1),
            row=1, col=3
        )
        
        fig.update_layout(
            title_text="Средние метрики: Чистые данные против методов атак",
            showlegend=False,
            height=500
        )
        
        return fig
    
    def _create_epsilon_lineplot(self):
        """Создание линейных графиков, показывающих связь между значениями эпсилон и метриками."""
        # Подготовка данных для линейных графиков
        plot_data = []
        no_attack_data = {}
        
        # Проверка, используют ли какие-либо эксперименты (кроме no_attack) eps=0
        other_attacks_use_zero = False
        for _, row in self.experiments_df.iterrows():
            if row['attack'] != 'no_attack' and row['eps'] == 0:
                other_attacks_use_zero = True
                break
        
        for _, row in self.experiments_df.iterrows():
            attack = row['attack']
            eps = row['eps']
            metrics = row['metrics']
            
            if attack == 'no_attack':
                # Сохранение данных no_attack для горизонтальных линий
                no_attack_data = {
                    'acc': metrics['acc'],
                    'asr': metrics['asr'],
                    'f1': metrics['f1']
                }
                # Включение в данные графика только если другие атаки также используют eps=0
                if other_attacks_use_zero and eps == 0:
                    plot_data.append({
                        'attack': 'Чистые данные',
                        'eps': eps,
                        'acc': metrics['acc'],
                        'asr': metrics['asr'],
                        'f1': metrics['f1']
                    })
            else:
                plot_data.append({
                    'attack': attack,
                    'eps': eps,
                    'acc': metrics['acc'],
                    'asr': metrics['asr'],
                    'f1': metrics['f1']
                })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Создание подграфиков для каждой метрики
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Точность против ε', 'Успешность атаки против ε', 'F1-мера против ε'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Определение цветов для различных атак
        colors = {
            'pgd': '#FF6B6B',
            'ifgsm': '#4ECDC4',
            'fgsm': '#45B7D1',
            'cw': '#96CEB4',
            'bim': '#FFEAA7',
            'Чистые данные': '#2ECC71'  # Зеленый для чистых данных
        }
        
        metrics = ['acc', 'asr', 'f1']
        metric_names = ['Точность', 'Успешность атаки', 'F1-мера']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names), 1):
            # Построение линий для каждого метода атаки
            unique_attacks = df_plot['attack'].unique()
            
            for attack in unique_attacks:
                attack_data = df_plot[df_plot['attack'] == attack].sort_values('eps')
                color = colors.get(attack, '#95A5A6')
                
                # Использование разного стиля линий для чистых данных
                line_style = dict(color=color, width=4, dash='dot') if attack == 'Чистые данные' else dict(color=color, width=3)
                
                fig.add_trace(
                    go.Scatter(
                        x=attack_data['eps'],
                        y=attack_data[metric],
                        mode='lines+markers',
                        name=f'{attack}' if i == 1 else f'{attack}',
                        line=line_style,
                        marker=dict(size=10 if attack == 'Чистые данные' else 8, color=color),
                        showlegend=True if i == 1 else False,
                        legendgroup=attack
                    ),
                    row=1, col=i
                )
            
            # Добавление горизонтальной линии для no_attack (производительность чистых данных)
            # Добавляется только если чистые данные еще не построены как линия
            if no_attack_data and metric in no_attack_data and 'Чистые данные' not in unique_attacks:
                ref_value = no_attack_data[metric]
                max_eps = df_plot['eps'].max() if not df_plot.empty else 10
                
                fig.add_trace(
                    go.Scatter(
                        x=[0, max_eps],
                        y=[ref_value, ref_value],
                        mode='lines',
                        name='Базовая линия чистых данных' if i == 1 else 'Базовая линия чистых данных',
                        line=dict(color='#2ECC71', width=3, dash='dash'),
                        showlegend=True if i == 1 else False,
                        legendgroup='clear'
                    ),
                    row=1, col=i
                )
        
        # Обновление макета
        fig.update_layout(
            title_text="Производительность атак против значений эпсилон",
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Обновление подписей осей x и y
        for i in range(1, 4):
            fig.update_xaxes(title_text="Эпсилон (ε)", row=1, col=i)
        
        fig.update_yaxes(title_text="Точность", row=1, col=1)
        fig.update_yaxes(title_text="Успешность атаки", row=1, col=2)
        fig.update_yaxes(title_text="F1-мера", row=1, col=3)
        
        return fig
    
    def _create_radar_chart(self):
        """Создание радарной диаграммы для сравнения атак по всем метрикам."""
        # Вычисление средних метрик для каждой атаки
        attack_metrics = {}
        for _, row in self.experiments_df.iterrows():
            attack = row['attack']
            if attack not in attack_metrics:
                attack_metrics[attack] = {'acc': [], 'asr': [], 'f1': []}
            
            attack_metrics[attack]['acc'].append(row['metrics']['acc'])
            attack_metrics[attack]['asr'].append(row['metrics']['asr'])
            attack_metrics[attack]['f1'].append(row['metrics']['f1'])
        
        fig = go.Figure()
        
        # Сначала обработка чистых данных, если они есть
        if 'no_attack' in attack_metrics:
            clear_metrics = attack_metrics['no_attack']
            avg_acc = sum(clear_metrics['acc']) / len(clear_metrics['acc'])
            avg_asr = sum(clear_metrics['asr']) / len(clear_metrics['asr'])
            avg_f1 = sum(clear_metrics['f1']) / len(clear_metrics['f1'])
            
            fig.add_trace(go.Scatterpolar(
                r=[avg_acc, 1-avg_asr, avg_f1],  # Инвертирование ASR для лучшей визуализации
                theta=['Точность', 'Робастность (1-ASR)', 'F1-мера'],
                fill='toself',
                name='Чистые данные',
                line_color='#2ECC71',
                fillcolor='rgba(46, 204, 113, 0.2)'
            ))
        
        # Обработка других атак
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        color_idx = 0
        
        for attack, metrics in attack_metrics.items():
            if attack != 'no_attack':
                avg_acc = sum(metrics['acc']) / len(metrics['acc'])
                avg_asr = sum(metrics['asr']) / len(metrics['asr'])
                avg_f1 = sum(metrics['f1']) / len(metrics['f1'])
                
                color = colors[color_idx % len(colors)]
                color_idx += 1
                
                fig.add_trace(go.Scatterpolar(
                    r=[avg_acc, 1-avg_asr, avg_f1],  # Инвертирование ASR для лучшей визуализации
                    theta=['Точность', 'Робастность (1-ASR)', 'F1-мера'],
                    fill='toself',
                    name=attack,
                    line_color=color,
                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)'
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Сравнение методов атак с чистыми данными (радарная диаграмма)"
        )
        
        return fig
    
    def _generate_results_tables_section(self):
        """Генерация секции с таблицами результатов."""
        tables_html = []
        
        # Таблица 1: Полные результаты
        tables_html.append(self._create_complete_results_table())
        
        # Таблица 2: Результаты, сгруппированные по метрике
        tables_html.append(self._create_metrics_grouped_table())
        
        # Таблица 3: Результаты, сгруппированные по атаке
        tables_html.append(self._create_attack_grouped_table())
        
        return f"""
        <div class="section results">
            <h2 class="section-title">3. Детальные таблицы результатов</h2>
            {''.join(tables_html)}
        </div>
        """
    
    def _create_complete_results_table(self):
        """Создание таблицы с полными результатами."""
        table_html = """
        <h3>Полные результаты экспериментов</h3>
        <table>
            <thead>
                <tr>
                    <th>Атака</th>
                    <th>Эпсилон (ε)</th>
                    <th>Точность</th>
                    <th>Успешность атаки</th>
                    <th>F1-мера</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Сначала добавляем строку с чистыми данными, если она есть
        clear_data_rows = []
        attack_rows = []
        
        for _, row in self.experiments_df.iterrows():
            metrics = row['metrics']
            attack_name = 'Чистые данные' if row['attack'] == 'no_attack' else row['attack']
            row_class = 'clear-data-row' if row['attack'] == 'no_attack' else ''
            
            row_html = f"""
                <tr class="{row_class}">
                    <td>{attack_name}</td>
                    <td>{row['eps']}</td>
                    <td class="metric-highlight">{metrics['acc']:.3f}</td>
                    <td class="metric-highlight">{metrics['asr']:.3f}</td>
                    <td class="metric-highlight">{metrics['f1']:.3f}</td>
                </tr>
            """
            
            if row['attack'] == 'no_attack':
                clear_data_rows.append(row_html)
            else:
                attack_rows.append(row_html)
        
        # Добавляем сначала чистые данные, затем атаки
        table_html += ''.join(clear_data_rows)
        table_html += ''.join(attack_rows)
        
        table_html += """
            </tbody>
        </table>
        """
        
        return table_html
    
    def _create_metrics_grouped_table(self):
        """Создание таблиц, сгруппированных по метрикам."""
        metrics_html = "<h3>Результаты, сгруппированные по метрикам</h3>"
        
        for metric_name, metric_key in [('Точность', 'acc'), ('Успешность атаки', 'asr'), ('F1-мера', 'f1')]:
            metrics_html += f"""
            <h4>{metric_name}</h4>
            <table>
                <thead>
                    <tr>
                        <th>Атака</th>
                        <th>Эпсилон (ε)</th>
                        <th>{metric_name}</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            # Разделение чистых данных и атак
            clear_data_rows = []
            attack_rows = []
            
            for _, row in self.experiments_df.iterrows():
                attack_name = 'Чистые данные' if row['attack'] == 'no_attack' else row['attack']
                row_class = 'clear-data-row' if row['attack'] == 'no_attack' else ''
                
                row_html = f"""
                    <tr class="{row_class}">
                        <td>{attack_name}</td>
                        <td>{row['eps']}</td>
                        <td class="metric-highlight">{row['metrics'][metric_key]:.3f}</td>
                    </tr>
                """
                
                if row['attack'] == 'no_attack':
                    clear_data_rows.append((row['metrics'][metric_key], row_html))
                else:
                    attack_rows.append((row['metrics'][metric_key], row_html))
            
            # Сортировка строк атак по значению метрики по убыванию
            attack_rows.sort(key=lambda x: x[0], reverse=True)
            
            # Добавляем сначала чистые данные, затем отсортированные атаки
            for _, row_html in clear_data_rows:
                metrics_html += row_html
            for _, row_html in attack_rows:
                metrics_html += row_html
            
            metrics_html += """
                </tbody>
            </table>
            """
        
        return metrics_html
    
    def _create_attack_grouped_table(self):
        """Создание таблиц, сгруппированных по методам атак."""
        attacks_html = "<h3>Результаты, сгруппированные по методам атак</h3>"
        
        unique_attacks = self.experiments_df['attack'].unique()
        
        # Сначала обработка чистых данных, если они есть
        if 'no_attack' in unique_attacks:
            clear_data = self.experiments_df[self.experiments_df['attack'] == 'no_attack']
            
            attacks_html += f"""
            <h4>ЧИСТЫЕ ДАННЫЕ (базовая производительность)</h4>
            <table>
                <thead>
                    <tr>
                        <th>Эпсилон (ε)</th>
                        <th>Точность</th>
                        <th>Успешность атаки</th>
                        <th>F1-мера</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for _, row in clear_data.iterrows():
                metrics = row['metrics']
                attacks_html += f"""
                    <tr class="clear-data-row">
                        <td>{row['eps']}</td>
                        <td class="metric-highlight">{metrics['acc']:.3f}</td>
                        <td class="metric-highlight">{metrics['asr']:.3f}</td>
                        <td class="metric-highlight">{metrics['f1']:.3f}</td>
                    </tr>
                """
            
            attacks_html += """
                </tbody>
            </table>
            """
        
        # Обработка других атак
        for attack in unique_attacks:
            if attack != 'no_attack':
                attack_data = self.experiments_df[self.experiments_df['attack'] == attack]
                
                attacks_html += f"""
                <h4>{attack.upper()}</h4>
                <table>
                    <thead>
                        <tr>
                            <th>Эпсилон (ε)</th>
                            <th>Точность</th>
                            <th>Успешность атаки</th>
                            <th>F1-мера</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for _, row in attack_data.iterrows():
                    metrics = row['metrics']
                    attacks_html += f"""
                        <tr>
                            <td>{row['eps']}</td>
                            <td class="metric-highlight">{metrics['acc']:.3f}</td>
                            <td class="metric-highlight">{metrics['asr']:.3f}</td>
                            <td class="metric-highlight">{metrics['f1']:.3f}</td>
                        </tr>
                    """
                
                attacks_html += """
                    </tbody>
                </table>
                """
        
        return attacks_html
    
    def _generate_footer(self):
        """Генерация подвала отчета."""
        return """
        <div class="footer">
            <p>Отчет сгенерирован инструментом оценки робастности моделей глубокого обучения</p>
            <p>По вопросам или проблемам обращайтесь к команде разработчиков.</p>
        </div>
        """


def main():
    """Основная функция для запуска генератора отчетов."""
    parser = argparse.ArgumentParser(description='Генерация HTML отчета из JSON данных оценки робастности')
    parser.add_argument('input_file', help='Путь к входному JSON файлу')
    parser.add_argument('-o', '--output', default='robustness_report_ru.html', help='Путь к выходному HTML файлу')
    
    args = parser.parse_args()
    
    # Загрузка JSON данных
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Ошибка: Файл {args.input_file} не найден.")
        return
    except json.JSONDecodeError as e:
        print(f"Ошибка: Неверный JSON в {args.input_file}: {e}")
        return
    
    # Генерация отчета
    generator = RobustnessReportGenerator(data)
    output_file = generator.generate_html_report(args.output)
    
    print(f"\n🎉 Отчет успешно сгенерирован!")
    print(f"📄 Выходной файл: {output_file}")
    print(f"🌐 Открыть в браузере: file://{Path(output_file).absolute()}")
    print(f"📋 Для экспорта в PDF: Откройте в браузере и используйте Печать > Сохранить как PDF")


if __name__ == "__main__":
    main()