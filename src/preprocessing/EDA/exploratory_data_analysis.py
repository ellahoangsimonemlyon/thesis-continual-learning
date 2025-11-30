#!/usr/bin/env python3
"""
SCOTUS Dataset Exploratory Data Analysis
Generates a comprehensive HTML report with visualizations and temporal bin statistics
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import json
from datetime import datetime
import tiktoken

def load_data(filepath):
    """Load the cleaned SCOTUS dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['date_filed'] = pd.to_datetime(df['date_filed'], errors='coerce')
    print(f"Loaded {len(df):,} cases")
    return df

def load_bin_statistics(stats_path):
    """Load the bin statistics JSON file if it exists"""
    if stats_path and Path(stats_path).exists():
        with open(stats_path, 'r') as f:
            return json.load(f)
    return None

def count_tokens(text, encoding_name="cl100k_base"):
    """Count tokens in text using tiktoken"""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback: character-based estimation (~4 characters per token)
        return len(text) // 4

def calculate_token_stats(df):
    """Calculate token counts for the dataset"""
    print("Calculating tokens...")
    
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        method = "tiktoken (cl100k_base)"
        print("Using tiktoken for accurate token counting...")
    except Exception as e:
        method = "character-based estimation"
        print("Using character-based estimation (~4 chars per token)")
    
    df['token_count'] = df['opinion_text'].apply(count_tokens)
    total_tokens = df['token_count'].sum()
    
    df['year'] = df['date_filed'].dt.year
    tokens_by_year = df.groupby('year')['token_count'].sum().reset_index()
    tokens_by_year.columns = ['year', 'tokens']
    
    avg_tokens = df['token_count'].mean()
    median_tokens = df['token_count'].median()
    
    stats = {
        'total_tokens': total_tokens,
        'avg_tokens': avg_tokens,
        'median_tokens': median_tokens,
        'tokens_by_year': tokens_by_year,
        'counting_method': method
    }
    
    print(f"Total tokens: {total_tokens:,}")
    return stats, df

def generate_summary_stats(df):
    """Generate summary statistics"""
    stats = {
        'total_cases': len(df),
        'date_range': f"{df['date_filed'].min().strftime('%Y-%m-%d')} to {df['date_filed'].max().strftime('%Y-%m-%d')}",
        'years_span': (df['date_filed'].max().year - df['date_filed'].min().year),
        'avg_opinion_length': df['opinion_len'].mean(),
        'median_opinion_length': df['opinion_len'].median(),
        'cases_with_judges': df['judges'].notna().sum(),
        'cases_with_syllabus': df['syllabus'].notna().sum(),
        'cases_with_summary': df['summary'].notna().sum(),
        'merged_cases': (df['rows_in_group'] > 1).sum(),
        'avg_citations': df['citation_count'].mean(),
    }
    return stats

def plot_cases_over_time(df):
    """Create timeline visualization of cases filed"""
    df['year'] = df['date_filed'].dt.year
    yearly_counts = df.groupby('year').size().reset_index(name='count')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly_counts['year'],
        y=yearly_counts['count'],
        mode='lines+markers',
        name='Cases Filed',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4),
        hovertemplate='<b>Year:</b> %{x}<br><b>Cases:</b> %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Supreme Court Cases Filed Over Time',
        xaxis_title='Year',
        yaxis_title='Number of Cases',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def plot_opinion_length_distribution(df):
    """Create distribution plots for opinion lengths"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Opinion Length Distribution', 'Opinion Length by Decade'),
        specs=[[{'type': 'histogram'}, {'type': 'box'}]]
    )
    
    fig.add_trace(
        go.Histogram(
            x=df['opinion_len'],
            nbinsx=50,
            name='Opinion Length',
            marker_color='#2ca02c',
            hovertemplate='<b>Length Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    df['decade'] = (df['date_filed'].dt.year // 10) * 10
    decades = sorted(df['decade'].dropna().unique())
    
    for decade in decades:
        decade_data = df[df['decade'] == decade]['opinion_len']
        fig.add_trace(
            go.Box(
                y=decade_data,
                name=f"{int(decade)}s",
                boxmean='sd',
                hovertemplate='<b>Length:</b> %{y}<extra></extra>'
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Opinion Length (characters)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Decade", row=1, col=2)
    fig.update_yaxes(title_text="Opinion Length (characters)", row=1, col=2)
    
    fig.update_layout(
        height=500,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def plot_merged_cases_analysis(df):
    """Analyze cases that were merged from multiple rows"""
    merged_df = df[df['rows_in_group'] > 1].copy()
    merge_counts = merged_df['rows_in_group'].value_counts().sort_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=merge_counts.index,
        y=merge_counts.values,
        marker_color='#d62728',
        hovertemplate='<b>Original Rows:</b> %{x}<br><b>Cases:</b> %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Distribution of Merged Cases ({len(merged_df):,} total merged cases)',
        xaxis_title='Number of Original Rows Merged',
        yaxis_title='Number of Cases',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_metadata_completeness(df):
    """Visualize completeness of metadata fields"""
    metadata_cols = ['judges', 'attorneys', 'syllabus', 'summary', 
                     'history', 'arguments', 'headmatter']
    
    completeness = []
    for col in metadata_cols:
        if col in df.columns:
            pct = (df[col].notna().sum() / len(df)) * 100
            completeness.append({'Field': col.capitalize(), 'Completeness (%)': pct})
    
    completeness_df = pd.DataFrame(completeness)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=completeness_df['Field'],
        x=completeness_df['Completeness (%)'],
        orientation='h',
        marker_color='#9467bd',
        text=completeness_df['Completeness (%)'].round(1),
        texttemplate='%{text}%',
        textposition='outside',
        hovertemplate='<b>%{y}:</b> %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Metadata Field Completeness',
        xaxis_title='Completeness (%)',
        yaxis_title='',
        template='plotly_white',
        height=400,
        xaxis_range=[0, 105]
    )
    
    return fig

def plot_citation_analysis(df):
    """Analyze citation counts"""
    df_cited = df[df['citation_count'] > 0].copy()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Citation Count Distribution', 'Citations Over Time'),
        specs=[[{'type': 'histogram'}, {'type': 'scatter'}]]
    )
    
    fig.add_trace(
        go.Histogram(
            x=df_cited['citation_count'],
            nbinsx=50,
            marker_color='#ff7f0e',
            hovertemplate='<b>Citations:</b> %{x}<br><b>Cases:</b> %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    df['year'] = df['date_filed'].dt.year
    yearly_citations = df.groupby('year')['citation_count'].mean().reset_index()
    
    fig.add_trace(
        go.Scatter(
            x=yearly_citations['year'],
            y=yearly_citations['citation_count'],
            mode='lines+markers',
            marker=dict(size=4),
            line=dict(color='#ff7f0e'),
            hovertemplate='<b>Year:</b> %{x}<br><b>Avg Citations:</b> %{y:.1f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Citation Count", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_yaxes(title_text="Average Citation Count", row=1, col=2)
    
    fig.update_layout(
        height=500,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def plot_monthly_seasonality(df):
    """Analyze monthly patterns in case filings"""
    df['month'] = df['date_filed'].dt.month
    monthly_counts = df.groupby('month').size().reset_index(name='count')
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_counts['month_name'] = monthly_counts['month'].apply(lambda x: month_names[x-1])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_counts['month_name'],
        y=monthly_counts['count'],
        marker_color='#17becf',
        hovertemplate='<b>%{x}:</b> %{y:,} cases<extra></extra>'
    ))
    
    fig.update_layout(
        title='Seasonality: Cases Filed by Month',
        xaxis_title='Month',
        yaxis_title='Total Cases',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_token_analysis(df, token_stats):
    """Visualize token counts"""
    tokens_by_year = token_stats['tokens_by_year']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Total Tokens by Year', 'Token Distribution per Case'),
        specs=[[{'type': 'scatter'}, {'type': 'histogram'}]]
    )
    
    fig.add_trace(
        go.Scatter(
            x=tokens_by_year['year'],
            y=tokens_by_year['tokens'] / 1_000_000,
            mode='lines+markers',
            marker=dict(size=4),
            line=dict(color='#e377c2'),
            hovertemplate='<b>Year:</b> %{x}<br><b>Tokens:</b> %{y:.2f}M<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=df['token_count'],
            nbinsx=50,
            marker_color='#e377c2',
            hovertemplate='<b>Token Range:</b> %{x}<br><b>Cases:</b> %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_yaxes(title_text="Total Tokens (Millions)", row=1, col=1)
    fig.update_xaxes(title_text="Tokens per Case", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    fig.update_layout(
        height=500,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def plot_bin_visualizations(bin_stats):
    """Create visualizations for temporal bin statistics"""
    steps = [s['step'] for s in bin_stats]
    cases_train = [s['num_cases_train'] for s in bin_stats]
    cases_val = [s['num_cases_val'] for s in bin_stats]
    tokens_train = [s['tokens_train'] for s in bin_stats]
    tokens_val = [s['tokens_val'] for s in bin_stats]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cases per Training Step (Train/Val Split)',
            'Tokens per Training Step',
            'Average Tokens per Case by Step',
            'Total Cases Distribution'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # Cases per step (stacked)
    fig.add_trace(
        go.Bar(x=steps, y=cases_train, name='Training', marker_color='#667eea',
               hovertemplate='<b>Step %{x}</b><br>Training: %{y:,}<extra></extra>'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=steps, y=cases_val, name='Validation', marker_color='#ffa500',
               hovertemplate='<b>Step %{x}</b><br>Validation: %{y:,}<extra></extra>'),
        row=1, col=1
    )
    
    # Tokens per step (stacked)
    fig.add_trace(
        go.Bar(x=steps, y=[t/1_000_000 for t in tokens_train], name='Training Tokens',
               marker_color='#2ca02c', showlegend=False,
               hovertemplate='<b>Step %{x}</b><br>Training: %{y:.2f}M tokens<extra></extra>'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=steps, y=[t/1_000_000 for t in tokens_val], name='Validation Tokens',
               marker_color='#90ee90', showlegend=False,
               hovertemplate='<b>Step %{x}</b><br>Validation: %{y:.2f}M tokens<extra></extra>'),
        row=1, col=2
    )
    
    # Average tokens
    avg_tokens = [s['avg_tokens_per_case'] for s in bin_stats]
    fig.add_trace(
        go.Scatter(x=steps, y=avg_tokens, mode='lines+markers', name='Avg Tokens',
                  line=dict(color='#d62728', width=3), marker=dict(size=10), showlegend=False,
                  hovertemplate='<b>Step %{x}</b><br>Avg: %{y:,} tokens/case<extra></extra>'),
        row=2, col=1
    )
    
    # Total cases
    cases_total = [s['num_cases_total'] for s in bin_stats]
    fig.add_trace(
        go.Bar(x=steps, y=cases_total, marker_color='#9467bd', showlegend=False,
               hovertemplate='<b>Step %{x}</b><br>Total: %{y:,} cases<extra></extra>'),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Training Step", row=1, col=1)
    fig.update_yaxes(title_text="Number of Cases", row=1, col=1)
    fig.update_xaxes(title_text="Training Step", row=1, col=2)
    fig.update_yaxes(title_text="Tokens (Millions)", row=1, col=2)
    fig.update_xaxes(title_text="Training Step", row=2, col=1)
    fig.update_yaxes(title_text="Avg Tokens per Case", row=2, col=1)
    fig.update_xaxes(title_text="Training Step", row=2, col=2)
    fig.update_yaxes(title_text="Total Cases", row=2, col=2)
    
    fig.update_layout(
        height=900,
        showlegend=True,
        template='plotly_white',
        barmode='stack',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_bin_table_html(bin_stats):
    """Create HTML table for bin overview"""
    html = """
    <div style="overflow-x: auto; margin: 20px 0;">
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background: #667eea; color: white;">
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Step</th>
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Period</th>
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Years</th>
                    <th style="padding: 12px; text-align: right; border: 1px solid #ddd;">Total Cases</th>
                    <th style="padding: 12px; text-align: right; border: 1px solid #ddd;">Train</th>
                    <th style="padding: 12px; text-align: right; border: 1px solid #ddd;">Val</th>
                    <th style="padding: 12px; text-align: right; border: 1px solid #ddd;">Total Tokens</th>
                    <th style="padding: 12px; text-align: right; border: 1px solid #ddd;">Avg Tokens</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for s in bin_stats:
        html += f"""
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>{s['step']}</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{s['name']}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{s['years']}</td>
                    <td style="padding: 10px; text-align: right; border: 1px solid #ddd;">{s['num_cases_total']:,}</td>
                    <td style="padding: 10px; text-align: right; border: 1px solid #ddd;">{s['num_cases_train']:,}</td>
                    <td style="padding: 10px; text-align: right; border: 1px solid #ddd;">{s['num_cases_val']:,}</td>
                    <td style="padding: 10px; text-align: right; border: 1px solid #ddd;">{s['tokens_total']:,}</td>
                    <td style="padding: 10px; text-align: right; border: 1px solid #ddd;">{s['avg_tokens_per_case']:,}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    return html

def generate_html_report(df, output_path, bin_data=None):
    """Generate complete HTML report"""
    
    print("Generating summary statistics...")
    stats = generate_summary_stats(df)
    
    print("Calculating token counts...")
    token_stats, df = calculate_token_stats(df)
    
    print("Creating visualizations...")
    fig1 = plot_cases_over_time(df)
    fig2 = plot_opinion_length_distribution(df)
    fig3 = plot_merged_cases_analysis(df)
    fig4 = plot_metadata_completeness(df)
    fig5 = plot_citation_analysis(df)
    fig6 = plot_monthly_seasonality(df)
    fig7 = plot_token_analysis(df, token_stats)
    
    # Convert figures to HTML
    fig1_html = fig1.to_html(include_plotlyjs='cdn', div_id='fig1')
    fig2_html = fig2.to_html(include_plotlyjs=False, div_id='fig2')
    fig3_html = fig3.to_html(include_plotlyjs=False, div_id='fig3')
    fig4_html = fig4.to_html(include_plotlyjs=False, div_id='fig4')
    fig5_html = fig5.to_html(include_plotlyjs=False, div_id='fig5')
    fig6_html = fig6.to_html(include_plotlyjs=False, div_id='fig6')
    fig7_html = fig7.to_html(include_plotlyjs=False, div_id='fig7')
    
    # Temporal bins section (if available)
    bin_section = ""
    if bin_data:
        print("Adding temporal bin analysis...")
        bin_stats = bin_data['bins']
        summary = bin_data['summary']
        
        fig_bins = plot_bin_visualizations(bin_stats)
        fig_bins_html = fig_bins.to_html(include_plotlyjs=False, div_id='fig_bins')
        bin_table_html = create_bin_table_html(bin_stats)
        
        bin_section = f"""
    <div class="section">
        <h2>Temporal Bins for Continual Learning</h2>
        <p>The dataset has been split into 10 temporal bins based on landmark Supreme Court cases and constitutional eras. 
        Each bin represents a distinct period in Supreme Court history, with 90% allocated to training and 10% to validation.</p>
        
        <div class="summary-box">
            <h3>Overall Summary</h3>
            <div class="stats-grid">
                <div>
                    <strong>Total Bins:</strong> {summary['total_bins']}<br>
                    <strong>Total Cases:</strong> {summary['total_cases']:,}
                </div>
                <div>
                    <strong>Training Cases:</strong> {summary['total_train_cases']:,}<br>
                    <strong>Validation Cases:</strong> {summary['total_val_cases']:,}
                </div>
                <div>
                    <strong>Total Tokens:</strong> {summary['total_tokens']:,}<br>
                    <strong>({summary['total_tokens']/1_000_000:.1f}M tokens)</strong>
                </div>
            </div>
        </div>

        <h3>Bin Details</h3>
        {bin_table_html}

        <h3>Visualizations</h3>
        {fig_bins_html}

        <div class="info-box">
            <strong>About the Temporal Bins:</strong>
            <ul>
                <li><strong>Step 1 (1791-1896):</strong> Foundations and Reconstruction - Ends with Plessy v. Ferguson</li>
                <li><strong>Step 2 (1897-1919):</strong> Lochner-Early Speech Era</li>
                <li><strong>Step 3 (1920-1936):</strong> Incorporation begins, Pre-New Deal</li>
                <li><strong>Step 4 (1937-1953):</strong> New Deal and War Powers - Ends before Brown</li>
                <li><strong>Step 5 (1954-1969):</strong> Warren Court Rights Revolution</li>
                <li><strong>Step 6 (1970-1984):</strong> Burger Court Recalibration - Roe, Chevron</li>
                <li><strong>Step 7 (1985-1994):</strong> Early Rehnquist Court</li>
                <li><strong>Step 8 (1995-2004):</strong> Federalism Revival</li>
                <li><strong>Step 9 (2005-2014):</strong> Roberts Court Part I</li>
                <li><strong>Step 10 (2015-2023):</strong> Roberts Court Part II - Modern Era</li>
            </ul>
        </div>

        <div class="note-box">
            <strong>Research Applications:</strong>
            These bins enable continual learning experiments where you can train sequentially on each temporal period,
            measure catastrophic forgetting between eras, test model adaptation to evolving legal reasoning,
            validate on held-out cases from each period, and compare performance across different training strategies.
        </div>
    </div>
        """
    
    # Create HTML document
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SCOTUS Dataset - Exploratory Data Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
            font-weight: 600;
        }}
        .header p {{
            margin: 5px 0;
            font-size: 1em;
            opacity: 0.9;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }}
        .stat-card .value {{
            font-size: 2em;
            font-weight: 600;
            color: #333;
        }}
        .stat-card .subvalue {{
            color: #999;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .section {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            font-weight: 600;
        }}
        .section h3 {{
            color: #555;
            margin-top: 30px;
            font-weight: 600;
        }}
        .summary-box, .info-box, .note-box {{
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
        }}
        .summary-box {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
        }}
        .info-box {{
            background: #e8f4f8;
            border-left: 4px solid #2196f3;
        }}
        .note-box {{
            background: #fff9e6;
            border-left: 4px solid #ffc107;
        }}
        .info-box ul, .note-box ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .footer {{
            text-align: center;
            color: #999;
            margin-top: 50px;
            padding: 20px;
            font-size: 0.9em;
        }}
        .plotly-graph-div {{
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Supreme Court Cases Dataset</h1>
        <p>Exploratory Data Analysis</p>
        <p style="font-size: 0.9em; opacity: 0.85;">Harvard COLD Cases Dataset - Supreme Court Opinions</p>
        <p style="font-size: 0.85em; opacity: 0.7;">Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <h3>Total Cases</h3>
            <div class="value">{stats['total_cases']:,}</div>
        </div>
        <div class="stat-card">
            <h3>Date Range</h3>
            <div class="value" style="font-size: 1.2em;">{stats['date_range'].split(' to ')[0]}</div>
            <div class="subvalue">to {stats['date_range'].split(' to ')[1]}</div>
        </div>
        <div class="stat-card">
            <h3>Years Covered</h3>
            <div class="value">{stats['years_span']}</div>
            <div class="subvalue">years</div>
        </div>
        <div class="stat-card">
            <h3>Total Tokens</h3>
            <div class="value">{token_stats['total_tokens']/1_000_000:.1f}M</div>
            <div class="subvalue">{token_stats['total_tokens']:,} tokens</div>
        </div>
        <div class="stat-card">
            <h3>Avg Tokens per Case</h3>
            <div class="value">{int(token_stats['avg_tokens']):,}</div>
            <div class="subvalue">median: {int(token_stats['median_tokens']):,}</div>
        </div>
        <div class="stat-card">
            <h3>Avg Opinion Length</h3>
            <div class="value">{int(stats['avg_opinion_length']):,}</div>
            <div class="subvalue">characters</div>
        </div>
        <div class="stat-card">
            <h3>Cases with Judges Info</h3>
            <div class="value">{stats['cases_with_judges']:,}</div>
            <div class="subvalue">{(stats['cases_with_judges']/stats['total_cases']*100):.1f}% of total</div>
        </div>
        <div class="stat-card">
            <h3>Merged Cases</h3>
            <div class="value">{stats['merged_cases']:,}</div>
            <div class="subvalue">had multiple source rows</div>
        </div>
        <div class="stat-card">
            <h3>Cases with Syllabus</h3>
            <div class="value">{stats['cases_with_syllabus']:,}</div>
            <div class="subvalue">{(stats['cases_with_syllabus']/stats['total_cases']*100):.1f}% of total</div>
        </div>
        <div class="stat-card">
            <h3>Avg Citation Count</h3>
            <div class="value">{stats['avg_citations']:.1f}</div>
            <div class="subvalue">citations per case</div>
        </div>
    </div>

    <div class="section">
        <h2>Cases Filed Over Time</h2>
        <p>Distribution of Supreme Court cases filed over time. Notable spikes may correspond to periods of increased litigation or expanded court jurisdiction.</p>
        {fig1_html}
    </div>

    <div class="section">
        <h2>Opinion Length Analysis</h2>
        <p>Analysis of opinion text length showing distribution and trends across decades.</p>
        {fig2_html}
    </div>

    <div class="section">
        <h2>Merged Cases Analysis</h2>
        <p>During data cleaning, {stats['merged_cases']:,} cases had multiple rows with the same case name and date. These were merged to create a single authoritative record.</p>
        {fig3_html}
    </div>

    <div class="section">
        <h2>Metadata Completeness</h2>
        <p>Percentage of cases that include various metadata fields such as judges, attorneys, syllabus, etc.</p>
        {fig4_html}
    </div>

    <div class="section">
        <h2>Citation Analysis</h2>
        <p>Distribution of citation counts and trends over time showing how often cases are cited by other cases.</p>
        {fig5_html}
    </div>

    <div class="section">
        <h2>Seasonal Patterns</h2>
        <p>Monthly patterns in Supreme Court case filings.</p>
        {fig6_html}
    </div>

    <div class="section">
        <h2>Token Analysis</h2>
        <p>Token counts for estimating computational requirements for processing this dataset with language models.</p>
        <p><strong>Counting method:</strong> {token_stats['counting_method']}</p>
        <div class="summary-box">
            <strong>Total tokens in dataset:</strong> {token_stats['total_tokens']:,} ({token_stats['total_tokens']/1_000_000:.2f}M tokens)
        </div>
        {fig7_html}
    </div>

{bin_section}

    <div class="footer">
        <p><strong>Dataset Source:</strong> Harvard COLD Cases Dataset (Supreme Court Opinions)</p>
        <p><strong>Data Cleaning:</strong> Removed exact duplicates, merged cases with same name and date</p>
        <p>Report generated using Python, Pandas, and Plotly</p>
    </div>

</body>
</html>
    """
    
    print(f"Writing HTML report to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report successfully generated: {output_path}")
    return output_path

def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage: python scotus_eda_complete.py <path_to_cleaned_csv> [path_to_bin_statistics.json]")
        print("Example: python scotus_eda_complete.py data/scotus_final_clean.csv data/bin_statistics.json")
        sys.exit(1)
    
    input_path = sys.argv[1]
    bin_stats_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_path = Path(input_path).parent / 'scotus_eda_report.html'
    
    # Load data
    df = load_data(input_path)
    
    # Load bin statistics if provided
    bin_data = load_bin_statistics(bin_stats_path) if bin_stats_path else None
    
    # Generate report
    generate_html_report(df, output_path, bin_data)
    
    print(f"Analysis complete!")
    print(f"Open the report in your browser: {output_path}")

if __name__ == "__main__":
    main()