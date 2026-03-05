import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from scipy import stats
from collections import Counter
import traceback

from sklearn.metrics import precision_score, recall_score, f1_score
from scripts.constants import GOLDEN_LABELS_PATH

def reasoning_metrics(pre_aggregation_results: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze the impact of reasoning prompts on EHR phenotyping performance.
    """
    try:
        # Load gold standard labels
        # TODO: fix GOLDEN PATHS
        gold_df = pd.read_csv(GOLDEN_LABELS_PATH)
        phenotype_name = pre_aggregation_results['phenotype'].iloc[0]
        
        # Get prompt strategy from config or file path
        if 'config' in pre_aggregation_results.columns:
            prompt_strategy = pre_aggregation_results['config'].str.split('-').str[2].iloc[0]
        else:
            # Try to get from file path or set default
            prompt_strategy = 'unknown'
            
        # Add prompt_strategy if not present
        pre_aggregation_results['prompt_strategy'] = prompt_strategy
        
        # Merge gold standard with pre-aggregation results
        analysis_df = pre_aggregation_results.merge(
            gold_df[['hadm_id', phenotype_name]], 
            on='hadm_id',
            how='left'
        )
        
        # Add correctness column
        analysis_df['is_correct'] = (analysis_df['phenotype_status'].astype(str) == 
                                   analysis_df[phenotype_name].astype(str))
        
        analyses = {}
        
        # Run analyses only if we have the required columns
        if 'prompt_strategy' in analysis_df.columns:
            # 1. Basic Performance Metrics
            basic_metrics = analyze_basic_performance(analysis_df, phenotype_name)
            analyses['basic_performance'] = basic_metrics
            
            # 2. Confidence Analysis
            if 'confidence' in analysis_df.columns:
                confidence_metrics = analyze_confidence(analysis_df)
                analyses['confidence'] = confidence_metrics
            
            # 3. Processing Efficiency
            if 'processing_time' in analysis_df.columns and 'total_tokens' in analysis_df.columns:
                efficiency_metrics = analyze_processing_efficiency(analysis_df)
                analyses['efficiency'] = efficiency_metrics
            
            # 4. Phenotype-specific Performance
            phenotype_metrics = analyze_phenotype_performance(analysis_df)
            analyses['phenotype_performance'] = phenotype_metrics
            
            # 5. Justification Analysis
            if 'justification' in analysis_df.columns:
                justification_metrics = analyze_justifications(analysis_df)
                analyses['justification'] = justification_metrics
            
            # 6. Error Analysis
            error_metrics = analyze_errors(analysis_df)
            analyses['errors'] = error_metrics
            
            # 7. Token Impact Analysis
            if any(col in analysis_df.columns for col in ['input_tokens', 'output_tokens', 'total_tokens']):
                token_impact_metrics = analyze_token_impact(analysis_df)
                analyses['token_impact'] = token_impact_metrics
        else:
            logging.warning("Missing prompt_strategy column, skipping analyses")
        
        return analyses
        
    except Exception as e:
        logging.error(f"Error in reasoning_metrics: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {}

def analyze_basic_performance(df: pd.DataFrame, phenotype_name: str) -> pd.DataFrame:
    """Calculate basic performance metrics by reasoning prompt."""
    try:
        # Calculate basic metrics
        metrics = df.groupby('prompt_strategy').agg({
            'is_correct': ['count', 'mean'],  # mean gives accuracy
            'phenotype_status': [
                lambda x: precision_score(df[phenotype_name].loc[x.index], x),
                lambda x: recall_score(df[phenotype_name].loc[x.index], x),
                lambda x: f1_score(df[phenotype_name].loc[x.index], x)
            ]
        }).reset_index()
        
        # Flatten column names
        metrics.columns = ['prompt_strategy', 'n_samples', 'accuracy', 'precision', 'recall', 'f1']
        
        # Add statistical significance tests if COT exists as baseline
        if 'COT' in df['prompt_strategy'].unique():
            baseline_correct = df[df['prompt_strategy'] == 'COT']['is_correct'].values
            
            significance_values = []
            for strategy in metrics['prompt_strategy']:
                if strategy == 'COT':
                    significance_values.append(1.0)  # p-value = 1 when comparing with itself
                else:
                    strategy_correct = df[df['prompt_strategy'] == strategy]['is_correct'].values
                    try:
                        _, p_value = stats.ttest_ind(strategy_correct, baseline_correct)
                        significance_values.append(p_value)
                    except:
                        significance_values.append(None)
            
            metrics['significance_vs_baseline'] = significance_values
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error in analyze_basic_performance: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def analyze_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze relationship between confidence levels and accuracy."""
    try:
        # Group by and calculate metrics
        confidence_metrics = df.groupby(['prompt_strategy', 'confidence']).agg({
            'is_correct': ['mean', 'count'],
            'phenotype_status': 'count'
        }).reset_index()
        
        # Flatten column names
        confidence_metrics.columns = ['prompt_strategy', 'confidence', 'accuracy', 'correct_count', 'total_count']
        
        # Add reliability score (accuracy deviation from expected confidence level)
        def get_expected_accuracy(confidence_level):
            confidence_map = {
                'HIGH': 0.9,
                'MEDIUM': 0.7,
                'LOW': 0.5
            }
            return confidence_map.get(confidence_level, 0.5)
        
        confidence_metrics['expected_accuracy'] = confidence_metrics['confidence'].apply(get_expected_accuracy)
        confidence_metrics['confidence_reliability'] = abs(confidence_metrics['accuracy'] - confidence_metrics['expected_accuracy'])
        
        return confidence_metrics
        
    except Exception as e:
        logging.error(f"Error in analyze_confidence: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def analyze_processing_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze processing efficiency vs performance."""
    efficiency_metrics = df.groupby('prompt_strategy').agg({
        'processing_time': ['mean', 'std'],
        'total_tokens': ['mean', 'std'],
        'is_correct': 'mean'
    }).reset_index()
    
    # Add efficiency score (accuracy per second)
    efficiency_metrics['efficiency_score'] = (
        efficiency_metrics[('is_correct', 'mean')] / 
        efficiency_metrics[('processing_time', 'mean')]
    )
    
    return efficiency_metrics

def analyze_phenotype_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance patterns by phenotype."""
    phenotype_metrics = df.groupby(['phenotype', 'prompt_strategy']).agg({
        'is_correct': ['mean', 'count'],
        'confidence': lambda x: (x == 'HIGH').mean()
    }).reset_index()
    
    return phenotype_metrics

def analyze_justifications(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze justification patterns and their relationship with accuracy."""
    df['justification_length'] = df['justification'].str.len()
    
    justification_metrics = df.groupby('prompt_strategy').agg({
        'justification_length': ['mean', 'std'],
        'is_correct': 'mean'
    }).reset_index()
    
    # Update correlation calculation
    justification_metrics['length_accuracy_correlation'] = df.groupby('prompt_strategy').apply(
        lambda x: safe_correlation(x['justification_length'], x['is_correct'])
    ).values
    
    return justification_metrics

def analyze_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze error patterns."""
    # Add error type (FP/FN)
    df['error_type'] = df.apply(
        lambda row: 'FP' if not row['is_correct'] and row['phenotype_status'] == '1' 
        else 'FN' if not row['is_correct'] and row['phenotype_status'] == '0'
        else 'correct',
        axis=1
    )
    
    error_metrics = df[df['error_type'] != 'correct'].groupby(
        ['prompt_strategy', 'phenotype', 'error_type', 'confidence']
    ).agg({
        'hadm_id': 'count',
        'is_correct': 'size'
    }).reset_index()
    
    error_metrics['error_rate'] = error_metrics['hadm_id'] / len(df)
    
    return error_metrics 

def safe_correlation(x, y):
    """Calculate correlation while handling constant inputs."""
    try:
        if len(x.unique()) == 1 or len(y.unique()) == 1:
            return np.nan
        return x.corr(y)
    except:
        return np.nan

def analyze_token_impact(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze the impact of token counts on prediction accuracy.
    
    Returns:
        Dictionary containing:
        - token_comparison: Statistical comparison of token counts between correct/incorrect predictions
        - token_correlations: Correlation coefficients between token counts and accuracy
        - token_thresholds: Analysis of optimal token thresholds for accuracy
    """
    analyses = {}
    
    try:
        # 1. Token Count Comparison between Correct and Incorrect Predictions
        token_stats = pd.DataFrame()
        
        for token_type in ['input_tokens', 'output_tokens', 'total_tokens']:
            if token_type in df.columns:
                # Calculate statistics for correct vs incorrect predictions
                correct_tokens = df[df['is_correct']][token_type]
                incorrect_tokens = df[~df['is_correct']][token_type]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(correct_tokens, incorrect_tokens)
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(
                    (len(correct_tokens) * correct_tokens.std()**2 + 
                     len(incorrect_tokens) * incorrect_tokens.std()**2) / 
                    (len(correct_tokens) + len(incorrect_tokens) - 2)
                )
                effect_size = (correct_tokens.mean() - incorrect_tokens.mean()) / pooled_std
                
                token_stats = pd.concat([token_stats, pd.DataFrame({
                    'token_type': [token_type],
                    'correct_mean': [correct_tokens.mean()],
                    'correct_std': [correct_tokens.std()],
                    'incorrect_mean': [incorrect_tokens.mean()],
                    'incorrect_std': [incorrect_tokens.std()],
                    't_statistic': [t_stat],
                    'p_value': [p_value],
                    'effect_size': [effect_size]
                })])
        
        analyses['token_comparison'] = token_stats
        
        # 2. Correlation Analysis
        correlation_data = []
        for token_type in ['input_tokens', 'output_tokens', 'total_tokens']:
            if token_type in df.columns:
                correlation_data.append({
                    'token_type': token_type,
                    'correlation': safe_correlation(df[token_type], df['is_correct'])
                })
        
        analyses['token_correlations'] = pd.DataFrame(correlation_data)
        
        # 3. Token Threshold Analysis
        threshold_data = []
        for token_type in ['input_tokens', 'output_tokens', 'total_tokens']:
            if token_type in df.columns:
                percentiles = range(0, 101, 10)  # 0, 10, 20, ..., 100
                for p in percentiles:
                    threshold = np.percentile(df[token_type], p)
                    subset = df[df[token_type] <= threshold]
                    accuracy = subset['is_correct'].mean()
                    
                    threshold_data.append({
                        'token_type': token_type,
                        'percentile': p,
                        'threshold': threshold,
                        'accuracy': accuracy,
                        'sample_size': len(subset)
                    })
        
        analyses['token_thresholds'] = pd.DataFrame(threshold_data)
        
        # 4. Prompt Strategy-specific Token Analysis
        def calculate_token_stats(group_df):
            stats_dict = {}
            for col in ['input_tokens', 'output_tokens', 'total_tokens']:
                if col in group_df.columns:
                    stats_dict[f'{col}_mean'] = group_df[col].mean()
                    stats_dict[f'{col}_std'] = group_df[col].std()
                    stats_dict[f'{col}_correlation'] = safe_correlation(group_df[col], group_df['is_correct'])
            return pd.Series(stats_dict)

        strategy_token_stats = df.groupby('prompt_strategy').apply(calculate_token_stats).reset_index()
        
        analyses['strategy_token_stats'] = strategy_token_stats
        
        return analyses
        
    except Exception as e:
        logging.error(f"Error in analyze_token_impact: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {} 