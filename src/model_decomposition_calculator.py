"""
SHAP Model-Based Decomposition Calculator.

Implements gradient boosting + SHAP attribution to decompose booking changes
between two time periods. Output format matches hierarchical decomposition
for compatibility with existing visualization infrastructure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from typing import Union, NamedTuple, Dict, Tuple, List
from datetime import datetime
from pathlib import Path

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("XGBoost not installed. Run: pip install xgboost")

try:
    import shap
except ImportError:
    raise ImportError("SHAP not installed. Run: pip install shap")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelDecompositionResults(NamedTuple):
    """Container for model-based decomposition results."""
    summary: pd.DataFrame
    segment_detail: pd.DataFrame
    metadata: dict


class ModelDecompositionCalculator:
    """
    SHAP-based booking decomposition calculator.

    Mirrors the interface of hierarchical decomposition but uses
    gradient boosting + SHAP for attribution.
    """

    def __init__(self, model_type='xgboost', **model_params):
        """
        Initialize calculator with model configuration.

        Parameters
        ----------
        model_type : str
            'xgboost' (default)
        model_params : dict
            Model hyperparameters (passed to XGBRegressor)
        """
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.feature_names = None
        self.X_train = None
        self.y_train = None
        self.predictions = None

    def fit(self, df: pd.DataFrame) -> None:
        """
        Train model on historical data.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with all months (pivoted format)
            Must have columns: lender, date column (e.g., month_begin_date, week_begin_date),
            num_tot_bks, [features...]
        """
        # Extract features and target
        X, y = extract_features_and_target(df)

        # Store for later diagnostics
        self.X_train = X
        self.y_train = y
        self.feature_names = list(X.columns)

        # Initialize model
        if self.model_type == 'xgboost':
            # Default params optimized for pivoted data (few observations, many features)
            # Uses strong regularization to prevent overfitting
            default_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 100,           # Reduced to avoid overfitting
                'learning_rate': 0.1,         # Increased for faster learning
                'max_depth': 3,               # Shallow trees
                'min_child_weight': 2,        # Minimum samples per leaf
                'reg_alpha': 1.0,             # L1 regularization
                'reg_lambda': 1.0,            # L2 regularization
                'subsample': 1,             # Row sampling
                'colsample_bytree': 1,      # Use only 30% of features per tree
                'random_state': 42,
                'verbosity': 0                # Suppress output
            }
            # Override with user params
            default_params.update(self.model_params)
            self.model = xgb.XGBRegressor(**default_params)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Only 'xgboost' is supported.")

        # Train model
        self.model.fit(X, y)

        # Store predictions
        self.predictions = self.model.predict(X)

    def calculate_decomposition(
        self,
        df: pd.DataFrame,
        date_a: Union[str, datetime],
        date_b: Union[str, datetime],
        lender: str = 'ACA',
        date_column: str = 'month_begin_date'
    ) -> ModelDecompositionResults:
        """
        Calculate SHAP-based decomposition between two dates.

        Returns structure identical to hierarchical decomposition.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset (pivoted format)
        date_a : str/datetime
            Base period (Period 1)
        date_b : str/datetime
            Comparison period (Period 2)
        lender : str
            Lender identifier
        date_column : str
            Name of the date column in the DataFrame (default 'month_begin_date').
            Use 'week_begin_date' for weekly analysis or any other date column name.

        Returns
        -------
        ModelDecompositionResults
            NamedTuple with:
            - summary: Aggregate effects (matching hierarchical format)
            - segment_detail: Segment-level breakdown (matching hierarchical format)
            - metadata: Model diagnostics + calculation metadata
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Validate date column exists
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")

        # Normalize dates
        date_a = pd.to_datetime(date_a)
        date_b = pd.to_datetime(date_b)

        # Filter to specific dates
        df_date_a = df[df[date_column] == date_a].copy()
        df_date_b = df[df[date_column] == date_b].copy()

        if len(df_date_a) == 0:
            raise ValueError(f"No data found for {date_a.date()}")
        if len(df_date_b) == 0:
            raise ValueError(f"No data found for {date_b.date()}")

        # Extract features for both dates
        X_date_a, _ = extract_features_and_target(df_date_a)
        X_date_b, _ = extract_features_and_target(df_date_b)

        # Calculate SHAP values
        shap_values_a, shap_values_b, base_value = calculate_shap_values(
            self.model, X_date_a, X_date_b
        )

        # Transform SHAP values to segment-level format
        segment_detail = transform_shap_to_segments(
            shap_values_a=shap_values_a,
            shap_values_b=shap_values_b,
            feature_names=self.feature_names,
            df_date_a=df_date_a,
            df_date_b=df_date_b,
            date_a=date_a,
            date_b=date_b,
            lender=lender
        )

        # Aggregate to summary
        summary = aggregate_segment_summary(segment_detail)

        # Prepare metadata
        diagnostics = self.get_model_diagnostics()
        metadata = {
            'lender': lender,
            'date_a': str(date_a.date()),
            'date_b': str(date_b.date()),
            'period_1_total_apps': int(df_date_a['num_tot_apps'].iloc[0]) if 'num_tot_apps' in df_date_a.columns else None,
            'period_2_total_apps': int(df_date_b['num_tot_apps'].iloc[0]) if 'num_tot_apps' in df_date_b.columns else None,
            'period_1_total_bookings': float(df_date_a['num_tot_bks'].iloc[0]),
            'period_2_total_bookings': float(df_date_b['num_tot_bks'].iloc[0]),
            'delta_total_bookings': float(df_date_b['num_tot_bks'].iloc[0] - df_date_a['num_tot_bks'].iloc[0]),
            'num_segments': len(segment_detail),
            'model_type': 'XGBoost',
            'model_mae': diagnostics['mae'],
            'model_r2': diagnostics['r2'],
            'shap_method': 'TreeExplainer',
            'shap_base_value': float(base_value),
            'calculation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method': 'model_based_shap'
        }

        return ModelDecompositionResults(
            summary=summary,
            segment_detail=segment_detail,
            metadata=metadata
        )

    def get_model_diagnostics(self) -> Dict:
        """
        Return model performance metrics.

        Returns
        -------
        dict
            - mae: Mean absolute error
            - rmse: Root mean squared error
            - r2: R-squared score
            - mean_actual: Mean of actual values
            - mae_pct: MAE as percentage of mean
            - feature_importance: Top 20 features by gain
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Calculate metrics
        mae = mean_absolute_error(self.y_train, self.predictions)
        rmse = np.sqrt(mean_squared_error(self.y_train, self.predictions))
        r2 = r2_score(self.y_train, self.predictions)
        mean_actual = self.y_train.mean()
        mae_pct = (mae / mean_actual) * 100

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            importance_df = pd.DataFrame()

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_actual': mean_actual,
            'mae_pct': mae_pct,
            'feature_importance': importance_df
        }

    def plot_diagnostics(self, output_dir: Union[str, Path]) -> None:
        """
        Generate and save model diagnostic plots.

        Saves to output_dir:
        - mae_chart.png: MAE distribution
        - calibration_plot.png: Predictions vs actuals scatter
        - feature_importance.png: Top 20 features by importance

        Parameters
        ----------
        output_dir : str or Path
            Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        diagnostics = self.get_model_diagnostics()

        # 1. MAE Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        errors = np.abs(self.y_train - self.predictions)
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(diagnostics['mae'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean MAE: {diagnostics['mae']:.2f}")
        ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2,
                   label=f"Median MAE: {np.median(errors):.2f}")
        ax.set_xlabel('Absolute Error (bookings)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Model Prediction Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'mae_chart.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Calibration Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.y_train, self.predictions, alpha=0.6, s=100, edgecolors='black')

        # Perfect calibration line
        min_val = min(self.y_train.min(), self.predictions.min())
        max_val = max(self.y_train.max(), self.predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Calibration')

        # Regression line
        z = np.polyfit(self.y_train, self.predictions, 1)
        p = np.poly1d(z)
        ax.plot(self.y_train, p(self.y_train), 'g-', linewidth=2, alpha=0.8,
                label=f'Actual Fit: y={z[0]:.3f}x+{z[1]:.1f}')

        ax.set_xlabel('Actual Bookings', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Bookings', fontsize=12, fontweight='bold')
        ax.set_title(f'Calibration Plot (R²={diagnostics["r2"]:.4f})',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(output_dir / 'calibration_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Feature Importance
        if not diagnostics['feature_importance'].empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            top_features = diagnostics['feature_importance'].head(20)
            ax.barh(range(len(top_features)), top_features['importance'],
                    color='steelblue', edgecolor='black')
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'], fontsize=9)
            ax.set_xlabel('Importance (Gain)', fontsize=12, fontweight='bold')
            ax.set_title('Top 20 Features by Importance', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()


def extract_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract X (features) and y (target) from pivoted dataframe.

    X: All columns except lender, month_begin_date, num_tot_bks
    y: num_tot_bks

    Parameters
    ----------
    df : pd.DataFrame
        Pivoted dataframe

    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector (bookings)
    """
    # Identify columns to exclude
    exclude_cols = ['lender', 'month_begin_date', 'num_tot_bks']

    # Feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()
    y = df['num_tot_bks'].copy()

    return X, y


def parse_feature_name(feature: str) -> Dict:
    """
    Parse pivoted feature name into components.

    Example:
    "High_FICO_multi_best_Used_pct_of_total_apps"
    →
    {
        'fico_bands': 'High_FICO',
        'offer_comp_tier': 'multi_best',
        'prod_line': 'Used',
        'metric': 'pct_of_total_apps'
    }

    Parameters
    ----------
    feature : str
        Feature name to parse

    Returns
    -------
    dict
        Parsed components
    """
    # Special case: num_tot_apps (no segment)
    if feature == 'num_tot_apps':
        return {
            'fico_bands': None,
            'offer_comp_tier': None,
            'prod_line': None,
            'metric': 'num_tot_apps'
        }

    # Pattern: {FICO}_{COMP}_{PROD}_{METRIC}
    # FICO: High_FICO, Med_FICO, Low_FICO, Null_FICO
    # COMP: multi_best, multi_other, solo_offer
    # PROD: Used, VMax
    # METRIC: pct_of_total_apps, str_apprv_rate, str_bk_rate, cond_apprv_rate, cond_bk_rate

    pattern = r'^(High_FICO|Med_FICO|Low_FICO|Null_FICO)_(multi_best|multi_other|solo_offer)_(Used|VMax)_(.+)$'
    match = re.match(pattern, feature)

    if not match:
        raise ValueError(f"Cannot parse feature name: {feature}")

    return {
        'fico_bands': match.group(1),
        'offer_comp_tier': match.group(2),
        'prod_line': match.group(3),
        'metric': match.group(4)
    }


def calculate_shap_values(
    model,
    X_date_a: pd.DataFrame,
    X_date_b: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate SHAP values for two dates.

    Parameters
    ----------
    model : fitted model
        Trained model
    X_date_a : pd.DataFrame
        Features for date A
    X_date_b : pd.DataFrame
        Features for date B

    Returns
    -------
    shap_a : np.ndarray
        SHAP values for date A (1D array)
    shap_b : np.ndarray
        SHAP values for date B (1D array)
    base_value : float
        Expected value (base prediction)
    """
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values_a = explainer.shap_values(X_date_a)
    shap_values_b = explainer.shap_values(X_date_b)

    # For single observation, flatten to 1D
    if shap_values_a.ndim > 1:
        shap_values_a = shap_values_a[0]
    if shap_values_b.ndim > 1:
        shap_values_b = shap_values_b[0]

    base_value = explainer.expected_value

    return shap_values_a, shap_values_b, base_value


def transform_shap_to_segments(
    shap_values_a: np.ndarray,
    shap_values_b: np.ndarray,
    feature_names: List[str],
    df_date_a: pd.DataFrame,
    df_date_b: pd.DataFrame,
    date_a: pd.Timestamp,
    date_b: pd.Timestamp,
    lender: str
) -> pd.DataFrame:
    """
    Transform SHAP attributions into segment-level decomposition.

    This is the KEY TRANSFORMATION FUNCTION.

    Process:
    1. Parse feature names to extract segment identifiers
    2. Group SHAP values by segment
    3. Map SHAP contributions to effect types
    4. Calculate deltas between date A and B
    5. Reconstruct period values from pivoted data

    Parameters
    ----------
    shap_values_a : np.ndarray
        SHAP values for date A
    shap_values_b : np.ndarray
        SHAP values for date B
    feature_names : List[str]
        Feature names
    df_date_a : pd.DataFrame
        Raw data for date A (pivoted)
    df_date_b : pd.DataFrame
        Raw data for date B (pivoted)
    date_a : pd.Timestamp
        Date A
    date_b : pd.Timestamp
        Date B
    lender : str
        Lender name

    Returns
    -------
    pd.DataFrame
        Segment-level breakdown matching hierarchical format
    """
    # Calculate delta SHAP (contribution to change)
    delta_shap = shap_values_b - shap_values_a

    # Initialize segment dictionary
    segments = {}

    # Parse features and aggregate SHAP values by segment
    volume_shap_total = 0.0
    segment_pcts = {}  # Track pct_of_total_apps for volume distribution

    for i, feature in enumerate(feature_names):
        parsed = parse_feature_name(feature)

        if parsed['metric'] == 'num_tot_apps':
            # Store volume SHAP for later distribution
            volume_shap_total = delta_shap[i]
        else:
            # Create segment key
            segment_key = (
                parsed['fico_bands'],
                parsed['offer_comp_tier'],
                parsed['prod_line']
            )

            # Initialize segment if not exists
            if segment_key not in segments:
                segments[segment_key] = {
                    'volume_effect': 0.0,
                    'mix_effect': 0.0,
                    'str_approval_effect': 0.0,
                    'cond_approval_effect': 0.0,
                    'str_booking_effect': 0.0,
                    'cond_booking_effect': 0.0
                }

            # Map metric to effect type
            if parsed['metric'] == 'pct_of_total_apps':
                segments[segment_key]['mix_effect'] += delta_shap[i]
                # Store pct for volume distribution
                col_name_a = f"{parsed['fico_bands']}_{parsed['offer_comp_tier']}_{parsed['prod_line']}_pct_of_total_apps"
                if col_name_a in df_date_a.columns:
                    segment_pcts[segment_key] = df_date_a[col_name_a].iloc[0]
            elif parsed['metric'] == 'str_apprv_rate':
                segments[segment_key]['str_approval_effect'] += delta_shap[i]
            elif parsed['metric'] == 'str_bk_rate':
                segments[segment_key]['str_booking_effect'] += delta_shap[i]
            elif parsed['metric'] == 'cond_apprv_rate':
                segments[segment_key]['cond_approval_effect'] += delta_shap[i]
            elif parsed['metric'] == 'cond_bk_rate':
                segments[segment_key]['cond_booking_effect'] += delta_shap[i]

    # Distribute volume effect proportionally by pct_of_total_apps
    total_pct = sum(segment_pcts.values())
    if total_pct > 0:
        for segment_key, pct in segment_pcts.items():
            segments[segment_key]['volume_effect'] = volume_shap_total * (pct / total_pct)

    # Build segment_detail DataFrame
    rows = []
    for segment_key, effects in segments.items():
        fico, comp, prod = segment_key

        # Construct column names for this segment
        prefix = f"{fico}_{comp}_{prod}"

        # Extract period values from pivoted data
        row = {
            'fico_bands': fico,
            'offer_comp_tier': comp,
            'prod_line': prod,
            'period_1_date': str(date_a.date()),
            'period_2_date': str(date_b.date())
        }

        # Period 1 values
        row['period_1_total_apps'] = int(df_date_a['num_tot_apps'].iloc[0]) if 'num_tot_apps' in df_date_a.columns else 0
        row['period_1_pct_of_total'] = df_date_a.get(f"{prefix}_pct_of_total_apps", pd.Series([0.0])).iloc[0]
        row['period_1_segment_apps'] = row['period_1_total_apps'] * row['period_1_pct_of_total']
        row['period_1_str_apprv_rate'] = df_date_a.get(f"{prefix}_str_apprv_rate", pd.Series([0.0])).iloc[0]
        row['period_1_str_bk_rate'] = df_date_a.get(f"{prefix}_str_bk_rate", pd.Series([0.0])).iloc[0]
        row['period_1_cond_apprv_rate'] = df_date_a.get(f"{prefix}_cond_apprv_rate", pd.Series([0.0])).iloc[0]
        row['period_1_cond_bk_rate'] = df_date_a.get(f"{prefix}_cond_bk_rate", pd.Series([0.0])).iloc[0]

        # Calculate period 1 segment bookings
        str_bks_1 = row['period_1_segment_apps'] * row['period_1_str_apprv_rate'] * row['period_1_str_bk_rate']
        cond_bks_1 = row['period_1_segment_apps'] * row['period_1_cond_apprv_rate'] * row['period_1_cond_bk_rate']
        row['period_1_segment_bookings'] = str_bks_1 + cond_bks_1

        # Period 2 values
        row['period_2_total_apps'] = int(df_date_b['num_tot_apps'].iloc[0]) if 'num_tot_apps' in df_date_b.columns else 0
        row['period_2_pct_of_total'] = df_date_b.get(f"{prefix}_pct_of_total_apps", pd.Series([0.0])).iloc[0]
        row['period_2_segment_apps'] = row['period_2_total_apps'] * row['period_2_pct_of_total']
        row['period_2_str_apprv_rate'] = df_date_b.get(f"{prefix}_str_apprv_rate", pd.Series([0.0])).iloc[0]
        row['period_2_str_bk_rate'] = df_date_b.get(f"{prefix}_str_bk_rate", pd.Series([0.0])).iloc[0]
        row['period_2_cond_apprv_rate'] = df_date_b.get(f"{prefix}_cond_apprv_rate", pd.Series([0.0])).iloc[0]
        row['period_2_cond_bk_rate'] = df_date_b.get(f"{prefix}_cond_bk_rate", pd.Series([0.0])).iloc[0]

        # Calculate period 2 segment bookings
        str_bks_2 = row['period_2_segment_apps'] * row['period_2_str_apprv_rate'] * row['period_2_str_bk_rate']
        cond_bks_2 = row['period_2_segment_apps'] * row['period_2_cond_apprv_rate'] * row['period_2_cond_bk_rate']
        row['period_2_segment_bookings'] = str_bks_2 + cond_bks_2

        # Deltas
        row['delta_total_apps'] = row['period_2_total_apps'] - row['period_1_total_apps']
        row['delta_pct_of_total'] = row['period_2_pct_of_total'] - row['period_1_pct_of_total']
        row['delta_str_apprv_rate'] = row['period_2_str_apprv_rate'] - row['period_1_str_apprv_rate']
        row['delta_str_bk_rate'] = row['period_2_str_bk_rate'] - row['period_1_str_bk_rate']
        row['delta_cond_apprv_rate'] = row['period_2_cond_apprv_rate'] - row['period_1_cond_apprv_rate']
        row['delta_cond_bk_rate'] = row['period_2_cond_bk_rate'] - row['period_1_cond_bk_rate']
        row['delta_segment_bookings'] = row['period_2_segment_bookings'] - row['period_1_segment_bookings']

        # Effects (from SHAP)
        row['volume_effect'] = effects['volume_effect']
        row['mix_effect'] = effects['mix_effect']
        row['str_approval_effect'] = effects['str_approval_effect']
        row['cond_approval_effect'] = effects['cond_approval_effect']
        row['str_booking_effect'] = effects['str_booking_effect']
        row['cond_booking_effect'] = effects['cond_booking_effect']
        row['total_effect'] = sum(effects.values())

        rows.append(row)

    # Create DataFrame with exact column order matching hierarchical
    segment_detail = pd.DataFrame(rows)

    # Reorder columns to match hierarchical format exactly
    col_order = [
        # Identifiers
        'fico_bands', 'offer_comp_tier', 'prod_line',

        # Period 1
        'period_1_date', 'period_1_total_apps', 'period_1_pct_of_total',
        'period_1_segment_apps', 'period_1_str_apprv_rate', 'period_1_str_bk_rate',
        'period_1_cond_apprv_rate', 'period_1_cond_bk_rate', 'period_1_segment_bookings',

        # Period 2
        'period_2_date', 'period_2_total_apps', 'period_2_pct_of_total',
        'period_2_segment_apps', 'period_2_str_apprv_rate', 'period_2_str_bk_rate',
        'period_2_cond_apprv_rate', 'period_2_cond_bk_rate', 'period_2_segment_bookings',

        # Deltas
        'delta_total_apps', 'delta_pct_of_total', 'delta_str_apprv_rate',
        'delta_str_bk_rate', 'delta_cond_apprv_rate', 'delta_cond_bk_rate',
        'delta_segment_bookings',

        # Effects
        'volume_effect', 'mix_effect', 'str_approval_effect', 'cond_approval_effect',
        'str_booking_effect', 'cond_booking_effect', 'total_effect'
    ]

    return segment_detail[col_order]


def aggregate_segment_summary(segment_detail: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate segment effects into summary table.

    Parameters
    ----------
    segment_detail : pd.DataFrame
        Segment-level breakdown

    Returns
    -------
    pd.DataFrame
        Summary with columns: effect_type, booking_impact, pct_of_total_change
        Matching hierarchical format exactly
    """
    # Aggregate effects
    effects = {
        'volume_effect': segment_detail['volume_effect'].sum(),
        'mix_effect': segment_detail['mix_effect'].sum(),
        'str_approval_effect': segment_detail['str_approval_effect'].sum(),
        'cond_approval_effect': segment_detail['cond_approval_effect'].sum(),
        'str_booking_effect': segment_detail['str_booking_effect'].sum(),
        'cond_booking_effect': segment_detail['cond_booking_effect'].sum(),
    }

    total_change = sum(effects.values())

    # Build summary DataFrame
    summary = pd.DataFrame({
        'effect_type': list(effects.keys()) + ['total_change'],
        'booking_impact': list(effects.values()) + [total_change]
    })

    return summary
