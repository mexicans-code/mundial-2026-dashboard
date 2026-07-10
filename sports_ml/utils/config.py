from dataclasses import dataclass, field
from typing import Optional


@dataclass
class APIConfig:
    football_api_key: str = ""
    football_api_base: str = "https://v3.football.api-sports.io"
    basketball_api_base: str = "https://www.balldontlie.io/api/v1"


@dataclass
class ModelConfig:
    model_dir: str = "models/saved"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    early_stopping_rounds: int = 50
    cv_folds: int = 5


@dataclass
class FeatureConfig:
    rolling_windows: list = field(default_factory=lambda: [5])
    include_odds_features: bool = True
    include_h2h: bool = True
    include_rest_days: bool = True
    use_home_advantage: bool = True
    include_fifa_rankings: bool = True
    fifa_rankings_path: str = "data/fifa_rankings.csv"
    fifa_rank_default: int = 150


@dataclass
class BettingConfig:
    stake_per_bet: float = 1.0
    min_odds_value: float = 1.50
    max_odds_value: float = 10.0
    kelly_fraction: float = 0.25
    confidence_threshold: float = 0.55


@dataclass
class Config:
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    betting: BettingConfig = field(default_factory=BettingConfig)
    sport: str = "football"
    league_id: int = 71  # Brasileirão Série A
    season: int = 2025
