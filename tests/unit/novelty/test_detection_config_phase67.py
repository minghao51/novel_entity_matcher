"""Tests for DetectionConfig phase 6-7 strategy config resolution."""

from novelentitymatcher.novelty.config.base import DetectionConfig
from novelentitymatcher.novelty.config.strategies import (
    EnergyConfig,
    MixtureGaussianConfig,
    ReActConfig,
)


def test_detection_config_resolves_new_strategy_configs():
    config = DetectionConfig(
        strategies=["energy_ood", "mixture_gaussian", "react_energy"],
        energy_ood=EnergyConfig(temperature=2.0),
        mixture_gaussian=MixtureGaussianConfig(regularization=1e-5),
        react_energy=ReActConfig(trim_percentile=0.8),
    )

    assert isinstance(config.get_strategy_config("energy_ood"), EnergyConfig)
    assert isinstance(
        config.get_strategy_config("mixture_gaussian"), MixtureGaussianConfig
    )
    assert isinstance(config.get_strategy_config("react_energy"), ReActConfig)
    assert config.get_strategy_config("energy_ood").temperature == 2.0
    assert config.get_strategy_config("mixture_gaussian").regularization == 1e-5
    assert config.get_strategy_config("react_energy").trim_percentile == 0.8
