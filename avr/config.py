from types import ModuleType
from typing import Iterable, Dict, List

from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig

from avr.data.augmentor import AugmentorFactory
from avr.data.dataset import DatasetSplit
from avr.data.rpm.pgm.rule_encoder import create_pgm_rule_encoder
from avr.data.rpm.raven.configuration import RavenConfiguration
from avr.data.rpm.raven.rule_encoder import create_raven_rule_encoder
from avr.data.vap.regime import VisualAnalogyRegime, AnswerGenerationStrategy
from avr.data.vap.rule_encoder import create_vap_rule_encoder
from avr.model.rule_predictor import RulePredictor
from avr.model.target_predictor import TargetPredictor
from avr.task.task import Task


def compose_config(
    config_path: str = "../config",
    config_name: str = "default",
    overrides: List[str] = (),
) -> DictConfig:
    with initialize(config_path=config_path):
        return compose(config_name=config_name, overrides=overrides)


def register_omega_conf_resolvers():
    resolvers = {
        "List": lambda *args: list(args),
        "Tuple": lambda *args: tuple(args),
        "DatasetSplit": lambda x: DatasetSplit[x],
        "RavenConfiguration": lambda x: RavenConfiguration[x],
        "RavenRuleEncoder": create_raven_rule_encoder,
        "PgmRuleEncoder": create_pgm_rule_encoder,
        "VisualAnalogyRegime": lambda x: VisualAnalogyRegime[x],
        "AnswerGenerationStrategy": lambda x: AnswerGenerationStrategy[x],
        "VAPRuleEncoder": create_vap_rule_encoder,
        "RulePredictor": lambda x: RulePredictor[x],
        "TargetPredictor": lambda x: TargetPredictor[x],
        "AugmentorFactory": lambda x: AugmentorFactory[x],
        "Task": create_task,
    }
    for name, resolver in resolvers.items():
        OmegaConf.register_new_resolver(name, resolver)


def resolve(module: ModuleType, class_name: str, *args) -> object:
    class_ = getattr(module, class_name)
    kwargs = to_kwargs(*args)
    return class_(**kwargs)


def create_task(*args) -> Task:
    return Task(**to_kwargs(*args))


def to_kwargs(*args) -> Dict:
    return {k: v for k, v in pairwise(args)}


def pairwise(iterable: Iterable):
    """s -> (s0, s1), (s2, s3), (s4, s5), ..."""
    iterator = iter(iterable)
    return zip(iterator, iterator)
