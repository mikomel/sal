from enum import Enum
from typing import List


class VisualAnalogyRegime(Enum):
    NOVEL_DOMAIN_TRANSFER = "novel.domain.transfer"
    NOVEL_TARGET_DOMAIN_LINE_TYPE = "novel.target.domain.line.type"
    NOVEL_TARGET_DOMAIN_SHAPE_COLOR = "novel.target.domain.shape.color"

    @staticmethod
    def all() -> List["VisualAnalogyRegime"]:
        return [e for e in VisualAnalogyRegime]

    def short_name(self) -> str:
        if self == VisualAnalogyRegime.NOVEL_DOMAIN_TRANSFER:
            return "transfer"
        elif self == VisualAnalogyRegime.NOVEL_TARGET_DOMAIN_LINE_TYPE:
            return "line-type"
        elif self == VisualAnalogyRegime.NOVEL_TARGET_DOMAIN_SHAPE_COLOR:
            return "shape-color"


class AnswerGenerationStrategy(Enum):
    NORMAL = "normal"
    LEARNING_BY_CONTRASTING = "lbc"
    MIX = "mix"


ALL_VISUAL_ANALOGY_REGIMES = VisualAnalogyRegime.all()
