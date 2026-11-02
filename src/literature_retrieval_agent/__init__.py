
"""Citation-grounded literature retrieval for cell-type annotation.

The network-facing agent is imported only when it is requested. This keeps
offline corpus building, validation, and appraisal commands independent from
the Europe PMC HTTP client dependencies.
"""

from literature_retrieval_agent.config import (
    DEFAULT_CONFIG,
    LiteratureRetrievalConfig,
)
from literature_retrieval_agent.schemas import (
    LiteratureRequest,
    RetrievalPilotResult,
)

__all__ = [
    "DEFAULT_CONFIG",
    "LiteratureRequest",
    "LiteratureRetrievalAgent",
    "LiteratureRetrievalConfig",
    "RetrievalPilotResult",
]


def __getattr__(name: str):
    if name == "LiteratureRetrievalAgent":
        from literature_retrieval_agent.agent import (
            LiteratureRetrievalAgent,
        )

        return LiteratureRetrievalAgent

    raise AttributeError(name)
