from dataclasses import dataclass, field


@dataclass(frozen=True)
class MarkerSupport:
    gene: str
    supported_by_cellmarker_panglaodb: bool = False
    supported_by_hpa: bool = False
    supported_by_literature: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def supported_anywhere(self) -> bool:
        return (
            self.supported_by_cellmarker_panglaodb
            or self.supported_by_hpa
            or self.supported_by_literature
        )


@dataclass(frozen=True)
class ClusterAuditResult:
    dataset: str
    cluster: str
    original_decision: str
    original_labels: str
    projected_decision: str
    projected_labels: str
    rescue_trigger_changed: bool
    final_decision_changed: bool
    reason: str
