"""Optional DisCoPy adapter for symbolic Markov string diagrams.

The adapter targets DisCoPy 1.2.2. It keeps numerical evaluation in
``markov_entropy`` and uses DisCoPy only as a symbolic and drawing layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..channels import Channel
from ..distributions import Distribution
from ..spaces import FiniteSpace


def _markov() -> Any:
    try:
        from discopy import markov
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "DisCoPy support requires `pip install markov-entropy[discopy]`"
        ) from error
    return markov


@dataclass(frozen=True)
class DiagramComparison:
    """Two parallel diagrams representing an actual and reference process."""

    actual: Any
    reference: Any
    description: str

    def draw(self, actual_path: str, reference_path: str, **kwargs: Any) -> None:
        """Draw both sides to separate files using DisCoPy's drawing backend."""
        self.actual.draw(path=actual_path, **kwargs)
        self.reference.draw(path=reference_path, **kwargs)


@dataclass
class DiscopyAdapter:
    """Translate finite spaces, channels, and information definitions to DisCoPy.

    A finite non-product space is represented by one atomic wire. Product
    spaces preserve their factors and become tensor products of atomic wires.
    Channel matrices remain in ``markov_entropy``; channel boxes are symbolic.
    """

    space_names: dict[FiniteSpace, str] = field(default_factory=dict)

    def register_space(self, space: FiniteSpace, name: str) -> None:
        """Assign a stable diagram label to an atomic finite space."""
        if space.components:
            raise ValueError("only atomic finite spaces can be registered")
        if not name:
            raise ValueError("space name must be non-empty")
        self.space_names[space] = name

    def type_for(self, space: FiniteSpace, name: str | None = None) -> Any:
        """Return the DisCoPy type corresponding to a finite space."""
        markov = _markov()
        if space.components:
            factors = [self.type_for(factor) for factor in space.factors]
            result = factors[0]
            for factor in factors[1:]:
                result = result @ factor
            return result
        if name is not None:
            self.register_space(space, name)
        label = self.space_names.get(space, self._default_name(space))
        return markov.Ty(label)

    def channel_box(self, channel: Channel, name: str = "f") -> Any:
        """Represent a numerical channel as a symbolic Markov box."""
        markov = _markov()
        return markov.Box(name, self.type_for(channel.domain), self.type_for(channel.codomain))

    def state_box(self, distribution: Distribution, name: str = "p") -> Any:
        """Represent a probability source ``I -> X`` as a symbolic box."""
        markov = _markov()
        return markov.Box(name, markov.Ty(), self.type_for(distribution.space))

    def copy_diagram(self, space: FiniteSpace) -> Any:
        """Return the canonical DisCoPy copy diagram for a space."""
        markov = _markov()
        return markov.Diagram.copy(self.type_for(space))

    def discard_diagram(self, space: FiniteSpace) -> Any:
        """Return the canonical DisCoPy discard diagram for a space."""
        markov = _markov()
        return markov.Diagram.copy(self.type_for(space), n=0)

    def compose_diagram(
        self,
        before: Channel,
        after: Channel,
        before_name: str = "f",
        after_name: str = "g",
    ) -> Any:
        """Represent ``after ∘ before`` as sequential symbolic composition."""
        if before.codomain != after.domain:
            raise ValueError("channel codomain/domain mismatch")
        return self.channel_box(before, before_name) >> self.channel_box(after, after_name)

    def tensor_diagram(
        self,
        left: Channel,
        right: Channel,
        left_name: str = "f",
        right_name: str = "g",
    ) -> Any:
        """Represent parallel channel composition as a tensor diagram."""
        return self.channel_box(left, left_name) @ self.channel_box(right, right_name)

    def entropy_diagrams(
        self,
        distribution: Distribution,
        state_name: str = "p",
    ) -> DiagramComparison:
        """Diagram the two processes compared by distribution entropy."""
        state = self.state_box(distribution, state_name)
        actual = state >> self.copy_diagram(distribution.space)
        reference = state @ state
        return DiagramComparison(actual, reference, "copy output versus independent copies")

    def channel_entropy_diagrams(
        self,
        channel: Channel,
        channel_name: str = "f",
    ) -> DiagramComparison:
        """Diagram the two processes compared by channel entropy."""
        box = self.channel_box(channel, channel_name)
        actual = box >> self.copy_diagram(channel.codomain)
        reference = self.copy_diagram(channel.domain) >> box @ box
        return DiagramComparison(actual, reference, "copied output versus repeated channel")

    def mutual_information_diagrams(
        self,
        distribution: Distribution,
        state_name: str = "pXY",
    ) -> DiagramComparison:
        """Diagram a joint state and the product of its two marginals."""
        if len(distribution.space.factors) != 2:
            raise ValueError("mutual information requires exactly two factors")
        actual = self.state_box(distribution, state_name)
        first = self.state_box(distribution.marginal(0), "pX")
        second = self.state_box(distribution.marginal(1), "pY")
        return DiagramComparison(actual, first @ second, "joint state versus product marginals")

    def channel_mutual_information_diagrams(
        self,
        channel: Channel,
        channel_name: str = "fYZ",
    ) -> DiagramComparison:
        """Diagram a two-output channel and its conditionally independent form."""
        if len(channel.codomain.factors) != 2:
            raise ValueError("mutual information requires exactly two output factors")
        actual = self.channel_box(channel, channel_name)
        first = self.channel_box(channel.marginal(0), "fY")
        second = self.channel_box(channel.marginal(1), "fZ")
        reference = self.copy_diagram(channel.domain) >> first @ second
        return DiagramComparison(actual, reference, "joint channel versus product marginals")

    @staticmethod
    def _default_name(space: FiniteSpace) -> str:
        labels = ",".join(str(label) for label in space.labels)
        return "{" + labels + "}"
