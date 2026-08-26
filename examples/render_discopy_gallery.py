"""Render the v0.2 DisCoPy diagram gallery."""

from __future__ import annotations

import argparse
from pathlib import Path

from markov_entropy import Channel, Distribution, FiniteSpace
from markov_entropy.adapters import DiagramComparison, DiscopyAdapter


def render_gallery(output_dir: Path) -> list[Path]:
    """Render canonical and information-theoretic diagrams as SVG files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    x = FiniteSpace((0, 1))
    y = FiniteSpace(("a", "b"))
    z = FiniteSpace(("u", "v"))
    source = Distribution(x, [0.4, 0.6])
    joint_state = Distribution(x.tensor(y), [0.35, 0.05, 0.10, 0.50])
    first = Channel(x, y, [[0.8, 0.1], [0.2, 0.9]])
    second = Channel(y, z, [[0.7, 0.3], [0.3, 0.7]])
    joint_channel = Channel(
        x,
        y.tensor(z),
        [[0.55, 0.10], [0.25, 0.20], [0.10, 0.30], [0.10, 0.40]],
    )
    adapter = DiscopyAdapter({x: "X", y: "Y", z: "Z"})

    diagrams = {
        "channel.svg": adapter.channel_box(first, "f"),
        "copy.svg": adapter.copy_diagram(x),
        "discard.svg": adapter.discard_diagram(x),
        "composition.svg": adapter.compose_diagram(first, second, "f", "g"),
        "tensor.svg": adapter.tensor_diagram(first, second, "f", "g"),
    }
    comparisons: dict[str, DiagramComparison] = {
        "state-entropy": adapter.entropy_diagrams(source),
        "channel-entropy": adapter.channel_entropy_diagrams(first),
        "mutual-information": adapter.mutual_information_diagrams(joint_state),
        "channel-mutual-information": adapter.channel_mutual_information_diagrams(joint_channel),
    }

    generated: list[Path] = []
    for filename, diagram in diagrams.items():
        path = output_dir / filename
        diagram.draw(path=str(path))
        generated.append(path)
    for stem, comparison in comparisons.items():
        actual_path = output_dir / f"{stem}-actual.svg"
        reference_path = output_dir / f"{stem}-reference.svg"
        comparison.draw(str(actual_path), str(reference_path))
        generated.extend((actual_path, reference_path))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/diagrams"),
        help="Directory for generated SVG files",
    )
    args = parser.parse_args()
    for path in render_gallery(args.output):
        print(path)


if __name__ == "__main__":
    main()
