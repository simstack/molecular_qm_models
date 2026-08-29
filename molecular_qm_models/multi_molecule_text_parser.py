from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from typing import Iterator


class XyzFrame(BaseModel):
    model_config = ConfigDict(frozen=True)

    natoms: int
    comment: str
    atom_lines: list[str]

    @property
    def energy(self) -> float | None:
        # CREST typically writes energy as the whole comment line (possibly with spaces)
        try:
            return float(self.comment.strip())
        except Exception:
            return None

    def to_xyz_string(self) -> str:
        return "\n".join([str(self.natoms), self.comment, *self.atom_lines]) + "\n"


def iter_multixyz_frames(text: str) -> Iterator[XyzFrame]:
    """
    Parse a concatenated (multi-)XYZ file into individual XYZ frames.
    Tolerant to blank lines between frames.
    """
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    i = 0
    n = len(lines)

    def skip_blanks(idx: int) -> int:
        while idx < n and not lines[idx].strip():
            idx += 1
        return idx

    i = skip_blanks(i)
    while i < n:
        # line 1: natoms
        try:
            natoms = int(lines[i].strip())
        except Exception as e:
            raise ValueError(f"Expected atom count at line {i+1}, got: {lines[i]!r}") from e
        if natoms <= 0:
            raise ValueError(f"Atom count must be > 0 at line {i+1}, got: {natoms}")

        # line 2: comment
        if i + 1 >= n:
            raise ValueError(f"Truncated XYZ: missing comment line after line {i+1}")
        comment = lines[i + 1]

        # next natoms lines: atoms
        start = i + 2
        end = start + natoms
        if end > n:
            raise ValueError(
                f"Truncated XYZ frame starting at line {i+1}: "
                f"expected {natoms} atom lines, got {max(0, n - start)}"
            )

        atom_lines = lines[start:end]
        # Optional sanity check: each atom line has at least 4 tokens
        for k, ln in enumerate(atom_lines, start=1):
            if len(ln.split()) < 4:
                raise ValueError(
                    f"Invalid atom line #{k} in frame starting at line {i+1}: {ln!r}"
                )

        yield XyzFrame(natoms=natoms, comment=comment, atom_lines=atom_lines)

        i = skip_blanks(end)


def iter_sdf_frames(text: str) -> Iterator[str]:
    """
    Parse a multi-molecule SDF file into individual molblocks.
    Molecules are delimited by '$$$$' lines.
    """
    # Use a more robust split that handles both Windows and Unix line endings
    # and potential trailing whitespace
    import re
    molblocks = re.split(r'\$\$\$\$\r?\n?', text)
    for block in molblocks:
        block = block.strip()
        if block:
            yield block
