"""The normalized description of a single field."""

from __future__ import annotations

import ast
from dataclasses import dataclass

__all__ = ["FieldSpec", "FIXED_KINDS"]

# Kinds whose encoded width is known from the declaration alone. Consecutive
# runs of these collapse into one struct.Struct call. ``bitrun`` is a synthetic
# kind produced by grouping adjacent ``Bits`` fields; raw ``bits`` specs never
# reach the emitters.
FIXED_KINDS = frozenset({"scalar", "const", "padding", "bitrun"})

# Kinds whose value may be absent, so ``__init__`` defaults them to None
# instead of demanding an argument.
OPTIONAL_KINDS = frozenset({"if", "switch"})


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One field, after annotations have been interpreted.

    Exactly one of ``count`` / ``expr`` is set for sized fields; both are None
    when a Bytes/Str field consumes the rest of the buffer.
    """

    name: str
    kind: str                      # scalar | const | padding | bytes | str | array | nested
                                   # | bits | bitrun | if | switch | pointer | checksum
    stored: bool = True            # becomes an instance attribute
    char: str | None = None        # struct character, for scalars
    size: int | None = None        # encoded width, when statically known
    count: int | None = None       # static length / element count
    expr: str | None = None        # length / count expression, inlined into generated code
    elem: type | None = None       # Array element type (scalar marker or structure class)
    nested: type | None = None     # structure class, for nested fields
    encoding: str | None = None    # text encoding, for str fields
    const: object | None = None    # expected literal: bytes for const, int for bits

    # -- wrapping and computed kinds -------------------------------------
    cond: str | None = None        # If: condition source, inlined
    disc: str | None = None        # Switch: discriminator expression, inlined
    off_expr: str | None = None    # Pointer: offset expression, inlined
    start: str | None = None       # Checksum: name of the field the span starts at
    fn: object | None = None       # Checksum: the callable
    inner: "FieldSpec | None" = None            # If / Pointer / Checksum payload
    variants: tuple | None = None               # Switch: ((value, spec | None), ...)
    members: tuple | None = None                # bitrun: the Bits specs it groups
    width: int | None = None                    # bits: width in bits
    nullable: bool = False                      # Pointer: offset 0 means absent

    @property
    def fixed(self) -> bool:
        return self.kind in FIXED_KINDS

    @property
    def optional(self) -> bool:
        """Whether the value may legitimately be ``None``."""
        return self.kind in OPTIONAL_KINDS or (self.kind == "pointer" and self.nullable)

    @property
    def is_integer(self) -> bool:
        """Whether this field can be referenced by a length or condition."""
        if self.kind == "bits":
            return True
        return self.kind == "scalar" and self.char not in ("f", "d")

    def expressions(self) -> tuple[str, ...]:
        """Every inlined expression this field carries, in no order."""
        return tuple(
            e for e in (self.expr, self.cond, self.disc, self.off_expr) if e
        )

    def ref_names(self) -> set[str]:
        """Field names this spec refers to, recursively.

        ``_check_shadowing`` subtracts this from the names it sees in the
        annotation source, so every expression-bearing attribute must be
        represented or a legitimate reference looks like a captured global.
        """
        names: set[str] = set()
        for expr in self.expressions():
            for node in ast.walk(ast.parse(expr, mode="eval")):
                if isinstance(node, ast.Name):
                    names.add(node.id)
        if self.start:
            names.add(self.start)
        if self.inner is not None:
            names |= self.inner.ref_names()
        for child in self.children():
            names |= child.ref_names()
        return names

    def children(self) -> tuple["FieldSpec", ...]:
        """Sub-specs that participate in emission."""
        out: list[FieldSpec] = []
        if self.variants:
            out.extend(spec for _, spec in self.variants if spec is not None)
        if self.members:
            out.extend(self.members)
        return tuple(out)
