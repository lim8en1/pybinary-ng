"""Annotation access across Python versions.

On 3.14+ annotations are lazy (PEP 649) and ``annotationlib`` hands back both a
``FORWARDREF`` view — where a reference to an earlier field survives as a
``ForwardRef`` instead of raising ``NameError`` — and a ``STRING`` view used to
catch names captured by module globals.

On 3.13 annotations are evaluated eagerly at class-body time. That still works,
but only for the **quoted** reference form: ``Bytes["n"]`` is a plain string
literal and evaluates fine, while ``Bytes[n]`` raises ``NameError`` before the
metaclass ever runs. Quoted references produce byte-identical marker tuples on
both versions, so the rest of the library is unaffected.

There is no ``STRING`` view below 3.14, which is correct rather than a gap:
quoted-only references cannot be captured by a global in the first place, so
there is nothing for the shadowing check to find.
"""

from __future__ import annotations

import sys

__all__ = ["ForwardRef", "HAS_LAZY_ANNOTATIONS", "read_annotations"]

HAS_LAZY_ANNOTATIONS = sys.version_info >= (3, 14)

if HAS_LAZY_ANNOTATIONS:
    import annotationlib
    from annotationlib import ForwardRef

    def read_annotations(ns: dict) -> tuple[dict, dict]:
        """``(annotations, sources)`` for a class namespace."""
        annotate = annotationlib.get_annotate_from_class_namespace(ns)
        if annotate is None:
            # Classes built by calling the metaclass directly have no annotate
            # function; accept a plain __annotations__ mapping instead.
            return dict(ns.get("__annotations__", {})), {}
        return (
            annotationlib.call_annotate_function(annotate, annotationlib.Format.FORWARDREF),
            annotationlib.call_annotate_function(annotate, annotationlib.Format.STRING),
        )

else:  # pragma: no cover - exercised only on 3.13

    class ForwardRef:  # noqa: D101 - a sentinel, never instantiated
        """Placeholder so ``isinstance`` checks compile below 3.14.

        Nothing is ever an instance of it: without lazy annotations a field
        reference arrives as a plain ``str``.
        """

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise TypeError("ForwardRef is unavailable below Python 3.14")

    def read_annotations(ns: dict) -> tuple[dict, dict]:
        """``(annotations, sources)`` for a class namespace."""
        return dict(ns.get("__annotations__", {})), {}
