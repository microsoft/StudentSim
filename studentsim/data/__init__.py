"""Building each domain's records from its source corpus.

One subpackage per domain, each with its own entry point, because the three
corpora share nothing but the record shape they end at: chess replays games,
:mod:`studentsim.data.l2` reads tagged essays, and :mod:`studentsim.data.math`
generates guidance against an answer key it audits first.

Nothing is re-exported here. The builders are run rather than imported, and
what they need from each other they import by module.
"""

__all__: list[str] = []
