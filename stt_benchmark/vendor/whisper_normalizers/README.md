Verbatim copies of openai/whisper `whisper/normalizers/{__init__,basic,english}.py` and `english.json`
(MIT licence, (c) OpenAI), repo commit 86098128c0b4f24f0e2aa2994de830614b474227, fetched 2026-10-06. NOT modified.
Copied from the Phase 1 scorer's vendor directory so both scorers use byte-identical English normalisation.
Only `EnglishTextNormalizer` is used by strict_scoring.py (English); `BasicTextNormalizer` is NOT used because it
splits words at combining marks (Yoruba, Igbo, Arabic) - see tests/check_strict_scoring.py.
Extra dependencies: `regex`, `more_itertools`.
