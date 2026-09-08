"""Audit whether code comments can decide parent eligibility in V10.7R.

Parent eligibility (``eligible_nodes``) renders the full Design Base block,
including its comments, against the model context budget. Comments carry no
execution semantics, so if comment-heavy nodes systematically lost parent
eligibility the capacity check would be wrong. This script scans archived
``tree_state.json`` files and reports the comment share per node, plus how
many nodes could plausibly flip eligibility if comments were stripped.

Token counts use whitespace words as a proxy (labeled as such); the model
budget is 16128 real tokens, so only nodes with hundreds of comment words
could matter at all.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tokenize
from pathlib import Path

COMMENT_FALLBACK_RE = re.compile(r'#[^\n]*')
# Model context budget for a complete prompt (real tokens, not words).
PROMPT_BUDGET = 16128
# A node needs hundreds of comment words before eligibility could plausibly
# depend on them under any tokenizer conversion.
FLAG_THRESHOLDS = (200, 1000, 2000)


def comment_words(code: str) -> int:
    try:
        comments = [
            token.string
            for token in tokenize.generate_tokens(io.StringIO(code).readline)
            if token.type == tokenize.COMMENT
        ]
    except (IndentationError, tokenize.TokenError, SyntaxError):
        comments = COMMENT_FALLBACK_RE.findall(code)
    return len(' '.join(comments).split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', help='result directories to scan')
    args = parser.parse_args()

    states = [path for root in args.roots for path in Path(root).rglob('tree_state.json')]
    total, flagged = 0, {threshold: 0 for threshold in FLAG_THRESHOLDS}
    # Flip analysis under a 2x word->token safety margin: a node counts as
    # "comment-decided" if its raw base block exceeds half the token budget
    # while the comment-stripped block stays below it.
    near_budget = stripped_fits = 0
    max_raw_words = max_raw_chars = 0
    worst = []
    for path in sorted(states):
        try:
            nodes = json.loads(path.read_text())['nodes']
        except (KeyError, json.JSONDecodeError, OSError) as error:
            print(f'skip {path}: {error}')
            continue
        for record in nodes:
            code = record.get('code') or ''
            idea = record.get('idea') or ''
            total += 1
            comments = comment_words(code)
            raw_block = 20 + len(idea.split()) + len(code.split())
            max_raw_words = max(max_raw_words, raw_block)
            max_raw_chars = max(max_raw_chars, len(code) + len(idea))
            if raw_block > PROMPT_BUDGET // 2:
                near_budget += 1
                if raw_block - comments <= PROMPT_BUDGET // 2:
                    stripped_fits += 1
            for threshold in FLAG_THRESHOLDS:
                if comments >= threshold:
                    flagged[threshold] += 1
            if comments >= FLAG_THRESHOLDS[0]:
                worst.append((comments, len(code.split()), str(path), record.get('id')))

    print(f'tree_states={len(states)} nodes={total} (word proxy, budget={PROMPT_BUDGET})')
    print(f'max raw base block: {max_raw_words} words, ~{max_raw_chars // 4} chars/4 tokens')
    print(f'raw base block > budget/2: {near_budget}; '
          f'of those, stripped block <= budget/2: {stripped_fits}')
    for threshold in FLAG_THRESHOLDS:
        print(f'nodes with comment_words >= {threshold}: {flagged[threshold]}')
    worst.sort(reverse=True)
    for words, code_words, path, node_id in worst[:20]:
        print(f'  comment_words={words} code_words={code_words} node={node_id} {path}')


if __name__ == '__main__':
    main()
