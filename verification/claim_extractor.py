"""Atomic Claim Decomposition module with subject coreference propagation and non-factual filtering."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ingestion.chunking import split_sentences_safe


@dataclass
class AtomicClaim:
    """Represents a single verifiable factual proposition extracted from an answer."""

    claim_id: str
    claim_text: str
    original_sentence: str
    sentence_index: int
    char_start: int
    char_end: int
    cited_doc_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "original_sentence": self.original_sentence,
            "sentence_index": self.sentence_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "cited_doc_ids": self.cited_doc_ids,
        }


NON_FACTUAL_PATTERNS = [
    re.compile(r"^\s*(i\s+think|in\s+my\s+opinion|i\s+believe|personally|feel\s+free\s+to|i\s+hope\s+this\s+helps)", re.IGNORECASE),
    re.compile(r"^\s*(hello|hi\s+there|sure[,!]|certainly[,!]|let\s+me\s+know\s+if)", re.IGNORECASE),
    re.compile(r"the\s+retrieved\s+evidence\s+is\s+insufficient\s+to\s+answer", re.IGNORECASE),
]

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]{3,40})\]")


class AtomicClaimExtractor:
    """Decomposes complex multi-clause responses into self-contained atomic factual claims."""

    @staticmethod
    def is_factual_sentence(sentence: str) -> bool:
        """Return False for conversational filler, pure opinions, or questions."""
        cleaned = CITATION_PATTERN.sub("", sentence).strip()
        if len(cleaned) < 8 or cleaned.endswith("?"):
            return False
        for pat in NON_FACTUAL_PATTERNS:
            if pat.search(cleaned):
                return False
        # Require at least 3 words
        if len(cleaned.split()) < 3:
            return False
        return True

    @staticmethod
    def _extract_nominal_subject(clause: str) -> str:
        """Extract the primary nominal entity or subject phrase from the first clause."""
        cleaned = CITATION_PATTERN.sub("", clause).strip()
        # Look for direct object of "launched X in YYYY" or capitalized entity before main verb
        launch_match = re.search(
            r"\b(?:launched|developed|introduced|deployed|built|created)\s+([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*)",
            cleaned,
        )
        if launch_match:
            return launch_match.group(1).strip()

        # Otherwise extract leading capitalized noun phrase or subject before auxiliary/verb
        subj_match = re.match(
            r"^([A-Z][A-Za-z0-9_-]*(?:\s+(?:of|the|for|and|[A-Z][A-Za-z0-9_-]*))*?)\s+"
            r"(?:is|are|was|were|has|have|had|launched|landed|achieved|reported|recorded|uses|used|showed|demonstrated|operates|contains|reached|reduced|increased)\b",
            cleaned,
        )
        if subj_match:
            return subj_match.group(1).strip()

        words = cleaned.split()
        cap_words: List[str] = []
        for w in words[:4]:
            if w[0].isupper():
                cap_words.append(w.rstrip(",.:;"))
            elif cap_words:
                break
        return " ".join(cap_words) if cap_words else ""

    def _split_into_atomic_clauses(self, sentence: str) -> List[str]:
        """Split compound sentences into atomic claims and resolve pronoun subjects."""
        no_cite = CITATION_PATTERN.sub("", sentence).strip().rstrip(".")
        # Split on coordinating conjunctions (", and ", ", while ", ", whereas ", ", but ", "; ")
        parts = re.split(
            r"(?:,\s+(?:and|but|while|whereas)\s+|;\s+|\s+and\s+(?=it\s+|they\s+|he\s+|she\s+))",
            no_cite,
        )
        if len(parts) <= 1:
            return [no_cite.strip() + "."]

        primary_subject = self._extract_nominal_subject(parts[0])
        atomic_clauses: List[str] = []

        for idx, raw_part in enumerate(parts):
            clause = raw_part.strip().rstrip(".")
            if not clause:
                continue
            if idx > 0 and primary_subject:
                # Replace leading pronoun ("it ", "they ", "this mission ") with resolved nominal subject
                pronoun_match = re.match(r"^(it|they|he|she|this\s+mission|this\s+system)\s+(.+)$", clause, flags=re.IGNORECASE)
                if pronoun_match:
                    clause = f"{primary_subject} {pronoun_match.group(2)}"
                elif re.match(r"^(successfully\s+|subsequently\s+|later\s+)?(landed|achieved|reached|failed|operated|orbit|detected|measured)\b", clause, flags=re.IGNORECASE):
                    clause = f"{primary_subject} {clause}"
            # Ensure clause has enough substance to stand alone
            if len(clause.split()) >= 3:
                clause_cap = clause[0].upper() + clause[1:] if clause else clause
                atomic_clauses.append(clause_cap + ".")
            elif atomic_clauses:
                atomic_clauses[-1] = atomic_clauses[-1].rstrip(".") + ", and " + clause + "."

        return atomic_clauses if atomic_clauses else [no_cite.strip() + "."]

    def extract_claims(
        self,
        answer_text: str,
        decompose: bool = True,
    ) -> List[AtomicClaim]:
        """Extract traceable atomic factual claims from an LLM-generated answer."""
        if not answer_text or not answer_text.strip():
            return []

        if not decompose:
            # Ablation mode: treat entire answer as a single claim
            cites = CITATION_PATTERN.findall(answer_text)
            cleaned_full = CITATION_PATTERN.sub("", answer_text).strip()
            if not cleaned_full:
                return []
            return [
                AtomicClaim(
                    claim_id="CLM-001",
                    claim_text=cleaned_full,
                    original_sentence=answer_text.strip(),
                    sentence_index=0,
                    char_start=0,
                    char_end=len(answer_text),
                    cited_doc_ids=list(dict.fromkeys(cites)),
                )
            ]

        sentences = split_sentences_safe(answer_text)
        claims: List[AtomicClaim] = []
        cursor = 0
        claim_counter = 1

        for s_idx, sent in enumerate(sentences):
            pos = answer_text.find(sent, cursor)
            if pos == -1:
                pos = cursor
            end_pos = pos + len(sent)
            cursor = end_pos

            if not self.is_factual_sentence(sent):
                continue

            sent_citations = list(dict.fromkeys(CITATION_PATTERN.findall(sent)))
            clauses = self._split_into_atomic_clauses(sent)
            for clause in clauses:
                claims.append(
                    AtomicClaim(
                        claim_id=f"CLM-{claim_counter:03d}",
                        claim_text=clause,
                        original_sentence=sent,
                        sentence_index=s_idx,
                        char_start=pos,
                        char_end=end_pos,
                        cited_doc_ids=sent_citations,
                    )
                )
                claim_counter += 1

        return claims
