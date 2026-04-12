use phoenix_chunker::split_sentence_ranges;
use phoenix_types::{ChunkKind, ChunkSpan, PosTag, SentenceSpan, TextRange, TokenClass, TokenSpan};

pub(crate) fn tokenize(text: &str) -> super::TokenizedDocument {
    let mut tokens = Vec::with_capacity(text.len() / 5);
    let mut normalized_tokens = Vec::with_capacity(text.len() / 5);
    let mut chars = text.char_indices().peekable();
    while let Some((start, ch)) = chars.next() {
        if ch.is_whitespace() {
            continue;
        }

        if ch.is_alphanumeric() || ch == '\'' || ch == '-' {
            let mut end = start + ch.len_utf8();
            while let Some((next_ix, next)) = chars.peek().copied() {
                if next.is_alphanumeric() || next == '\'' || next == '-' {
                    chars.next();
                    end = next_ix + next.len_utf8();
                } else {
                    break;
                }
            }

            let surface = &text[start..end];
            let normalized = super::normalize_token_surface(surface);
            let token_class = if surface.chars().all(|value| value.is_ascii_digit()) {
                TokenClass::Number
            } else {
                TokenClass::Word
            };
            let capitalized = surface
                .chars()
                .next()
                .is_some_and(|value| value.is_uppercase());
            tokens.push(TokenSpan {
                range: super::to_range(start, end),
                token_class: Some(token_class),
                pos: Some(guess_pos(surface, &normalized, capitalized)),
                masked: false,
                capitalized,
            });
            normalized_tokens.push(normalized);
        } else {
            tokens.push(TokenSpan {
                range: super::to_range(start, start + ch.len_utf8()),
                token_class: Some(if ch.is_ascii_punctuation() {
                    TokenClass::Punctuation
                } else {
                    TokenClass::Symbol
                }),
                pos: Some(PosTag::Punctuation),
                masked: false,
                capitalized: false,
            });
            normalized_tokens.push(String::new());
        }
    }

    retag_with_context(text, &mut tokens);
    super::TokenizedDocument {
        tokens,
        normalized_tokens,
    }
}

pub(crate) fn sentence_spans(text: &str) -> Vec<SentenceSpan> {
    let mut spans = Vec::new();
    for (index, (start, end)) in split_sentence_ranges(text).into_iter().enumerate() {
        spans.push(SentenceSpan {
            index,
            range: super::to_range(start, end),
        });
    }
    if spans.is_empty() && !text.trim().is_empty() {
        let start = text.len().saturating_sub(text.trim_start().len());
        let end = text.trim_end().len();
        spans.push(SentenceSpan {
            index: 0,
            range: super::to_range(start, end),
        });
    }
    spans
}

pub(crate) fn build_chunks(
    text: &str,
    tokens: &[TokenSpan],
    normalized_tokens: &[String],
    sentences: &[SentenceSpan],
) -> Vec<ChunkSpan> {
    let mut chunks = Vec::with_capacity(tokens.len().saturating_mul(2));
    let mut token_start = 0usize;

    for sentence in sentences {
        while token_start < tokens.len() && tokens[token_start].range.end <= sentence.range.start {
            token_start += 1;
        }
        let mut token_end = token_start;
        while token_end < tokens.len() && tokens[token_end].range.start < sentence.range.end {
            token_end += 1;
        }
        if token_start >= token_end {
            continue;
        }

        let mut index = token_start;
        while index < token_end {
            if token_kind(tokens, index) == Some(TokenClass::Punctuation) {
                index += 1;
                continue;
            }

            if let Some((chunk, consumed)) =
                try_clause(text, tokens, normalized_tokens, index, token_end, sentence.index)
            {
                chunks.push(chunk);
                index += consumed;
                continue;
            }
            if let Some((chunk, consumed)) =
                try_preposition_phrase(tokens, index, token_end, sentence.index)
            {
                chunks.push(chunk);
                index += consumed;
                continue;
            }
            if let Some((chunk, consumed)) =
                try_verb_phrase(text, tokens, normalized_tokens, index, token_end, sentence.index)
            {
                chunks.push(chunk);
                index += consumed;
                continue;
            }
            if let Some((chunk, consumed)) = try_noun_phrase(tokens, index, token_end, sentence.index)
            {
                chunks.push(chunk);
                index += consumed;
                continue;
            }
            if let Some((chunk, consumed)) = try_adj_phrase(tokens, index, token_end, sentence.index)
            {
                chunks.push(chunk);
                index += consumed;
                continue;
            }

            index += 1;
        }

        chunks.push(ChunkSpan {
            kind: Some(ChunkKind::Clause),
            range: sentence.range,
            head: sentence.range,
            modifiers: Vec::new(),
            sentence_index: sentence.index,
        });
        token_start = token_end;
    }

    chunks.sort_by_key(|chunk| {
        (
            chunk.sentence_index,
            chunk.range.start,
            chunk.range.end,
            chunk_kind_rank(chunk.kind.as_ref()),
        )
    });
    chunks.dedup_by(|left, right| left.kind == right.kind && left.range == right.range);
    chunks
}

fn try_noun_phrase(
    tokens: &[TokenSpan],
    start: usize,
    limit: usize,
    sentence_index: usize,
) -> Option<(ChunkSpan, usize)> {
    let mut index = start;
    let mut modifiers = Vec::new();

    if token_pos(tokens, index) == Some(PosTag::Determiner) {
        modifiers.push(tokens[index].range);
        index += 1;
    }
    while index < limit && token_pos(tokens, index) == Some(PosTag::Adjective) {
        modifiers.push(tokens[index].range);
        index += 1;
    }

    let noun_start = index;
    while index < limit
        && token_pos(tokens, index).is_some_and(|tag| is_nominal(&tag))
        && token_kind(tokens, index) == Some(TokenClass::Word)
    {
        index += 1;
    }
    if noun_start == index {
        return None;
    }

    Some((
        ChunkSpan {
            kind: Some(ChunkKind::Np),
            range: combine_ranges(tokens[start].range, tokens[index - 1].range),
            head: tokens[index - 1].range,
            modifiers,
            sentence_index,
        },
        index - start,
    ))
}

fn try_verb_phrase(
    text: &str,
    tokens: &[TokenSpan],
    normalized_tokens: &[String],
    start: usize,
    limit: usize,
    sentence_index: usize,
) -> Option<(ChunkSpan, usize)> {
    let mut index = start;
    let mut modifiers = Vec::new();

    while index < limit && token_pos(tokens, index).is_some_and(|tag| matches!(tag, PosTag::Auxiliary | PosTag::Modal))
    {
        modifiers.push(tokens[index].range);
        index += 1;
    }
    while index < limit && token_pos(tokens, index) == Some(PosTag::Adverb) {
        modifiers.push(tokens[index].range);
        index += 1;
    }

    let head_index = if index < limit && is_verb_like(text, tokens, normalized_tokens, index) {
        let head = index;
        index += 1;
        Some(head)
    } else {
        None
    }?;

    while index < limit && token_pos(tokens, index) == Some(PosTag::Adverb) {
        modifiers.push(tokens[index].range);
        index += 1;
    }

    Some((
        ChunkSpan {
            kind: Some(ChunkKind::Vp),
            range: combine_ranges(tokens[start].range, tokens[index - 1].range),
            head: tokens[head_index].range,
            modifiers,
            sentence_index,
        },
        index - start,
    ))
}

fn try_preposition_phrase(
    tokens: &[TokenSpan],
    start: usize,
    limit: usize,
    sentence_index: usize,
) -> Option<(ChunkSpan, usize)> {
    if token_pos(tokens, start) != Some(PosTag::Preposition) {
        return None;
    }
    let (np, consumed) = try_noun_phrase(tokens, start + 1, limit, sentence_index)?;
    Some((
        ChunkSpan {
            kind: Some(ChunkKind::Pp),
            range: combine_ranges(tokens[start].range, np.range),
            head: tokens[start].range,
            modifiers: std::iter::once(np.head)
                .chain(np.modifiers.iter().copied())
                .collect(),
            sentence_index,
        },
        consumed + 1,
    ))
}

fn try_adj_phrase(
    tokens: &[TokenSpan],
    start: usize,
    limit: usize,
    sentence_index: usize,
) -> Option<(ChunkSpan, usize)> {
    let mut index = start;
    let mut modifiers = Vec::new();
    while index < limit && token_pos(tokens, index) == Some(PosTag::Adverb) {
        modifiers.push(tokens[index].range);
        index += 1;
    }
    if index >= limit || token_pos(tokens, index) != Some(PosTag::Adjective) || modifiers.is_empty() {
        return None;
    }
    Some((
        ChunkSpan {
            kind: Some(ChunkKind::AdjP),
            range: combine_ranges(tokens[start].range, tokens[index].range),
            head: tokens[index].range,
            modifiers,
            sentence_index,
        },
        index - start + 1,
    ))
}

fn try_clause(
    text: &str,
    tokens: &[TokenSpan],
    normalized_tokens: &[String],
    start: usize,
    limit: usize,
    sentence_index: usize,
) -> Option<(ChunkSpan, usize)> {
    if token_pos(tokens, start) != Some(PosTag::RelativePronoun) {
        return None;
    }
    let (vp, consumed) = try_verb_phrase(
        text,
        tokens,
        normalized_tokens,
        start + 1,
        limit,
        sentence_index,
    )?;
    let mut end_range = vp.range;
    let mut total = consumed + 1;
    if let Some((np, np_consumed)) = try_noun_phrase(tokens, start + 1 + consumed, limit, sentence_index)
    {
        end_range = np.range;
        total += np_consumed;
    }
    Some((
        ChunkSpan {
            kind: Some(ChunkKind::Clause),
            range: combine_ranges(tokens[start].range, end_range),
            head: vp.head,
            modifiers: vec![tokens[start].range],
            sentence_index,
        },
        total,
    ))
}

fn retag_with_context(text: &str, tokens: &mut [TokenSpan]) {
    for index in 0..tokens.len() {
        let current = token_pos(tokens, index).unwrap_or(PosTag::Other);
        let previous = index
            .checked_sub(1)
            .and_then(|value| token_pos(tokens, value))
            .unwrap_or(PosTag::Other);
        let next = tokens.get(index + 1).and_then(|_| token_pos(tokens, index + 1));

        if matches!(previous, PosTag::Determiner | PosTag::Adjective) && is_verbal(&current) {
            tokens[index].pos = Some(PosTag::Noun);
        }
        if previous == PosTag::Modal && is_nominal(&current) {
            tokens[index].pos = Some(PosTag::Verb);
        }
        if current == PosTag::Noun
            && previous == PosTag::Determiner
            && next.as_ref().is_some_and(is_nominal)
        {
            tokens[index].pos = Some(PosTag::Adjective);
        }
        if current == PosTag::ProperNoun && next == Some(PosTag::Verb) {
            tokens[index].pos = Some(PosTag::Noun);
        }
        if current == PosTag::Other {
            let surface = super::slice_or_empty(text, tokens[index].range);
            if surface.ends_with("ly") {
                tokens[index].pos = Some(PosTag::Adverb);
            }
        }
    }
}

fn guess_pos(surface: &str, normalized: &str, capitalized: bool) -> PosTag {
    if normalized.is_empty() {
        return PosTag::Other;
    }
    if matches!(surface, "." | "," | "!" | "?" | "(" | ")" | ":" | ";") {
        return PosTag::Punctuation;
    }
    if is_pronoun(normalized) {
        return PosTag::Pronoun;
    }
    if matches!(normalized, "who" | "that" | "which" | "whom") {
        return PosTag::RelativePronoun;
    }
    if matches!(normalized, "the" | "a" | "an" | "this" | "that" | "these" | "those") {
        return PosTag::Determiner;
    }
    if matches!(
        normalized,
        "in" | "on" | "at" | "to" | "from" | "with" | "into" | "over" | "under" | "around"
            | "through" | "after" | "before" | "for" | "of" | "by" | "near" | "within"
    ) {
        return PosTag::Preposition;
    }
    if matches!(normalized, "and" | "or" | "but" | "nor" | "yet") {
        return PosTag::Conjunction;
    }
    if matches!(normalized, "can" | "could" | "should" | "would" | "may" | "might" | "will") {
        return PosTag::Modal;
    }
    if matches!(
        normalized,
        "is" | "are" | "was" | "were" | "be" | "been" | "being" | "have" | "has" | "had"
            | "do" | "does" | "did"
    ) {
        return PosTag::Auxiliary;
    }
    if normalized.ends_with("ly") {
        return PosTag::Adverb;
    }
    if [
        "ous", "ful", "ive", "al", "less", "able", "ible", "ic", "ish", "ant", "ent",
    ]
    .iter()
    .any(|suffix| normalized.ends_with(suffix))
    {
        return PosTag::Adjective;
    }
    if super::is_verb_token(normalized)
        || normalized.ends_with("ed")
        || normalized.ends_with("ing")
        || matches!(normalized, "said" | "told" | "left" | "built" | "found" | "felt")
    {
        return PosTag::Verb;
    }
    if capitalized {
        return PosTag::ProperNoun;
    }
    PosTag::Noun
}

fn combine_ranges(start: TextRange, end: TextRange) -> TextRange {
    TextRange {
        start: start.start,
        end: end.end,
    }
}

fn chunk_kind_rank(kind: Option<&ChunkKind>) -> u8 {
    match kind {
        Some(ChunkKind::Clause) => 0,
        Some(ChunkKind::Np) => 1,
        Some(ChunkKind::Vp) => 2,
        Some(ChunkKind::Pp) => 3,
        Some(ChunkKind::AdjP) => 4,
        None => 5,
    }
}

fn token_pos(tokens: &[TokenSpan], index: usize) -> Option<PosTag> {
    tokens.get(index).and_then(|token| token.pos.clone())
}

fn token_kind(tokens: &[TokenSpan], index: usize) -> Option<TokenClass> {
    tokens.get(index).and_then(|token| token.token_class.clone())
}

fn is_pronoun(value: &str) -> bool {
    matches!(
        value,
        "he" | "him" | "his" | "she" | "her" | "hers" | "it" | "its" | "they" | "them"
            | "their" | "we" | "us" | "i" | "me" | "you"
    )
}

fn is_nominal(tag: &PosTag) -> bool {
    matches!(tag, PosTag::Noun | PosTag::Pronoun | PosTag::ProperNoun)
}

fn is_verbal(tag: &PosTag) -> bool {
    matches!(tag, PosTag::Verb | PosTag::Auxiliary | PosTag::Modal)
}

fn is_verb_like(
    text: &str,
    tokens: &[TokenSpan],
    normalized_tokens: &[String],
    index: usize,
) -> bool {
    matches!(token_pos(tokens, index), Some(PosTag::Verb | PosTag::Auxiliary | PosTag::Modal))
        || normalized_tokens
            .get(index)
            .is_some_and(|value| super::is_verb_token(value))
        || super::slice_or_empty(text, tokens[index].range).ends_with("ed")
}

#[cfg(test)]
mod tests {
    use phoenix_types::{ChunkKind, PosTag};

    use super::{build_chunks, sentence_spans, tokenize};

    #[test]
    fn sentence_splitter_respects_short_guards() {
        let text = "Dr. Luffy ran. Mr. Zoro stayed. Wow!";
        let spans = sentence_spans(text);
        assert_eq!(spans.len(), 3);
        assert_eq!(&text[spans[0].range.start as usize..spans[0].range.end as usize], "Dr. Luffy ran.");
    }

    #[test]
    fn tokenization_recovers_richer_pos_tags() {
        let tokenized = tokenize("The quick fox can move into harbor.");
        let tags = tokenized
            .tokens
            .iter()
            .map(|token| token.pos.clone().expect("tag"))
            .collect::<Vec<_>>();
        assert_eq!(tags[0], PosTag::Determiner);
        assert_eq!(tags[1], PosTag::Adjective);
        assert_eq!(tags[3], PosTag::Modal);
        assert_eq!(tags[5], PosTag::Preposition);
    }

    #[test]
    fn chunk_builder_recovers_np_vp_and_pp_spans() {
        let text = "The brave captain quickly moved into the harbor.";
        let tokenized = tokenize(text);
        let sentences = sentence_spans(text);
        let chunks = build_chunks(text, &tokenized.tokens, &tokenized.normalized_tokens, &sentences);

        assert!(chunks.iter().any(|chunk| chunk.kind == Some(ChunkKind::Np)));
        assert!(chunks.iter().any(|chunk| chunk.kind == Some(ChunkKind::Vp)));
        assert!(chunks.iter().any(|chunk| chunk.kind == Some(ChunkKind::Pp)));
        assert!(chunks.iter().any(|chunk| chunk.kind == Some(ChunkKind::Clause)));
    }
}
