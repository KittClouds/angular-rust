/// Minimal normalize helpers extracted from phoenix-alex.
/// Only the two functions needed by the sentence splitter.

const SENTENCE_GUARDS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
];

fn is_joiner(ch: char) -> bool {
    matches!(
        ch,
        '\'' | '\u{2019}'
            | '\u{2018}'
            | '-'
            | '\u{2013}'
            | '\u{2014}'
            | '\u{00B7}'
            | '.'
            | '_'
            | '/'
            | '#'
            | '&'
    )
}

fn lower_char(ch: char) -> char {
    match ch {
        '\u{2019}' | '\u{2018}' => '\'',
        '\u{2013}' | '\u{2014}' => '-',
        _ => ch.to_ascii_lowercase(),
    }
}

pub fn normalize_raw(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut last_was_space = true;

    for ch in text.chars() {
        let mapped = lower_char(ch);
        if mapped.is_ascii_alphanumeric() || mapped.is_whitespace() || is_joiner(mapped) {
            if mapped.is_whitespace() {
                if !last_was_space {
                    output.push(' ');
                    last_was_space = true;
                }
            } else {
                output.push(mapped);
                last_was_space = false;
            }
        } else if !last_was_space {
            output.push(' ');
            last_was_space = true;
        }
    }

    while output.ends_with(' ') {
        output.pop();
    }

    output
}

pub fn is_sentence_guard(token: &str) -> bool {
    SENTENCE_GUARDS.contains(&token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_keeps_joiners_and_collapses_spaces() {
        assert_eq!(normalize_raw("Monkey D. Luffy"), "monkey d. luffy");
        assert_eq!(normalize_raw("Jean-Luc  Picard"), "jean-luc picard");
    }

    #[test]
    fn sentence_guards_cover_common_abbreviations() {
        assert!(is_sentence_guard("dr"));
        assert!(is_sentence_guard("mr"));
        assert!(is_sentence_guard("e.g"));
        assert!(!is_sentence_guard("hello"));
    }
}
