use phoenix_alex::normalize_raw;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClauseType {
    Term,
    Phrase,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clause {
    pub pattern: String,
    pub clause_type: ClauseType,
    pub raw_input: String,
}

pub fn parse_query(input: &str) -> Vec<Clause> {
    let mut clauses = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;

    let flush_term = |clauses: &mut Vec<Clause>, current: &mut String| {
        if current.is_empty() {
            return;
        }
        let raw = std::mem::take(current);
        let pattern = normalize_raw(&raw);
        if !pattern.is_empty() {
            clauses.push(Clause {
                pattern,
                clause_type: ClauseType::Term,
                raw_input: raw,
            });
        }
    };

    for ch in input.chars() {
        if ch == '"' {
            if in_quote {
                let raw = std::mem::take(&mut current);
                let pattern = normalize_raw(&raw);
                if !pattern.is_empty() {
                    clauses.push(Clause {
                        pattern,
                        clause_type: ClauseType::Phrase,
                        raw_input: raw,
                    });
                }
                in_quote = false;
            } else {
                flush_term(&mut clauses, &mut current);
                in_quote = true;
            }
        } else if ch.is_whitespace() && !in_quote {
            flush_term(&mut clauses, &mut current);
        } else {
            current.push(ch);
        }
    }

    if !current.is_empty() {
        let raw = std::mem::take(&mut current);
        let pattern = normalize_raw(&raw);
        if !pattern.is_empty() {
            clauses.push(Clause {
                pattern,
                clause_type: if in_quote {
                    ClauseType::Term
                } else {
                    ClauseType::Term
                },
                raw_input: raw,
            });
        }
    }

    clauses
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_query_keeps_phrase_and_unclosed_term_semantics() {
        let clauses = parse_query(r#"term "quoted phrase" tail"#);
        assert_eq!(clauses.len(), 3);
        assert_eq!(clauses[0].pattern, "term");
        assert_eq!(clauses[1].pattern, "quoted phrase");
        assert_eq!(clauses[1].clause_type, ClauseType::Phrase);
        assert_eq!(clauses[2].pattern, "tail");

        let unclosed = parse_query(r#"mixed "quote logic"#);
        assert_eq!(unclosed.len(), 2);
        assert_eq!(unclosed[1].clause_type, ClauseType::Term);
        assert_eq!(unclosed[1].pattern, "quote logic");
    }
}
