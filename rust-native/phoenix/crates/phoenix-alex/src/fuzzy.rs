use std::collections::BTreeMap;

pub const MIN_FUZZY_LEN: usize = 4;
pub const MAX_STRING_LEN: usize = 32;

pub fn dl_within(a: &str, b: &str, max: usize) -> bool {
    if a == b {
        return true;
    }

    let la = a.chars().count();
    let lb = b.chars().count();
    if la.abs_diff(lb) > max || la > MAX_STRING_LEN || lb > MAX_STRING_LEN {
        return false;
    }

    let a_chars = a.chars().collect::<Vec<_>>();
    let b_chars = b.chars().collect::<Vec<_>>();
    let mut dp = vec![vec![0usize; lb + 1]; la + 1];

    for i in 0..=la {
        dp[i][0] = i;
    }
    for j in 0..=lb {
        dp[0][j] = j;
    }

    for i in 1..=la {
        let mut row_min = usize::MAX;
        for j in 1..=lb {
            let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
            let mut value = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                value = value.min(dp[i - 2][j - 2] + 1);
            }
            dp[i][j] = value;
            row_min = row_min.min(value);
        }
        if row_min > max {
            return false;
        }
    }

    dp[la][lb] <= max
}

pub fn tok_match(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    if a.len() < MIN_FUZZY_LEN || b.len() < MIN_FUZZY_LEN {
        return false;
    }
    if a.chars().next() != b.chars().next() {
        return false;
    }
    let max = if a.len().max(b.len()) <= 5 { 1 } else { 2 };
    dl_within(a, b, max)
}

pub fn find_matching_anchors<V>(token: &str, anchors: &BTreeMap<String, V>) -> Vec<String> {
    let Some(first) = token.chars().next() else {
        return Vec::new();
    };
    anchors
        .keys()
        .filter(|anchor| anchor.starts_with(first) && tok_match(token, anchor))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_matching_handles_small_typos() {
        assert!(tok_match("luffy", "luffu"));
        assert!(tok_match("zoroo", "zoro"));
        assert!(!tok_match("luffy", "muffy"));
    }
}
