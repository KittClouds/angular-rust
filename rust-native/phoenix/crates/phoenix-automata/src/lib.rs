use aho_corasick::{AhoCorasick, MatchKind};
use compact_str::CompactString;
use phoenix_alex::{AlexError, ExactSurfacePattern, Lexicon};
use phoenix_types::{CausalKind, LexiconEntry, LexiconSnapshot, SourceRange};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AutomataError {
    #[error("failed to build lexicon: {0}")]
    Lexicon(#[from] AlexError),
    #[error("failed to build aho-corasick matcher")]
    PatternBuild,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PatternHit {
    pub pattern: CompactString,
    pub range: SourceRange,
}

pub struct PatternBank {
    matcher: AhoCorasick,
    patterns: Vec<CompactString>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CausalCueDirection {
    LeftToRight,
    RightToLeft,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CausalCueRole {
    Cause,
    Effect,
    Blocker,
    Blocked,
    Motivation,
    Goal,
    Explanation,
    Condition,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalCueFrame {
    pub cue: CompactString,
    pub kind: CausalKind,
    pub direction: CausalCueDirection,
    pub left_role: CausalCueRole,
    pub right_role: CausalCueRole,
    pub requires_clause_boundary: bool,
    pub allows_nominal_args: bool,
    pub priority: u16,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalCueHit {
    pub frame: CausalCueFrame,
    pub range: SourceRange,
}

pub struct CausalCueMatcher {
    matcher: AhoCorasick,
    frames: Vec<CausalCueFrame>,
}

impl Default for CausalCueMatcher {
    fn default() -> Self {
        Self::new().expect("causal cue matcher should build")
    }
}

impl CausalCueMatcher {
    pub fn new() -> Result<Self, AutomataError> {
        let frames = default_causal_cue_frames();
        let matcher = AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .match_kind(MatchKind::LeftmostLongest)
            .build(frames.iter().map(|frame| frame.cue.as_str()))
            .map_err(|_| AutomataError::PatternBuild)?;
        Ok(Self { matcher, frames })
    }

    pub fn find_iter(&self, haystack: &str) -> Vec<CausalCueHit> {
        self.matcher
            .find_iter(haystack)
            .map(|hit| CausalCueHit {
                frame: self.frames[hit.pattern()].clone(),
                range: SourceRange::new(hit.start() as u32, hit.end() as u32),
            })
            .collect()
    }
}

pub fn default_causal_cue_frames() -> Vec<CausalCueFrame> {
    use CausalCueDirection::{LeftToRight, RightToLeft};
    use CausalCueRole::{
        Blocked, Blocker, Cause, Condition, Effect, Explanation, Goal, Motivation,
    };

    vec![
        cue_frame(
            "because",
            CausalKind::Causes,
            RightToLeft,
            Effect,
            Cause,
            1000,
        ),
        cue_frame(
            "because of",
            CausalKind::Causes,
            RightToLeft,
            Effect,
            Cause,
            1005,
        ),
        cue_frame(
            "due to",
            CausalKind::Causes,
            RightToLeft,
            Effect,
            Cause,
            990,
        ),
        cue_frame(
            "owing to",
            CausalKind::Explains,
            RightToLeft,
            Effect,
            Explanation,
            920,
        ),
        cue_frame(
            "caused by",
            CausalKind::Causes,
            RightToLeft,
            Effect,
            Cause,
            1000,
        ),
        cue_frame(
            "led to",
            CausalKind::ResultsIn,
            LeftToRight,
            Cause,
            Effect,
            960,
        ),
        cue_frame(
            "resulted in",
            CausalKind::ResultsIn,
            LeftToRight,
            Cause,
            Effect,
            960,
        ),
        cue_frame(
            "triggered",
            CausalKind::TriggerFor,
            LeftToRight,
            Cause,
            Effect,
            950,
        ),
        cue_frame(
            "enabled",
            CausalKind::Enables,
            LeftToRight,
            Cause,
            Effect,
            940,
        ),
        cue_frame(
            "allowed",
            CausalKind::Enables,
            LeftToRight,
            Cause,
            Effect,
            930,
        ),
        cue_frame(
            "prevented",
            CausalKind::Prevents,
            LeftToRight,
            Blocker,
            Blocked,
            970,
        ),
        cue_frame(
            "blocked",
            CausalKind::Prevents,
            LeftToRight,
            Blocker,
            Blocked,
            965,
        ),
        cue_frame(
            "in order to",
            CausalKind::PurposeOf,
            LeftToRight,
            Motivation,
            Goal,
            930,
        ),
        cue_frame(
            "so that",
            CausalKind::PurposeOf,
            LeftToRight,
            Motivation,
            Goal,
            900,
        ),
        cue_frame(
            "so as to",
            CausalKind::PurposeOf,
            LeftToRight,
            Motivation,
            Goal,
            900,
        ),
        cue_frame(
            "wanted to",
            CausalKind::Motivates,
            LeftToRight,
            Motivation,
            Goal,
            880,
        ),
        cue_frame(
            "therefore",
            CausalKind::Explains,
            LeftToRight,
            Cause,
            Effect,
            860,
        ),
        cue_frame(
            "thus",
            CausalKind::Explains,
            LeftToRight,
            Cause,
            Effect,
            850,
        ),
        cue_frame(
            "hence",
            CausalKind::Explains,
            LeftToRight,
            Cause,
            Effect,
            850,
        ),
        cue_frame(
            "as a result",
            CausalKind::ResultsIn,
            LeftToRight,
            Cause,
            Effect,
            900,
        ),
        cue_frame(
            "consequently",
            CausalKind::ResultsIn,
            LeftToRight,
            Cause,
            Effect,
            890,
        ),
        CausalCueFrame {
            cue: CompactString::from("if"),
            kind: CausalKind::ConditionFor,
            direction: LeftToRight,
            left_role: Condition,
            right_role: Effect,
            requires_clause_boundary: true,
            allows_nominal_args: false,
            priority: 780,
        },
    ]
}

fn cue_frame(
    cue: &str,
    kind: CausalKind,
    direction: CausalCueDirection,
    left_role: CausalCueRole,
    right_role: CausalCueRole,
    priority: u16,
) -> CausalCueFrame {
    CausalCueFrame {
        cue: CompactString::from(cue),
        kind,
        direction,
        left_role,
        right_role,
        requires_clause_boundary: false,
        allows_nominal_args: true,
        priority,
    }
}

impl PatternBank {
    pub fn new(patterns: impl IntoIterator<Item = impl AsRef<str>>) -> Result<Self, AutomataError> {
        let patterns = patterns
            .into_iter()
            .map(|pattern| CompactString::from(pattern.as_ref()))
            .collect::<Vec<_>>();
        let matcher = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(patterns.iter().map(|pattern| pattern.as_str()))
            .map_err(|_| AutomataError::PatternBuild)?;
        Ok(Self { matcher, patterns })
    }

    pub fn find_iter(&self, haystack: &str) -> Vec<PatternHit> {
        self.matcher
            .find_iter(haystack)
            .map(|hit| PatternHit {
                pattern: self.patterns[hit.pattern()].clone(),
                range: SourceRange::new(hit.start() as u32, hit.end() as u32),
            })
            .collect()
    }
}

pub struct LexiconBank {
    inner: Lexicon,
}

impl LexiconBank {
    pub fn from_entries(entries: &[LexiconEntry]) -> Result<Self, AutomataError> {
        Ok(Self {
            inner: Lexicon::from_entries(entries)?,
        })
    }

    pub fn from_snapshot(snapshot: LexiconSnapshot) -> Result<Self, AutomataError> {
        Ok(Self {
            inner: Lexicon::from_snapshot(snapshot)?,
        })
    }

    pub fn snapshot(&self) -> LexiconSnapshot {
        self.inner.to_snapshot()
    }

    pub fn exact_surface_patterns(&self) -> Vec<ExactSurfacePattern> {
        self.inner.exact_surface_patterns()
    }

    pub fn inner(&self) -> &Lexicon {
        &self.inner
    }
}
