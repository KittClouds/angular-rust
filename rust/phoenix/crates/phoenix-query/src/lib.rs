use compact_str::CompactString;
use phoenix_index::DeterministicIndex;
use phoenix_kernel::DeterministicKernel;
use phoenix_types::{LexicalSearchResult, QueryRequest};

#[derive(Clone, Debug, Default)]
pub struct QueryPlan {
    pub normalized_query: CompactString,
    pub lexical_limit: usize,
    pub include_candidate_graph: bool,
}

pub struct DeterministicQuery<'a> {
    pub index: &'a DeterministicIndex,
    pub kernel: &'a DeterministicKernel,
}

impl<'a> DeterministicQuery<'a> {
    pub fn plan(request: &QueryRequest) -> QueryPlan {
        QueryPlan {
            normalized_query: CompactString::from(request.query.trim().to_lowercase()),
            lexical_limit: request.limit.unwrap_or(20),
            include_candidate_graph: request.include_candidate_graph,
        }
    }

    pub fn lexical_recall(&self, request: &QueryRequest) -> LexicalSearchResult {
        let plan = Self::plan(request);
        self.index
            .search(plan.normalized_query.as_str(), &request.scope, plan.lexical_limit)
    }
}
