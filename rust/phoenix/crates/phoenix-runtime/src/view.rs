use phoenix_types::{
    AnalyzeTextRequest, IngestDocument, IngestRequest, NoteId, QueryRequest, QueryTarget,
    ResolverEntitySeed, ScanArtifact, ScanRequest, ScopeKey, SessionId, StructureRequest,
    TemporalMarker,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScopeKeyView<'a> {
    pub world_id: Option<&'a str>,
    pub narrative_id: Option<&'a str>,
    pub folder_id: Option<&'a str>,
    pub folder_path: Option<&'a str>,
}

impl<'a> ScopeKeyView<'a> {
    pub fn to_owned(self) -> ScopeKey {
        ScopeKey {
            world_id: self.world_id.map(str::to_owned),
            narrative_id: self.narrative_id.map(str::to_owned),
            folder_id: self.folder_id.map(str::to_owned),
            folder_path: self.folder_path.map(str::to_owned),
        }
    }
}

impl<'a> From<&'a ScopeKey> for ScopeKeyView<'a> {
    fn from(value: &'a ScopeKey) -> Self {
        Self {
            world_id: value.world_id.as_deref(),
            narrative_id: value.narrative_id.as_deref(),
            folder_id: value.folder_id.as_deref(),
            folder_path: value.folder_path.as_deref(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IngestDocumentView<'a> {
    pub document_id: phoenix_types::DocumentId,
    pub note_id: Option<NoteId>,
    pub title: &'a str,
    pub text: &'a str,
    pub scope: ScopeKeyView<'a>,
}

impl<'a> IngestDocumentView<'a> {
    pub fn to_owned(&self) -> IngestDocument {
        IngestDocument {
            document_id: self.document_id.clone(),
            note_id: self.note_id.clone(),
            title: self.title.to_owned(),
            text: self.text.to_owned(),
            scope: self.scope.to_owned(),
        }
    }
}

impl<'a> From<&'a IngestDocument> for IngestDocumentView<'a> {
    fn from(value: &'a IngestDocument) -> Self {
        Self {
            document_id: value.document_id.clone(),
            note_id: value.note_id.clone(),
            title: &value.title,
            text: &value.text,
            scope: ScopeKeyView::from(&value.scope),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IngestRequestView<'a> {
    pub session_id: Option<SessionId>,
    pub documents: Vec<IngestDocumentView<'a>>,
    pub commit: bool,
}

impl<'a> IngestRequestView<'a> {
    pub fn to_owned(&self) -> IngestRequest {
        IngestRequest {
            session_id: self.session_id.clone(),
            documents: self
                .documents
                .iter()
                .map(IngestDocumentView::to_owned)
                .collect(),
            commit: self.commit,
        }
    }
}

impl<'a> From<&'a IngestRequest> for IngestRequestView<'a> {
    fn from(value: &'a IngestRequest) -> Self {
        Self {
            session_id: value.session_id.clone(),
            documents: value
                .documents
                .iter()
                .map(IngestDocumentView::from)
                .collect(),
            commit: value.commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryRequestView<'a> {
    pub session_id: Option<SessionId>,
    pub query: &'a str,
    pub scope: ScopeKeyView<'a>,
    pub targets: &'a [QueryTarget],
    pub limit: Option<usize>,
    pub temporal: Option<&'a TemporalMarker>,
    pub semantic_query_vector: Option<&'a [f32]>,
    pub include_candidate_graph: bool,
}

impl<'a> QueryRequestView<'a> {
    pub fn to_owned(self) -> QueryRequest {
        QueryRequest {
            session_id: self.session_id.clone(),
            query: self.query.to_owned(),
            scope: self.scope.to_owned(),
            targets: self.targets.to_vec(),
            limit: self.limit,
            temporal: self.temporal.cloned(),
            semantic_query_vector: self.semantic_query_vector.map(|values| {
                phoenix_types::SemanticQueryVector {
                    values: values.to_vec(),
                }
            }),
            include_candidate_graph: self.include_candidate_graph,
        }
    }
}

impl<'a> From<&'a QueryRequest> for QueryRequestView<'a> {
    fn from(value: &'a QueryRequest) -> Self {
        Self {
            session_id: value.session_id.clone(),
            query: &value.query,
            scope: ScopeKeyView::from(&value.scope),
            targets: &value.targets,
            limit: value.limit,
            temporal: value.temporal.as_ref(),
            semantic_query_vector: value
                .semantic_query_vector
                .as_ref()
                .map(|vector| vector.values.as_slice()),
            include_candidate_graph: value.include_candidate_graph,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnalyzeTextRequestView<'a> {
    pub text: &'a str,
}

impl<'a> AnalyzeTextRequestView<'a> {
    pub fn to_owned(self) -> AnalyzeTextRequest {
        AnalyzeTextRequest {
            text: self.text.to_owned(),
        }
    }
}

impl<'a> From<&'a AnalyzeTextRequest> for AnalyzeTextRequestView<'a> {
    fn from(value: &'a AnalyzeTextRequest) -> Self {
        Self { text: &value.text }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanRequestView<'a> {
    pub text: &'a str,
    pub scope: ScopeKeyView<'a>,
    pub session_id: Option<SessionId>,
    pub resolver_seed: &'a [ResolverEntitySeed],
}

impl<'a> ScanRequestView<'a> {
    pub fn to_owned(&self) -> ScanRequest {
        ScanRequest {
            text: self.text.to_owned(),
            scope: self.scope.to_owned(),
            session_id: self.session_id.clone(),
            resolver_seed: self.resolver_seed.to_vec(),
        }
    }
}

impl<'a> From<&'a ScanRequest> for ScanRequestView<'a> {
    fn from(value: &'a ScanRequest) -> Self {
        Self {
            text: &value.text,
            scope: ScopeKeyView::from(&value.scope),
            session_id: value.session_id.clone(),
            resolver_seed: &value.resolver_seed,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StructureRequestView<'a> {
    pub text: &'a str,
    pub scan: &'a ScanArtifact,
}

impl<'a> StructureRequestView<'a> {
    pub fn to_owned(self) -> StructureRequest {
        StructureRequest {
            text: self.text.to_owned(),
            scan: self.scan.clone(),
        }
    }
}

impl<'a> From<&'a StructureRequest> for StructureRequestView<'a> {
    fn from(value: &'a StructureRequest) -> Self {
        Self {
            text: &value.text,
            scan: &value.scan,
        }
    }
}
