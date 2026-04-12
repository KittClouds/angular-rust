use phoenix_graph_kernel::{KernelGraphSnapshot, KernelViewRequest, PhoenixGraphKernel};
use phoenix_store_native_core::{PhoenixGraphPatchStore, PhoenixSemanticGraphPatchStore};
use phoenix_types::ScopeKey;

use crate::api::{load_projection_kernel, GraphQueryError};

pub struct ScopeQuerySession {
    scope: ScopeKey,
    kernel: PhoenixGraphKernel,
}

impl ScopeQuerySession {
    pub fn scope(&self) -> &ScopeKey {
        &self.scope
    }

    pub fn view_as_of(&self, request: KernelViewRequest) -> KernelGraphSnapshot {
        self.kernel.view_as_of(request)
    }

    pub(crate) fn kernel(&self) -> &PhoenixGraphKernel {
        &self.kernel
    }
}

pub fn open_scope_query_session<S>(
    store: &S,
    scope: &ScopeKey,
) -> Result<Option<ScopeQuerySession>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore,
{
    let Some(kernel) = load_projection_kernel(store, scope)? else {
        return Ok(None);
    };
    Ok(Some(ScopeQuerySession {
        scope: scope.clone(),
        kernel,
    }))
}
