use std::env;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::thread;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

static ORT_DYLIB_INIT: Once = Once::new();

pub fn configure_preferred_ort_dylib_path() {
    ORT_DYLIB_INIT.call_once(|| {
        if env::var_os("ORT_DYLIB_PATH").is_some() {
            return;
        }
        if let Some(path) = default_ort_dylib_path(&project_root()) {
            env::set_var("ORT_DYLIB_PATH", path);
        }
    });
}

pub fn load_session(path: &Path) -> Result<Session, ort::Error> {
    load_session_with_intra_threads(path, recommended_thread_count())
}

pub fn load_session_with_intra_threads(
    path: &Path,
    intra_threads: usize,
) -> Result<Session, ort::Error> {
    configure_preferred_ort_dylib_path();
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_parallel_execution(false)?
        .with_inter_threads(1)?
        .with_intra_threads(intra_threads.max(1))?
        .commit_from_file(path)
}

pub fn recommended_thread_count() -> usize {
    thread::available_parallelism()
        .map(|count| count.get().min(8))
        .unwrap_or(1)
}

pub fn default_ort_dylib_path(project_root: &Path) -> Option<PathBuf> {
    [
        project_root
            .join("node_modules")
            .join("@huggingface")
            .join("transformers")
            .join("node_modules")
            .join("onnxruntime-node")
            .join("bin")
            .join("napi-v6")
            .join("win32")
            .join("x64")
            .join("onnxruntime.dll"),
        project_root
            .join("node_modules")
            .join("onnxruntime-node")
            .join("bin")
            .join("napi-v3")
            .join("win32")
            .join("x64")
            .join("onnxruntime.dll"),
    ]
    .into_iter()
    .find(|path| path.exists())
}

pub fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("project root")
        .to_path_buf()
}
