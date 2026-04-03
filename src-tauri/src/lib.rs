mod phoenix_rpc;

use phoenix_rpc::{PhoenixApi, PhoenixApiImpl};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(taurpc::create_ipc_handler(
            PhoenixApiImpl::default().into_handler(),
        ))
        .run(tauri::generate_context!())
        .expect("error while running Phoenix Tauri shell");
}
