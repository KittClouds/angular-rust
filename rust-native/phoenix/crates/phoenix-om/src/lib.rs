use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

use phoenix_scanner::PhoenixScanner;
use phoenix_store_cozo::{CompactRowView, PhoenixCozoStore, StoreError};
use phoenix_structure::PhoenixStructure;
use phoenix_types::{
    ChatRuntimeConfig, OmConfig, OmGeneration, OmObservationResult, OmPendingAction, OmRecord,
    OmReflectorMessage, OmReflectorModelRequest, OmReflectorModelResponse, OmReflectorSession,
    OmReflectorStep, OmReflectorToolCall, OmReflectorToolResult, OmReflectorToolSpec,
    ThreadMessage,
};
use serde_json::{json, Value};

mod bridge;
pub use bridge::OmNativeBridge;

#[cfg(not(target_arch = "wasm32"))]
use openrouter_rs::{
    api::chat::{ChatCompletionRequest, Message},
    types::{Role, Tool},
    OpenRouterClient,
};

const OBSERVE_KIND: &str = "observe";
const REFLECT_KIND: &str = "reflect";
const REFLECT_TOOL_RECOVER_LOST_MEMORY: &str = "recover_lost_memory";
const REFLECT_TOOL_MEMORY_GRAPH_SEARCH: &str = "memory_graph_search";
const FINAL_REFLECTOR_PROMPT: &str =
    "Produce the final reflected observational memory now. Return only XML with <observations> and any optional supported tags.";
const OM_RECORD_COLUMNS: &[&str] = &[
    "thread_id",
    "observations",
    "current_task",
    "suggested_continuation",
    "last_observed_at",
    "obs_token_count",
    "generation_num",
    "created_at",
    "updated_at",
];
const THREAD_MESSAGE_COLUMNS: &[&str] = &[
    "id",
    "thread_id",
    "role",
    "content",
    "narrative_id",
    "created_at",
    "updated_at",
    "is_streaming",
    "token_count",
    "is_observed",
];

#[derive(Debug, thiserror::Error)]
pub enum OmError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("unsupported OM action kind: {0}")]
    UnsupportedAction(String),
    #[error("OM transport error: {0}")]
    Transport(String),
}

pub trait OmTransport {
    fn observe(&self, action: &OmPendingAction) -> Result<String, OmError>;
    fn reflect(&self, action: &OmPendingAction) -> Result<String, OmError>;

    fn reflect_model(
        &self,
        _request: &OmReflectorModelRequest,
    ) -> Result<OmReflectorModelResponse, OmError> {
        Err(OmError::Transport(
            "reflector model requests unsupported".to_owned(),
        ))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OmProcessReport {
    pub already_running: bool,
    pub mutated: bool,
}

#[derive(Default)]
pub struct OmReflectorRunner {
    sessions: Mutex<HashMap<String, OmReflectorSession>>,
    next_session_id: AtomicU64,
}

impl OmReflectorRunner {
    pub fn start(&self, action: &OmPendingAction) -> Result<OmReflectorStep, OmError> {
        if action.kind != REFLECT_KIND {
            return Err(OmError::UnsupportedAction(action.kind.clone()));
        }
        let session_id = format!(
            "om-reflector-{}-{}",
            now_ms(),
            self.next_session_id.fetch_add(1, Ordering::Relaxed)
        );
        let now = now_ms();
        let messages = vec![
            OmReflectorMessage {
                role: "system".to_owned(),
                content: action.system_prompt.clone(),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            },
            OmReflectorMessage {
                role: "user".to_owned(),
                content: action.user_prompt.clone(),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            },
        ];
        let session = OmReflectorSession {
            session_id: session_id.clone(),
            thread_id: action.thread_id.clone(),
            model: action.model.clone(),
            tool_rounds_used: 0,
            max_tool_rounds: action.reflector_max_tool_rounds,
            final_request_sent: false,
            awaiting_tool_results: false,
            messages,
            created_at: now,
            updated_at: now,
        };
        let step = build_reflector_model_step(
            &session,
            action.reflector_tooling_enabled && action.reflector_max_tool_rounds > 0,
        );
        self.sessions
            .lock()
            .expect("om reflector sessions poisoned")
            .insert(session_id, session);
        Ok(step)
    }

    pub fn submit_model_response(
        &self,
        session_id: &str,
        response: OmReflectorModelResponse,
    ) -> Result<OmReflectorStep, OmError> {
        let mut sessions = self
            .sessions
            .lock()
            .expect("om reflector sessions poisoned");
        let mut session = sessions.remove(session_id).ok_or_else(|| {
            OmError::Transport(format!("unknown reflector session: {session_id}"))
        })?;
        if session.awaiting_tool_results {
            sessions.insert(session_id.to_owned(), session);
            return Err(OmError::Transport(
                "reflector session is waiting for tool results".to_owned(),
            ));
        }

        if !response.tool_calls.is_empty() {
            session.messages.push(OmReflectorMessage {
                role: "assistant".to_owned(),
                content: response.content,
                name: None,
                tool_call_id: None,
                tool_calls: response.tool_calls.clone(),
            });
            session.awaiting_tool_results = true;
            session.updated_at = now_ms();
            let thread_id = session.thread_id.clone();
            sessions.insert(session_id.to_owned(), session);
            return Ok(OmReflectorStep::ToolCalls {
                session_id: session_id.to_owned(),
                thread_id,
                tool_calls: response.tool_calls,
            });
        }

        if !response.content.trim().is_empty() {
            session.messages.push(OmReflectorMessage {
                role: "assistant".to_owned(),
                content: response.content.clone(),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            });
        }
        if looks_like_reflector_xml(&response.content) {
            return Ok(OmReflectorStep::Complete {
                session_id: session_id.to_owned(),
                thread_id: session.thread_id,
                response: response.content,
            });
        }
        if session.final_request_sent {
            return Err(OmError::Transport(
                "reflector did not produce a final XML response".to_owned(),
            ));
        }

        session.messages.push(OmReflectorMessage {
            role: "user".to_owned(),
            content: FINAL_REFLECTOR_PROMPT.to_owned(),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        });
        session.final_request_sent = true;
        session.updated_at = now_ms();
        let step = build_reflector_model_step(&session, false);
        sessions.insert(session_id.to_owned(), session);
        Ok(step)
    }

    pub fn submit_tool_results(
        &self,
        session_id: &str,
        results: &[OmReflectorToolResult],
    ) -> Result<OmReflectorStep, OmError> {
        let mut sessions = self
            .sessions
            .lock()
            .expect("om reflector sessions poisoned");
        let mut session = sessions.remove(session_id).ok_or_else(|| {
            OmError::Transport(format!("unknown reflector session: {session_id}"))
        })?;
        if !session.awaiting_tool_results {
            sessions.insert(session_id.to_owned(), session);
            return Err(OmError::Transport(
                "reflector session is not waiting for tool results".to_owned(),
            ));
        }

        for result in results {
            session.messages.push(OmReflectorMessage {
                role: "tool".to_owned(),
                content: result.result_json.clone(),
                name: Some(result.name.clone()),
                tool_call_id: Some(result.tool_call_id.clone()),
                tool_calls: Vec::new(),
            });
        }
        session.awaiting_tool_results = false;
        session.tool_rounds_used = session.tool_rounds_used.saturating_add(1);
        session.updated_at = now_ms();
        let allow_tools =
            !session.final_request_sent && session.tool_rounds_used < session.max_tool_rounds;
        if !allow_tools && !session.final_request_sent {
            session.messages.push(OmReflectorMessage {
                role: "user".to_owned(),
                content: FINAL_REFLECTOR_PROMPT.to_owned(),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            });
            session.final_request_sent = true;
        }
        let step = build_reflector_model_step(&session, allow_tools);
        sessions.insert(session_id.to_owned(), session);
        Ok(step)
    }

    pub fn drop_session(&self, session_id: &str) -> bool {
        self.sessions
            .lock()
            .expect("om reflector sessions poisoned")
            .remove(session_id)
            .is_some()
    }
}

#[derive(Default)]
pub struct OmEngine {
    active_threads: Mutex<HashSet<String>>,
    reflector_runner: OmReflectorRunner,
}

impl OmEngine {
    pub fn config_from_runtime(config: &ChatRuntimeConfig) -> OmConfig {
        OmConfig {
            enabled: config.om_enabled,
            model: config
                .om_model
                .clone()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| config.model.clone()),
            observe_threshold: config.observe_threshold.unwrap_or(2_000),
            reflect_threshold: config.reflect_threshold.unwrap_or(4_000),
            graph_index_enabled: true,
            index_raw_messages: true,
            index_observations: true,
            index_reflections: true,
            reflector_tooling_enabled: true,
            reflector_max_tool_rounds: 2,
        }
    }

    pub fn build_context_block(
        &self,
        store: &PhoenixCozoStore,
        thread_id: &str,
    ) -> Result<Option<String>, OmError> {
        let Some(record) = self.load_record(store, thread_id)? else {
            return Ok(None);
        };
        let observations = record.observations.trim();
        if observations.is_empty() {
            return Ok(None);
        }

        let suggested = record
            .suggested_continuation
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let current_task = record.current_task.trim();
        let mut block = String::with_capacity(
            observations.len()
                + current_task.len()
                + suggested.map(str::len).unwrap_or_default()
                + 64,
        );
        block.push_str("Observational memory\n\n");
        block.push_str(observations);
        if !current_task.is_empty() {
            block.push_str("\n\nCurrent task: ");
            block.push_str(current_task);
        }
        if let Some(suggested) = suggested {
            block.push_str("\n\nSuggested continuation: ");
            block.push_str(suggested);
        }
        Ok(Some(block))
    }

    pub fn start_reflector(&self, action: &OmPendingAction) -> Result<OmReflectorStep, OmError> {
        self.reflector_runner.start(action)
    }

    pub fn submit_reflector_model_response(
        &self,
        session_id: &str,
        response: OmReflectorModelResponse,
    ) -> Result<OmReflectorStep, OmError> {
        self.reflector_runner
            .submit_model_response(session_id, response)
    }

    pub fn submit_reflector_tool_results(
        &self,
        session_id: &str,
        results: &[OmReflectorToolResult],
    ) -> Result<OmReflectorStep, OmError> {
        self.reflector_runner
            .submit_tool_results(session_id, results)
    }

    pub fn drop_reflector_session(&self, session_id: &str) -> bool {
        self.reflector_runner.drop_session(session_id)
    }

    pub fn prepare_pending_action(
        &self,
        store: &PhoenixCozoStore,
        thread_id: &str,
        config: &OmConfig,
    ) -> Result<Option<OmPendingAction>, OmError> {
        if !config.enabled || config.model.trim().is_empty() {
            return Ok(None);
        }

        let record = self.get_or_create_record(store, thread_id)?;
        if record.obs_token_count > i64::from(config.reflect_threshold) {
            return Ok(Some(OmPendingAction {
                kind: REFLECT_KIND.to_owned(),
                thread_id: thread_id.to_owned(),
                model: config.model.clone(),
                system_prompt: build_reflector_system_prompt(),
                user_prompt: build_reflector_user_prompt(&record.observations),
                message_ids: Vec::new(),
                reflector_tooling_enabled: config.reflector_tooling_enabled,
                reflector_max_tool_rounds: config.reflector_max_tool_rounds,
            }));
        }

        let eligible = self.eligible_messages(store, thread_id, &record)?;
        if eligible.is_empty() {
            return Ok(None);
        }

        let total_tokens: i64 = eligible
            .iter()
            .map(|message| {
                message
                    .token_count
                    .unwrap_or_else(|| approx_token_count(&message.content))
            })
            .sum();

        let threshold = effective_observe_threshold(config, record.obs_token_count);
        if total_tokens < threshold {
            return Ok(None);
        }

        Ok(Some(OmPendingAction {
            kind: OBSERVE_KIND.to_owned(),
            thread_id: thread_id.to_owned(),
            model: config.model.clone(),
            system_prompt: build_observer_system_prompt(),
            user_prompt: build_observer_user_prompt(&record, &eligible),
            message_ids: eligible.into_iter().map(|message| message.id).collect(),
            reflector_tooling_enabled: false,
            reflector_max_tool_rounds: 0,
        }))
    }

    pub fn prepare_pending_action_with_graph(
        &self,
        store: &PhoenixCozoStore,
        scanner: &PhoenixScanner,
        structure: &PhoenixStructure,
        bridge: &OmNativeBridge,
        thread_id: &str,
        config: &OmConfig,
    ) -> Result<Option<OmPendingAction>, OmError> {
        if !config.enabled || config.model.trim().is_empty() {
            return Ok(None);
        }

        let record = self.get_or_create_record(store, thread_id)?;
        if record.obs_token_count > i64::from(config.reflect_threshold) {
            return Ok(Some(OmPendingAction {
                kind: REFLECT_KIND.to_owned(),
                thread_id: thread_id.to_owned(),
                model: config.model.clone(),
                system_prompt: build_reflector_system_prompt(),
                user_prompt: build_reflector_user_prompt(&record.observations),
                message_ids: Vec::new(),
                reflector_tooling_enabled: config.reflector_tooling_enabled,
                reflector_max_tool_rounds: config.reflector_max_tool_rounds,
            }));
        }

        let eligible = self.eligible_messages(store, thread_id, &record)?;
        if eligible.is_empty() {
            return Ok(None);
        }

        let total_tokens: i64 = eligible
            .iter()
            .map(|message| {
                message
                    .token_count
                    .unwrap_or_else(|| approx_token_count(&message.content))
            })
            .sum();
        let threshold = effective_observe_threshold(config, record.obs_token_count);
        if total_tokens < threshold {
            return Ok(None);
        }

        if config.graph_index_enabled && config.index_raw_messages {
            let source_key = message_window_source_key(&eligible);
            bridge
                .index_message_window(store, scanner, structure, thread_id, &source_key, &eligible)
                .map_err(|error| OmError::Transport(error.to_string()))?;
        }

        Ok(Some(OmPendingAction {
            kind: OBSERVE_KIND.to_owned(),
            thread_id: thread_id.to_owned(),
            model: config.model.clone(),
            system_prompt: build_observer_system_prompt(),
            user_prompt: build_observer_user_prompt(&record, &eligible),
            message_ids: eligible.into_iter().map(|message| message.id).collect(),
            reflector_tooling_enabled: false,
            reflector_max_tool_rounds: 0,
        }))
    }

    pub fn apply_pending_action(
        &self,
        store: &PhoenixCozoStore,
        action: &OmPendingAction,
        response: &str,
    ) -> Result<bool, OmError> {
        match action.kind.as_str() {
            OBSERVE_KIND => self.apply_observation(store, action, response),
            REFLECT_KIND => self.apply_reflection(store, action, response),
            other => Err(OmError::UnsupportedAction(other.to_owned())),
        }
    }

    pub fn apply_pending_action_with_graph(
        &self,
        store: &PhoenixCozoStore,
        scanner: &PhoenixScanner,
        structure: &PhoenixStructure,
        bridge: &OmNativeBridge,
        config: &OmConfig,
        action: &OmPendingAction,
        response: &str,
    ) -> Result<bool, OmError> {
        match action.kind.as_str() {
            OBSERVE_KIND => self.apply_observation_with_graph(
                store, scanner, structure, bridge, config, action, response,
            ),
            REFLECT_KIND => self.apply_reflection_with_graph(
                store, scanner, structure, bridge, config, action, response,
            ),
            other => Err(OmError::UnsupportedAction(other.to_owned())),
        }
    }

    pub fn process_thread_with_transport<T: OmTransport>(
        &self,
        store: &PhoenixCozoStore,
        thread_id: &str,
        config: &OmConfig,
        transport: &T,
    ) -> Result<OmProcessReport, OmError> {
        if !self.begin_thread(thread_id) {
            return Ok(OmProcessReport {
                already_running: true,
                mutated: false,
            });
        }

        let result = (|| {
            let mut mutated = false;
            for _ in 0..4 {
                let Some(action) = self.prepare_pending_action(store, thread_id, config)? else {
                    break;
                };

                let response = if action.kind == OBSERVE_KIND {
                    transport.observe(&action)?
                } else if action.reflector_tooling_enabled {
                    self.run_reflector_with_transport(store, &action, transport)?
                } else {
                    transport.reflect(&action)?
                };

                mutated |= self.apply_pending_action(store, &action, &response)?;
            }

            Ok(OmProcessReport {
                already_running: false,
                mutated,
            })
        })();

        self.finish_thread(thread_id);
        result
    }

    fn begin_thread(&self, thread_id: &str) -> bool {
        let mut active = self
            .active_threads
            .lock()
            .expect("om active_threads poisoned");
        if active.contains(thread_id) {
            false
        } else {
            active.insert(thread_id.to_owned());
            true
        }
    }

    fn finish_thread(&self, thread_id: &str) {
        let mut active = self
            .active_threads
            .lock()
            .expect("om active_threads poisoned");
        active.remove(thread_id);
    }

    fn run_reflector_with_transport<T: OmTransport>(
        &self,
        store: &PhoenixCozoStore,
        action: &OmPendingAction,
        transport: &T,
    ) -> Result<String, OmError> {
        let bridge = OmNativeBridge::default();
        let mut step = self.start_reflector(action)?;
        loop {
            step = match step {
                OmReflectorStep::ModelRequest { request } => {
                    let response = transport.reflect_model(&request)?;
                    self.submit_reflector_model_response(&request.session_id, response)?
                }
                OmReflectorStep::ToolCalls {
                    session_id,
                    thread_id,
                    tool_calls,
                } => {
                    let results = tool_calls
                        .iter()
                        .map(|tool_call| {
                            execute_reflector_tool(&bridge, store, &thread_id, tool_call)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    self.submit_reflector_tool_results(&session_id, &results)?
                }
                OmReflectorStep::Complete { response, .. } => return Ok(response),
            };
        }
    }

    fn apply_observation(
        &self,
        store: &PhoenixCozoStore,
        action: &OmPendingAction,
        response: &str,
    ) -> Result<bool, OmError> {
        let parsed = parse_observation_result(response);
        if parsed.observations.trim().is_empty() {
            return Ok(false);
        }

        let Some(mut record) = self.load_record(store, &action.thread_id)? else {
            return Ok(false);
        };

        let observed_ids: HashSet<&str> = action.message_ids.iter().map(String::as_str).collect();
        let mut last_observed_at = record.last_observed_at;
        let mut updated_messages = Vec::new();
        for mut message in self.load_messages_by_ids(store, &action.message_ids)? {
            if message.thread_id != action.thread_id || !observed_ids.contains(message.id.as_str())
            {
                continue;
            }
            message.is_observed = true;
            message.token_count = Some(
                message
                    .token_count
                    .unwrap_or_else(|| approx_token_count(&message.content)),
            );
            last_observed_at = last_observed_at.max(message.created_at);
            updated_messages.push(message);
        }

        if updated_messages.is_empty() {
            return Ok(false);
        }
        self.save_messages(store, &updated_messages)?;

        record.observations = parsed.observations;
        if let Some(current_task) = parsed.current_task {
            if !current_task.trim().is_empty() {
                record.current_task = current_task;
            }
        }
        record.suggested_continuation = parsed
            .suggested_continuation
            .and_then(|value| (!value.trim().is_empty()).then_some(value));
        record.last_observed_at = last_observed_at;
        record.obs_token_count = approx_token_count(&record.observations);
        record.updated_at = now_ms();
        self.save_record(store, &record)?;
        Ok(true)
    }

    fn apply_reflection(
        &self,
        store: &PhoenixCozoStore,
        action: &OmPendingAction,
        response: &str,
    ) -> Result<bool, OmError> {
        let parsed = parse_observation_result(response);
        if parsed.observations.trim().is_empty() {
            return Ok(false);
        }

        let Some(mut record) = self.load_record(store, &action.thread_id)? else {
            return Ok(false);
        };

        let input_text = std::mem::take(&mut record.observations);
        let input_tokens = record.obs_token_count;
        let output_tokens = approx_token_count(&parsed.observations);

        let generation = OmGeneration {
            id: format!("omgen-{}-{}", now_ms(), record.generation_num + 1),
            thread_id: action.thread_id.clone(),
            generation: record.generation_num + 1,
            input_tokens,
            output_tokens,
            input_text,
            output_text: parsed.observations.clone(),
            created_at: now_ms(),
        };
        self.add_generation(store, &generation)?;

        record.observations = parsed.observations;
        if let Some(current_task) = parsed.current_task {
            if !current_task.trim().is_empty() {
                record.current_task = current_task;
            }
        }
        record.suggested_continuation = parsed
            .suggested_continuation
            .and_then(|value| (!value.trim().is_empty()).then_some(value));
        record.obs_token_count = output_tokens;
        record.generation_num += 1;
        record.updated_at = now_ms();
        self.save_record(store, &record)?;
        Ok(true)
    }

    fn apply_observation_with_graph(
        &self,
        store: &PhoenixCozoStore,
        scanner: &PhoenixScanner,
        structure: &PhoenixStructure,
        bridge: &OmNativeBridge,
        config: &OmConfig,
        action: &OmPendingAction,
        response: &str,
    ) -> Result<bool, OmError> {
        let applied = self.apply_observation(store, action, response)?;
        if !applied || !(config.graph_index_enabled && config.index_observations) {
            return Ok(applied);
        }

        if let Some(record) = self.load_record(store, &action.thread_id)? {
            let source_key = format!("obs:{}", record.last_observed_at);
            let _ = bridge.index_observation_delta(
                store,
                scanner,
                structure,
                &action.thread_id,
                &source_key,
                &record.observations,
            );
        }
        Ok(applied)
    }

    fn apply_reflection_with_graph(
        &self,
        store: &PhoenixCozoStore,
        scanner: &PhoenixScanner,
        structure: &PhoenixStructure,
        bridge: &OmNativeBridge,
        config: &OmConfig,
        action: &OmPendingAction,
        response: &str,
    ) -> Result<bool, OmError> {
        let applied = self.apply_reflection(store, action, response)?;
        if !applied || !(config.graph_index_enabled && config.index_reflections) {
            return Ok(applied);
        }

        if let Some(record) = self.load_record(store, &action.thread_id)? {
            let source_key = format!("reflect:{}", record.generation_num);
            let _ = bridge.index_reflection_summary(
                store,
                scanner,
                structure,
                &action.thread_id,
                &source_key,
                &record.observations,
            );
        }
        Ok(applied)
    }

    fn eligible_messages(
        &self,
        store: &PhoenixCozoStore,
        thread_id: &str,
        record: &OmRecord,
    ) -> Result<Vec<ThreadMessage>, OmError> {
        let mut messages = self
            .load_messages(store, thread_id)?
            .into_iter()
            .filter(|message| !message.is_streaming)
            .filter(|message| !message.is_observed || message.created_at > record.last_observed_at)
            .filter(|message| !message.content.trim().is_empty())
            .collect::<Vec<_>>();
        messages.sort_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(messages)
    }

    fn get_or_create_record(
        &self,
        store: &PhoenixCozoStore,
        thread_id: &str,
    ) -> Result<OmRecord, OmError> {
        if let Some(record) = self.load_record(store, thread_id)? {
            return Ok(record);
        }

        let record = OmRecord {
            thread_id: thread_id.to_owned(),
            created_at: now_ms(),
            updated_at: now_ms(),
            ..OmRecord::default()
        };
        self.save_record(store, &record)?;
        Ok(record)
    }

    fn load_record(
        &self,
        store: &PhoenixCozoStore,
        thread_id: &str,
    ) -> Result<Option<OmRecord>, OmError> {
        Ok(store
            .fetch_compact_rows_where_str("om_records", OM_RECORD_COLUMNS, "thread_id", thread_id)?
            .into_iter()
            .next()
            .map(|row| record_from_view(&CompactRowView::new(OM_RECORD_COLUMNS, &row)))
            .transpose()?)
    }

    fn save_record(&self, store: &PhoenixCozoStore, record: &OmRecord) -> Result<(), OmError> {
        store.put_row(
            "om_records",
            json!({
                "thread_id": record.thread_id,
                "observations": record.observations,
                "current_task": record.current_task,
                "suggested_continuation": record.suggested_continuation,
                "last_observed_at": record.last_observed_at,
                "obs_token_count": record.obs_token_count,
                "generation_num": record.generation_num,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            }),
        )?;
        Ok(())
    }

    fn add_generation(
        &self,
        store: &PhoenixCozoStore,
        generation: &OmGeneration,
    ) -> Result<(), OmError> {
        store.put_row(
            "om_generations",
            json!({
                "id": generation.id,
                "thread_id": generation.thread_id,
                "generation": generation.generation,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
                "input_text": generation.input_text,
                "output_text": generation.output_text,
                "created_at": generation.created_at,
            }),
        )?;
        Ok(())
    }

    fn load_messages(
        &self,
        store: &PhoenixCozoStore,
        thread_id: &str,
    ) -> Result<Vec<ThreadMessage>, OmError> {
        let mut messages = store
            .fetch_compact_rows_where_str(
                "thread_messages",
                THREAD_MESSAGE_COLUMNS,
                "thread_id",
                thread_id,
            )?
            .into_iter()
            .map(|row| message_from_view(&CompactRowView::new(THREAD_MESSAGE_COLUMNS, &row)))
            .collect::<Result<Vec<_>, _>>()?;
        messages.sort_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(messages)
    }

    fn load_messages_by_ids(
        &self,
        store: &PhoenixCozoStore,
        message_ids: &[String],
    ) -> Result<Vec<ThreadMessage>, OmError> {
        let mut messages = store
            .fetch_compact_rows_where_in_strings(
                "thread_messages",
                THREAD_MESSAGE_COLUMNS,
                "id",
                message_ids,
            )?
            .into_iter()
            .map(|row| message_from_view(&CompactRowView::new(THREAD_MESSAGE_COLUMNS, &row)))
            .collect::<Result<Vec<_>, _>>()?;
        messages.sort_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(messages)
    }

    fn save_messages(
        &self,
        store: &PhoenixCozoStore,
        messages: &[ThreadMessage],
    ) -> Result<(), OmError> {
        if messages.is_empty() {
            return Ok(());
        }
        let rows = messages.iter().map(thread_message_json).collect::<Vec<_>>();
        store.put_rows("thread_messages", &rows)?;
        Ok(())
    }
}

fn build_reflector_model_step(session: &OmReflectorSession, allow_tools: bool) -> OmReflectorStep {
    OmReflectorStep::ModelRequest {
        request: OmReflectorModelRequest {
            session_id: session.session_id.clone(),
            thread_id: session.thread_id.clone(),
            model: session.model.clone(),
            allow_tools,
            tools: if allow_tools {
                reflector_tool_specs()
            } else {
                Vec::new()
            },
            messages: session.messages.clone(),
        },
    }
}

fn reflector_tool_specs() -> Vec<OmReflectorToolSpec> {
    vec![
        OmReflectorToolSpec {
            name: REFLECT_TOOL_RECOVER_LOST_MEMORY.to_owned(),
            description:
                "Recover facts, entities, and relations found in raw thread messages but missing from the compressed observation surface."
                    .to_owned(),
            parameters_json: json!({
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "minimum": 1, "maximum": 20 },
                    "focus": { "type": "string" }
                }
            }),
        },
        OmReflectorToolSpec {
            name: REFLECT_TOOL_MEMORY_GRAPH_SEARCH.to_owned(),
            description:
                "Search the thread-local OM graph for relevant entities, snippets, and relation summaries."
                    .to_owned(),
            parameters_json: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 20 }
                },
                "required": ["query"]
            }),
        },
    ]
}

fn looks_like_reflector_xml(content: &str) -> bool {
    let trimmed = content.trim();
    trimmed.contains("<observations>") && trimmed.contains("</observations>")
}

fn execute_reflector_tool(
    bridge: &OmNativeBridge,
    store: &PhoenixCozoStore,
    thread_id: &str,
    tool_call: &OmReflectorToolCall,
) -> Result<OmReflectorToolResult, OmError> {
    let args = serde_json::from_str::<Value>(&tool_call.arguments_json).unwrap_or(Value::Null);
    let result_json = match tool_call.name.as_str() {
        REFLECT_TOOL_RECOVER_LOST_MEMORY => {
            let limit = clamp_tool_limit(args.get("limit").and_then(Value::as_u64), 10);
            let focus = args.get("focus").and_then(Value::as_str);
            serde_json::to_string(
                &bridge
                    .recover_lost_memory(store, thread_id, limit, focus)
                    .map_err(|error| OmError::Transport(error.to_string()))?,
            )
            .map_err(|error| OmError::Transport(error.to_string()))?
        }
        REFLECT_TOOL_MEMORY_GRAPH_SEARCH => {
            let limit = clamp_tool_limit(args.get("limit").and_then(Value::as_u64), 10);
            let query = args
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or_default();
            serde_json::to_string(
                &bridge
                    .memory_graph_search(store, thread_id, query, limit)
                    .map_err(|error| OmError::Transport(error.to_string()))?,
            )
            .map_err(|error| OmError::Transport(error.to_string()))?
        }
        other => {
            serde_json::to_string(&json!({ "error": format!("Unsupported OM tool: {other}") }))
                .map_err(|error| OmError::Transport(error.to_string()))?
        }
    };
    Ok(OmReflectorToolResult {
        tool_call_id: tool_call.id.clone(),
        name: tool_call.name.clone(),
        result_json,
    })
}

fn clamp_tool_limit(value: Option<u64>, default: usize) -> usize {
    value
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(default)
        .clamp(1, 20)
}

#[cfg(not(target_arch = "wasm32"))]
fn openrouter_message_from_reflector(message: &OmReflectorMessage) -> Message {
    match message.role.as_str() {
        "system" => Message::new(Role::System, message.content.clone()),
        "assistant" if !message.tool_calls.is_empty() => Message::assistant_with_tool_calls(
            message.content.clone(),
            message
                .tool_calls
                .iter()
                .map(|tool_call| openrouter_rs::types::completion::ToolCall {
                    id: tool_call.id.clone(),
                    type_: "function".to_owned(),
                    function: openrouter_rs::types::completion::FunctionCall {
                        name: tool_call.name.clone(),
                        arguments: tool_call.arguments_json.clone(),
                    },
                    index: None,
                })
                .collect(),
        ),
        "assistant" => Message::new(Role::Assistant, message.content.clone()),
        "tool" => {
            if let Some(tool_call_id) = message.tool_call_id.as_deref() {
                Message::tool_response_named(
                    tool_call_id,
                    message.name.as_deref().unwrap_or("tool"),
                    message.content.clone(),
                )
            } else {
                Message::new(Role::Tool, message.content.clone())
            }
        }
        _ => Message::new(Role::User, message.content.clone()),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn openrouter_tool_from_spec(spec: &OmReflectorToolSpec) -> Tool {
    Tool::new(&spec.name, &spec.description, spec.parameters_json.clone())
}

#[cfg(not(target_arch = "wasm32"))]
pub struct NativeOpenRouterTransport {
    client: OpenRouterClient,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeOpenRouterTransport {
    pub fn new(
        api_key: &str,
        referer: &str,
        title: &str,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
    ) -> Result<Self, OmError> {
        let client = OpenRouterClient::builder()
            .api_key(api_key)
            .http_referer(referer)
            .x_title(title)
            .build()
            .map_err(|error| OmError::Transport(error.to_string()))?;
        Ok(Self {
            client,
            temperature,
            max_tokens,
        })
    }

    fn send_action(&self, action: &OmPendingAction) -> Result<String, OmError> {
        let mut builder = ChatCompletionRequest::builder();
        builder.model(action.model.clone());
        builder.messages(vec![
            Message::new(Role::System, action.system_prompt.clone()),
            Message::new(Role::User, action.user_prompt.clone()),
        ]);
        if let Some(temperature) = self.temperature {
            builder.temperature(temperature);
        }
        if let Some(max_tokens) = self.max_tokens {
            builder.max_tokens(max_tokens);
        }
        let request = builder
            .build()
            .map_err(|error| OmError::Transport(error.to_string()))?;
        Ok(self.send_request(&request)?.content)
    }

    fn send_request(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<OmReflectorModelResponse, OmError> {
        let response = native_runtime()
            .block_on(self.client.send_chat_completion(request))
            .map_err(|error| OmError::Transport(error.to_string()))?;
        let choice = response.choices.first();
        let content = choice
            .and_then(|choice| choice.content())
            .unwrap_or("")
            .to_owned();
        let tool_calls = choice
            .and_then(|choice| choice.tool_calls())
            .unwrap_or(&[])
            .iter()
            .map(|tool_call| OmReflectorToolCall {
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                arguments_json: tool_call.function.arguments.clone(),
            })
            .collect();
        Ok(OmReflectorModelResponse {
            content,
            tool_calls,
        })
    }

    fn build_reflector_request(
        &self,
        request: &OmReflectorModelRequest,
    ) -> Result<ChatCompletionRequest, OmError> {
        let mut builder = ChatCompletionRequest::builder();
        builder.model(request.model.clone());
        builder.messages(
            request
                .messages
                .iter()
                .map(openrouter_message_from_reflector)
                .collect(),
        );
        if request.allow_tools {
            for tool in &request.tools {
                builder.tool(openrouter_tool_from_spec(tool));
            }
            builder.tool_choice_auto();
        }
        if let Some(temperature) = self.temperature {
            builder.temperature(temperature);
        }
        if let Some(max_tokens) = self.max_tokens {
            builder.max_tokens(max_tokens);
        }
        builder
            .build()
            .map_err(|error| OmError::Transport(error.to_string()))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl OmTransport for NativeOpenRouterTransport {
    fn observe(&self, action: &OmPendingAction) -> Result<String, OmError> {
        self.send_action(action)
    }

    fn reflect(&self, action: &OmPendingAction) -> Result<String, OmError> {
        self.send_action(action)
    }

    fn reflect_model(
        &self,
        request: &OmReflectorModelRequest,
    ) -> Result<OmReflectorModelResponse, OmError> {
        let request = self.build_reflector_request(request)?;
        self.send_request(&request)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("build OM native runtime")
    })
}

pub fn approx_token_count(text: &str) -> i64 {
    if text.trim().is_empty() {
        0
    } else {
        (text.len() / 4).max(1) as i64
    }
}

fn message_window_source_key(messages: &[ThreadMessage]) -> String {
    let first = messages
        .first()
        .map(|message| message.id.as_str())
        .unwrap_or("none");
    let last = messages
        .last()
        .map(|message| message.id.as_str())
        .unwrap_or("none");
    format!("msg:{first}:{last}")
}

fn effective_observe_threshold(config: &OmConfig, current_obs_tokens: i64) -> i64 {
    let observe_threshold = f64::from(config.observe_threshold.max(1));
    let reflect_threshold = f64::from(config.reflect_threshold.max(1));
    let ratio = (current_obs_tokens.max(0) as f64 / reflect_threshold).clamp(0.0, 1.0);
    let scale = (1.0 - ratio).max(0.5);
    (observe_threshold * scale).round() as i64
}

fn build_observer_system_prompt() -> String {
    "You are the observational memory system for an AI assistant.\nRewrite the assistant's durable memory to incorporate the new messages.\nReturn XML with <observations>, optional <current_task>, and optional <suggested_continuation>.\nKeep observations concise, factual, and useful for continuing the thread."
        .to_owned()
}

fn build_observer_user_prompt(record: &OmRecord, messages: &[ThreadMessage]) -> String {
    let mut prompt = String::with_capacity(
        record.observations.len()
            + messages
                .iter()
                .map(|message| message.content.len() + message.role.len() + 8)
                .sum::<usize>()
            + 128,
    );
    if !record.observations.trim().is_empty() {
        prompt.push_str("Existing observations:\n");
        prompt.push_str(record.observations.trim());
        prompt.push_str("\n\n");
    }
    prompt.push_str("New messages:\n");
    for message in messages {
        prompt.push_str("- ");
        prompt.push_str(message.role.trim());
        prompt.push_str(": ");
        prompt.push_str(message.content.trim());
        prompt.push('\n');
    }
    prompt.push_str(
        "\nRewrite the full memory. Preserve important context, active tasks, and the best next continuation hint.",
    );
    prompt
}

fn build_reflector_system_prompt() -> String {
    "You are the reflection stage of observational memory.\nCompress the assistant's current observations without losing important context.\nReturn XML with <observations>, optional <current_task>, and optional <suggested_continuation>."
        .to_owned()
}

fn build_reflector_user_prompt(observations: &str) -> String {
    let mut prompt = String::with_capacity(observations.len() + 64);
    prompt.push_str("Current observations:\n");
    prompt.push_str(observations);
    prompt.push_str("\n\nCompress this memory so it stays complete but shorter.");
    prompt
}

fn parse_observation_result(content: &str) -> OmObservationResult {
    OmObservationResult {
        observations: extract_tag(content, "observations")
            .unwrap_or_else(|| content.trim().to_owned()),
        current_task: extract_tag(content, "current_task")
            .or_else(|| extract_tag(content, "current-task"))
            .filter(|value| !value.trim().is_empty()),
        suggested_continuation: extract_tag(content, "suggested_continuation")
            .or_else(|| extract_tag(content, "suggested-continuation"))
            .filter(|value| !value.trim().is_empty()),
    }
}

fn extract_tag(content: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = content.find(&open)? + open.len();
    let end = content[start..].find(&close)? + start;
    Some(content[start..end].trim().to_owned())
}

fn record_from_view(row: &CompactRowView<'_>) -> Result<OmRecord, StoreError> {
    Ok(OmRecord {
        thread_id: row.get_str("thread_id").unwrap_or_default().to_owned(),
        observations: row.get_str("observations").unwrap_or_default().to_owned(),
        current_task: row.get_str("current_task").unwrap_or_default().to_owned(),
        suggested_continuation: row
            .get_str("suggested_continuation")
            .map(str::to_owned)
            .filter(|value| !value.is_empty()),
        last_observed_at: row.get_i64("last_observed_at").unwrap_or_default(),
        obs_token_count: row.get_i64("obs_token_count").unwrap_or_default(),
        generation_num: row.get_i64("generation_num").unwrap_or_default(),
        created_at: row.get_i64("created_at").unwrap_or_default(),
        updated_at: row.get_i64("updated_at").unwrap_or_default(),
    })
}

fn message_from_view(row: &CompactRowView<'_>) -> Result<ThreadMessage, StoreError> {
    Ok(ThreadMessage {
        id: row.get_str("id").unwrap_or_default().to_owned(),
        thread_id: row.get_str("thread_id").unwrap_or_default().to_owned(),
        role: row.get_str("role").unwrap_or_default().to_owned(),
        content: row.get_str("content").unwrap_or_default().to_owned(),
        narrative_id: row.get_str("narrative_id").unwrap_or_default().to_owned(),
        created_at: row.get_i64("created_at").unwrap_or_default(),
        updated_at: row.get_i64("updated_at").unwrap_or_default(),
        is_streaming: row.get_bool("is_streaming").unwrap_or(false),
        token_count: row.get_i64("token_count"),
        is_observed: row.get_bool("is_observed").unwrap_or(false),
    })
}

fn thread_message_json(message: &ThreadMessage) -> Value {
    json!({
        "id": message.id,
        "thread_id": message.thread_id,
        "role": message.role,
        "content": message.content,
        "narrative_id": nullable_string(&message.narrative_id),
        "created_at": message.created_at,
        "updated_at": message.updated_at,
        "is_streaming": message.is_streaming,
        "token_count": message.token_count,
        "is_observed": message.is_observed,
    })
}

fn nullable_string(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_owned())
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    use phoenix_store_cozo::{PhoenixCozoStore, StoreConfig};
    use phoenix_types::{OmConfig, StorageMode};
    use serde_json::{json, Value};

    use super::{
        approx_token_count, OmEngine, OmPendingAction, OmReflectorModelRequest,
        OmReflectorModelResponse, OmReflectorStep, OmTransport,
    };

    struct MockTransport {
        observe_calls: Arc<Mutex<usize>>,
        reflect_calls: Arc<Mutex<usize>>,
        observe_response: String,
        reflect_response: String,
        delay_ms: u64,
    }

    impl MockTransport {
        fn new(observe_response: &str, reflect_response: &str) -> Self {
            Self {
                observe_calls: Arc::new(Mutex::new(0)),
                reflect_calls: Arc::new(Mutex::new(0)),
                observe_response: observe_response.to_owned(),
                reflect_response: reflect_response.to_owned(),
                delay_ms: 0,
            }
        }

        fn with_delay(mut self, delay_ms: u64) -> Self {
            self.delay_ms = delay_ms;
            self
        }
    }

    impl OmTransport for MockTransport {
        fn observe(&self, _action: &OmPendingAction) -> Result<String, super::OmError> {
            thread::sleep(Duration::from_millis(self.delay_ms));
            *self.observe_calls.lock().expect("observe calls poisoned") += 1;
            Ok(self.observe_response.clone())
        }

        fn reflect(&self, _action: &OmPendingAction) -> Result<String, super::OmError> {
            *self.reflect_calls.lock().expect("reflect calls poisoned") += 1;
            Ok(self.reflect_response.clone())
        }

        fn reflect_model(
            &self,
            _request: &OmReflectorModelRequest,
        ) -> Result<OmReflectorModelResponse, super::OmError> {
            *self.reflect_calls.lock().expect("reflect calls poisoned") += 1;
            Ok(OmReflectorModelResponse {
                content: self.reflect_response.clone(),
                tool_calls: Vec::new(),
            })
        }
    }

    fn store() -> PhoenixCozoStore {
        let store = PhoenixCozoStore::open(StoreConfig {
            mode: StorageMode::CozoMem,
            path: None,
        })
        .expect("open store");
        store.init_schema().expect("init schema");
        store
    }

    fn config() -> OmConfig {
        OmConfig {
            enabled: true,
            model: "mock/model".to_owned(),
            observe_threshold: 5,
            reflect_threshold: 20,
            ..OmConfig::default()
        }
    }

    fn insert_message(
        store: &PhoenixCozoStore,
        thread_id: &str,
        id: &str,
        content: &str,
        created_at: i64,
        is_streaming: bool,
    ) {
        store
            .put_row(
                "thread_messages",
                json!({
                    "id": id,
                    "thread_id": thread_id,
                    "role": "user",
                    "content": content,
                    "narrative_id": null,
                    "created_at": created_at,
                    "updated_at": created_at,
                    "is_streaming": is_streaming,
                    "token_count": approx_token_count(content),
                    "is_observed": false,
                }),
            )
            .expect("insert thread message");
    }

    #[test]
    fn prepare_skips_when_under_threshold() {
        let store = store();
        let engine = OmEngine::default();
        insert_message(&store, "thread-1", "msg-1", "tiny", 1, false);

        let action = engine
            .prepare_pending_action(
                &store,
                "thread-1",
                &OmConfig {
                    observe_threshold: 100,
                    ..config()
                },
            )
            .expect("prepare action");

        assert!(action.is_none());
    }

    #[test]
    fn observation_marks_only_covered_messages() {
        let store = store();
        let engine = OmEngine::default();
        insert_message(
            &store,
            "thread-1",
            "msg-1",
            "This is the first useful message.",
            1,
            false,
        );
        insert_message(&store, "thread-1", "msg-2", "streaming draft", 2, true);

        let transport = MockTransport::new(
            "<observations>Remember the first useful message.</observations><current_task>track state</current_task>",
            "<observations>unused</observations>",
        );

        let report = engine
            .process_thread_with_transport(&store, "thread-1", &config(), &transport)
            .expect("process thread");
        assert!(report.mutated);

        let record = store
            .fetch_rows("om_records")
            .expect("records")
            .into_iter()
            .next()
            .expect("om record");
        assert_eq!(
            record.get("current_task").and_then(Value::as_str),
            Some("track state")
        );

        let messages = store.fetch_rows("thread_messages").expect("messages");
        let observed_message = messages
            .iter()
            .find(|row| row.get("id").and_then(Value::as_str) == Some("msg-1"))
            .expect("observed message");
        let streaming_message = messages
            .iter()
            .find(|row| row.get("id").and_then(Value::as_str) == Some("msg-2"))
            .expect("streaming message");

        assert_eq!(
            observed_message.get("is_observed").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            streaming_message
                .get("is_observed")
                .and_then(Value::as_bool),
            Some(false)
        );
    }

    #[test]
    fn reflection_adds_generation_and_rewrites_record() {
        let store = store();
        let engine = OmEngine::default();
        store
            .put_row(
                "om_records",
                json!({
                    "thread_id": "thread-1",
                    "observations": "A very long memory block that should be reflected because it is too large.",
                    "current_task": "",
                    "suggested_continuation": null,
                    "last_observed_at": 0,
                    "obs_token_count": 100,
                    "generation_num": 0,
                    "created_at": 1,
                    "updated_at": 1,
                }),
            )
            .expect("seed record");

        let transport = MockTransport::new(
            "<observations>unused</observations>",
            "<observations>Compressed memory.</observations>",
        );

        let report = engine
            .process_thread_with_transport(
                &store,
                "thread-1",
                &OmConfig {
                    reflect_threshold: 10,
                    ..config()
                },
                &transport,
            )
            .expect("process reflection");
        assert!(report.mutated);

        let generations = store.fetch_rows("om_generations").expect("generations");
        assert_eq!(generations.len(), 1);

        let record = store
            .fetch_rows("om_records")
            .expect("records")
            .into_iter()
            .next()
            .expect("record");
        assert_eq!(
            record.get("observations").and_then(Value::as_str),
            Some("Compressed memory.")
        );
    }

    #[test]
    fn concurrent_processing_only_calls_transport_once() {
        let store = Arc::new(store());
        let engine = Arc::new(OmEngine::default());
        insert_message(
            &store,
            "thread-1",
            "msg-1",
            "This is enough text to trigger OM processing.",
            1,
            false,
        );

        let transport = Arc::new(
            MockTransport::new(
                "<observations>Memory updated.</observations>",
                "<observations>unused</observations>",
            )
            .with_delay(100),
        );

        let left_store = store.clone();
        let left_engine = engine.clone();
        let left_transport = transport.clone();
        let left = thread::spawn(move || {
            left_engine
                .process_thread_with_transport(&left_store, "thread-1", &config(), &*left_transport)
                .expect("left process")
        });

        thread::sleep(Duration::from_millis(10));

        let right_store = store.clone();
        let right_engine = engine.clone();
        let right_transport = transport.clone();
        let right = thread::spawn(move || {
            right_engine
                .process_thread_with_transport(
                    &right_store,
                    "thread-1",
                    &config(),
                    &*right_transport,
                )
                .expect("right process")
        });

        let left_report = left.join().expect("left join");
        let right_report = right.join().expect("right join");

        assert!(left_report.mutated || right_report.mutated);
        assert!(left_report.already_running || right_report.already_running);
        assert_eq!(
            *transport.observe_calls.lock().expect("observe calls lock"),
            1
        );
    }

    #[test]
    fn reflector_runner_completes_tool_free_xml_in_one_round() {
        let engine = OmEngine::default();
        let action = OmPendingAction {
            kind: "reflect".to_owned(),
            thread_id: "thread-1".to_owned(),
            model: "mock/model".to_owned(),
            system_prompt: "system".to_owned(),
            user_prompt: "user".to_owned(),
            message_ids: Vec::new(),
            reflector_tooling_enabled: true,
            reflector_max_tool_rounds: 2,
        };

        let step = engine.start_reflector(&action).expect("start reflector");
        let session_id = match step {
            OmReflectorStep::ModelRequest { ref request } => request.session_id.clone(),
            other => panic!("unexpected step: {other:?}"),
        };
        let complete = engine
            .submit_reflector_model_response(
                &session_id,
                OmReflectorModelResponse {
                    content: "<observations>Compressed memory.</observations>".to_owned(),
                    tool_calls: Vec::new(),
                },
            )
            .expect("submit model response");
        match complete {
            OmReflectorStep::Complete { response, .. } => {
                assert_eq!(response, "<observations>Compressed memory.</observations>");
            }
            other => panic!("unexpected complete step: {other:?}"),
        }
        assert!(!engine.drop_reflector_session(&session_id));
    }

    #[test]
    fn reflector_runner_requests_final_xml_after_tool_round_limit() {
        let engine = OmEngine::default();
        let action = OmPendingAction {
            kind: "reflect".to_owned(),
            thread_id: "thread-1".to_owned(),
            model: "mock/model".to_owned(),
            system_prompt: "system".to_owned(),
            user_prompt: "user".to_owned(),
            message_ids: Vec::new(),
            reflector_tooling_enabled: true,
            reflector_max_tool_rounds: 1,
        };

        let step = engine.start_reflector(&action).expect("start reflector");
        let session_id = match step {
            OmReflectorStep::ModelRequest { ref request } => request.session_id.clone(),
            other => panic!("unexpected step: {other:?}"),
        };
        let tool_step = engine
            .submit_reflector_model_response(
                &session_id,
                OmReflectorModelResponse {
                    content: String::new(),
                    tool_calls: vec![phoenix_types::OmReflectorToolCall {
                        id: "tool-1".to_owned(),
                        name: "recover_lost_memory".to_owned(),
                        arguments_json: "{}".to_owned(),
                    }],
                },
            )
            .expect("submit tool response");
        let tool_calls = match tool_step {
            OmReflectorStep::ToolCalls { tool_calls, .. } => tool_calls,
            other => panic!("unexpected tool step: {other:?}"),
        };
        assert_eq!(tool_calls.len(), 1);
        let next = engine
            .submit_reflector_tool_results(
                &session_id,
                &[phoenix_types::OmReflectorToolResult {
                    tool_call_id: "tool-1".to_owned(),
                    name: "recover_lost_memory".to_owned(),
                    result_json: "[]".to_owned(),
                }],
            )
            .expect("submit tool results");
        match next {
            OmReflectorStep::ModelRequest { request } => {
                assert!(!request.allow_tools);
                assert_eq!(
                    request
                        .messages
                        .last()
                        .map(|message| message.content.as_str()),
                    Some(super::FINAL_REFLECTOR_PROMPT)
                );
            }
            other => panic!("unexpected next step: {other:?}"),
        }
    }
}
