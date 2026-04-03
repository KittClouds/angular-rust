use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use phoenix_store_cozo::{CompactRow, CompactRowView, StoreError};
use phoenix_types::{
    ChatPlannerMessage, ChatPlannerModelRequest, ChatPlannerModelResponse, ChatPlannerStep,
    ChatPlannerToolCall, ChatPlannerToolSpec, ChatRun, ChatRunEvent, ChatRunStatus, ChatToolCall,
    ChatWorkspaceArtifact, EvidenceItem, QueryRequest, QueryTarget, ScopeKey, SessionId,
};
use serde_json::{json, Value};

use crate::{now_ms, PhoenixRuntime};

const PLANNER_PRODUCED_BY: &str = "phoenix-chat-rlm";
const PLANNER_MAX_TOOL_ROUNDS: usize = 4;
const PLANNER_MAX_ARTIFACTS: usize = 24;
const PLANNER_FINAL_PROMPT: &str = "Produce the final planning summary now. No more tool calls. Summarize the best evidence and how the final assistant should answer.";
const ARTIFACT_COLUMNS: &[&str] = &[
    "key",
    "thread_id",
    "narrative_id",
    "folder_id",
    "kind",
    "payload",
    "pinned",
    "produced_by",
    "created_at",
    "updated_at",
];

#[derive(Clone, Debug)]
struct ChatPlannerSession {
    run_id: String,
    thread_id: String,
    model: String,
    mutations_enabled: bool,
    messages: Vec<ChatPlannerMessage>,
    current_step: ChatPlannerStep,
    tool_rounds_used: usize,
    max_tool_rounds: usize,
    final_request_sent: bool,
    previous_signature: Option<String>,
    repeated_signature_count: usize,
}

#[derive(Default)]
pub struct ChatPlannerRunner {
    sessions: Mutex<HashMap<String, ChatPlannerSession>>,
    next_id: AtomicU64,
}

impl ChatPlannerRunner {
    pub fn get_step(
        &self,
        runtime: &PhoenixRuntime,
        run: &ChatRun,
    ) -> Result<Option<ChatPlannerStep>, StoreError> {
        if run.status != ChatRunStatus::Planning {
            return Ok(None);
        }
        if deadline_exceeded(run) {
            self.degrade_run(runtime, run, "Planner deadline reached.", None)?;
            self.drop_session(&run.id);
            return Ok(None);
        }

        let mut sessions = self.sessions.lock().expect("planner sessions poisoned");
        let session = sessions
            .entry(run.id.clone())
            .or_insert_with(|| self.build_session(run));
        runtime.chat.persist_run(
            &runtime.store,
            &ChatRun {
                planner_messages_json: serialize_messages(&session.messages)
                    .unwrap_or_else(|_| "[]".to_owned()),
                ..run.clone()
            },
        )?;
        Ok(Some(session.current_step.clone()))
    }

    pub fn peek_step(&self, run_id: &str) -> Option<ChatPlannerStep> {
        self.sessions
            .lock()
            .expect("planner sessions poisoned")
            .get(run_id)
            .map(|session| session.current_step.clone())
    }

    pub fn submit_model_response(
        &self,
        runtime: &PhoenixRuntime,
        run: &ChatRun,
        response: ChatPlannerModelResponse,
    ) -> Result<Option<ChatPlannerStep>, StoreError> {
        let mut session = self.take_session(run.id.as_str())?;
        if !matches!(session.current_step, ChatPlannerStep::ModelRequest { .. }) {
            let step = session.current_step.clone();
            self.store_session(session);
            return Ok(Some(step));
        }

        if deadline_exceeded(run) {
            self.degrade_run(
                runtime,
                run,
                "Planner deadline reached before applying the model response.",
                Some(&session.messages),
            )?;
            return Ok(None);
        }

        if !response.tool_calls.is_empty() {
            session.messages.push(ChatPlannerMessage {
                role: "assistant".to_owned(),
                content: response.content,
                name: None,
                tool_call_id: None,
                tool_calls: response.tool_calls.clone(),
            });
            session.current_step = ChatPlannerStep::ToolCalls {
                run_id: run.id.clone(),
                tool_calls: response.tool_calls,
            };
            self.persist_messages(runtime, run, &session.messages)?;
            let step = session.current_step.clone();
            self.store_session(session);
            return Ok(Some(step));
        }

        let content = response.content.trim();
        if !content.is_empty() {
            let complete = ChatPlannerStep::Complete {
                run_id: run.id.clone(),
                response: content.to_owned(),
            };
            session.messages.push(ChatPlannerMessage {
                role: "assistant".to_owned(),
                content: content.to_owned(),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            });
            self.complete_run(runtime, run, &session, content)?;
            return Ok(Some(complete));
        }

        if session.final_request_sent {
            self.degrade_run(
                runtime,
                run,
                "Planner finished without producing a usable summary.",
                Some(&session.messages),
            )?;
            return Ok(None);
        }

        session.messages.push(ChatPlannerMessage {
            role: "user".to_owned(),
            content: PLANNER_FINAL_PROMPT.to_owned(),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        });
        session.final_request_sent = true;
        session.current_step = build_model_request_step(&session, false);
        self.persist_messages(runtime, run, &session.messages)?;
        let step = session.current_step.clone();
        self.store_session(session);
        Ok(Some(step))
    }

    pub fn advance(
        &self,
        runtime: &PhoenixRuntime,
        run: &ChatRun,
    ) -> Result<Option<ChatPlannerStep>, StoreError> {
        let mut session = self.take_session(run.id.as_str())?;
        let tool_calls = match &session.current_step {
            ChatPlannerStep::ToolCalls { tool_calls, .. } => tool_calls.clone(),
            other => {
                let step = other.clone();
                self.store_session(session);
                return Ok(Some(step));
            }
        };

        if deadline_exceeded(run) {
            self.degrade_run(
                runtime,
                run,
                "Planner deadline reached during tool execution.",
                Some(&session.messages),
            )?;
            return Ok(None);
        }

        let signature = tool_signature(&tool_calls);
        let repeated = if session.previous_signature.as_deref() == Some(signature.as_str()) {
            session.repeated_signature_count = session.repeated_signature_count.saturating_add(1);
            true
        } else {
            session.previous_signature = Some(signature);
            session.repeated_signature_count = 0;
            false
        };

        let tool_outcome =
            self.execute_tool_calls(runtime, run, &tool_calls, &mut session.messages)?;
        session.tool_rounds_used = session.tool_rounds_used.saturating_add(1);

        if tool_outcome.external_pending {
            let mut updated = run.clone();
            updated.status = ChatRunStatus::AwaitingToolHost;
            updated.planner_messages_json = serialize_messages(&session.messages)
                .map_err(|error| StoreError::Query(error.to_string()))?;
            updated.updated_at = now_ms();
            runtime.chat.persist_run(&runtime.store, &updated)?;
            runtime.chat.persist_event(
                &runtime.store,
                &ChatRunEvent {
                    id: self.next_event_id("planner-event"),
                    run_id: run.id.clone(),
                    phase: "awaiting_tool_host".to_owned(),
                    kind: "status".to_owned(),
                    label: "Waiting for note workspace tools".to_owned(),
                    detail: Some(
                        "Canvas planner is waiting on the TypeScript editor host.".to_owned(),
                    ),
                    status: Some("running".to_owned()),
                    payload: None,
                    latency_ms: None,
                    created_at: updated.updated_at,
                },
            )?;
            return Ok(None);
        }

        let artifact_count = list_run_artifacts(runtime, run)?.len();
        let force_final = repeated
            || tool_outcome.had_error
            || session.tool_rounds_used >= session.max_tool_rounds
            || artifact_count >= PLANNER_MAX_ARTIFACTS;

        if force_final {
            if session.final_request_sent {
                self.degrade_run(
                    runtime,
                    run,
                    "Planner stalled after tool execution.",
                    Some(&session.messages),
                )?;
                return Ok(None);
            }
            session.messages.push(ChatPlannerMessage {
                role: "user".to_owned(),
                content: PLANNER_FINAL_PROMPT.to_owned(),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            });
            session.final_request_sent = true;
            session.current_step = build_model_request_step(&session, false);
        } else {
            session.current_step = build_model_request_step(&session, true);
        }

        self.persist_messages(runtime, run, &session.messages)?;
        let step = session.current_step.clone();
        self.store_session(session);
        Ok(Some(step))
    }

    pub fn drop_session(&self, run_id: &str) -> bool {
        self.sessions
            .lock()
            .expect("planner sessions poisoned")
            .remove(run_id)
            .is_some()
    }

    fn build_session(&self, run: &ChatRun) -> ChatPlannerSession {
        let model = run
            .options
            .planner_model
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| run.options.final_model.clone());
        let messages = restore_planner_messages(run).unwrap_or_else(|| {
            vec![
                ChatPlannerMessage {
                    role: "system".to_owned(),
                    content: build_planner_system_prompt(run),
                    name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                },
                ChatPlannerMessage {
                    role: "user".to_owned(),
                    content: build_planner_user_prompt(run),
                    name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                },
            ]
        });
        let tool_rounds_used = messages
            .iter()
            .filter(|message| message.role == "assistant" && !message.tool_calls.is_empty())
            .count();
        let final_request_sent = messages.iter().any(|message| {
            message.role == "user" && message.content.trim() == PLANNER_FINAL_PROMPT
        });
        let mut session = ChatPlannerSession {
            run_id: run.id.clone(),
            thread_id: run.thread_id.0.clone(),
            model,
            mutations_enabled: run.options.mutations_enabled,
            messages,
            current_step: ChatPlannerStep::Complete {
                run_id: run.id.clone(),
                response: String::new(),
            },
            tool_rounds_used,
            max_tool_rounds: PLANNER_MAX_TOOL_ROUNDS,
            final_request_sent,
            previous_signature: None,
            repeated_signature_count: 0,
        };
        session.current_step = build_model_request_step(&session, !session.final_request_sent);
        session
    }

    fn take_session(&self, run_id: &str) -> Result<ChatPlannerSession, StoreError> {
        self.sessions
            .lock()
            .expect("planner sessions poisoned")
            .remove(run_id)
            .ok_or_else(|| StoreError::Query(format!("unknown planner session for run {run_id}")))
    }

    fn store_session(&self, session: ChatPlannerSession) {
        self.sessions
            .lock()
            .expect("planner sessions poisoned")
            .insert(session.run_id.clone(), session);
    }

    fn persist_messages(
        &self,
        runtime: &PhoenixRuntime,
        run: &ChatRun,
        messages: &[ChatPlannerMessage],
    ) -> Result<(), StoreError> {
        let mut updated = run.clone();
        updated.planner_messages_json = serde_json::to_string(messages)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        updated.updated_at = now_ms();
        runtime.chat.persist_run(&runtime.store, &updated)
    }

    fn complete_run(
        &self,
        runtime: &PhoenixRuntime,
        run: &ChatRun,
        session: &ChatPlannerSession,
        response: &str,
    ) -> Result<(), StoreError> {
        let summary_artifact = persist_run_artifact(
            runtime,
            run,
            None,
            "draft_answer",
            json!({ "summary": response }),
            true,
        )?;
        let artifacts = list_run_artifacts(runtime, run)?;
        let planner_context = format_planner_context(response, &artifacts);
        let mut updated = run.clone();
        updated.status = ChatRunStatus::ReadyToAnswer;
        updated.error = None;
        updated.planner_messages_json = serialize_messages(&session.messages)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        if !planner_context.is_empty() {
            updated.prepared_context = append_section(&updated.prepared_context, &planner_context);
            updated.prepared_system_prompt = append_section(
                &updated.prepared_system_prompt,
                &format!("Use these planner artifacts while answering:\n\n{planner_context}"),
            );
        }
        updated.evidence_json = merge_evidence_json(
            &updated.evidence_json,
            planner_evidence_items(response, &artifacts, &summary_artifact.key),
        )?;
        updated.updated_at = now_ms();
        runtime.chat.persist_run(&runtime.store, &updated)?;
        runtime.chat.persist_event(
            &runtime.store,
            &ChatRunEvent {
                id: self.next_event_id("planner-event"),
                run_id: run.id.clone(),
                phase: "planning".to_owned(),
                kind: "status".to_owned(),
                label: "Planner complete".to_owned(),
                detail: Some(
                    "Pinned planner artifacts were promoted into the answer context.".to_owned(),
                ),
                status: Some("done".to_owned()),
                payload: None,
                latency_ms: None,
                created_at: updated.updated_at,
            },
        )?;
        runtime.chat.persist_event(
            &runtime.store,
            &ChatRunEvent {
                id: self.next_event_id("planner-event"),
                run_id: run.id.clone(),
                phase: "answer".to_owned(),
                kind: "status".to_owned(),
                label: "Ready to answer".to_owned(),
                detail: Some("Planner finished preparing the final reply context.".to_owned()),
                status: Some("done".to_owned()),
                payload: None,
                latency_ms: None,
                created_at: updated.updated_at,
            },
        )?;
        Ok(())
    }

    pub fn degrade_run(
        &self,
        runtime: &PhoenixRuntime,
        run: &ChatRun,
        reason: &str,
        messages: Option<&[ChatPlannerMessage]>,
    ) -> Result<(), StoreError> {
        let mut updated = run.clone();
        updated.status = ChatRunStatus::Degraded;
        updated.error = Some(reason.to_owned());
        if let Some(messages) = messages {
            updated.planner_messages_json = serialize_messages(messages)
                .map_err(|error| StoreError::Query(error.to_string()))?;
        }
        updated.updated_at = now_ms();
        runtime.chat.persist_run(&runtime.store, &updated)?;
        runtime.chat.persist_event(
            &runtime.store,
            &ChatRunEvent {
                id: self.next_event_id("planner-event"),
                run_id: run.id.clone(),
                phase: "planning".to_owned(),
                kind: "status".to_owned(),
                label: "Planner degraded".to_owned(),
                detail: Some(reason.to_owned()),
                status: Some("error".to_owned()),
                payload: None,
                latency_ms: None,
                created_at: updated.updated_at,
            },
        )?;
        Ok(())
    }

    fn execute_tool_calls(
        &self,
        runtime: &PhoenixRuntime,
        run: &ChatRun,
        tool_calls: &[ChatPlannerToolCall],
        messages: &mut Vec<ChatPlannerMessage>,
    ) -> Result<ToolRoundOutcome, StoreError> {
        let mut had_error = false;
        let mut external_pending = false;
        for tool_call in tool_calls {
            if let Some((host, class)) = external_tool_metadata(&tool_call.name, run) {
                external_pending = true;
                let now = now_ms();
                runtime.chat.persist_tool_call(
                    &runtime.store,
                    &ChatToolCall {
                        id: format!("tool-call:{}:{}", run.id, tool_call.id),
                        run_id: run.id.clone(),
                        tool_call_id: tool_call.id.clone(),
                        tool_name: tool_call.name.clone(),
                        host: host.to_owned(),
                        class: class.to_owned(),
                        status: "pending_host".to_owned(),
                        arguments_json: tool_call.arguments_json.clone(),
                        result_json: None,
                        error: None,
                        idempotency_key: Some(format!("{}:{}", run.id, tool_call.id)),
                        approval_id: None,
                        started_at: Some(now),
                        completed_at: None,
                        latency_ms: None,
                    },
                )?;
                runtime.chat.persist_event(
                    &runtime.store,
                    &ChatRunEvent {
                        id: self.next_event_id("planner-tool"),
                        run_id: run.id.clone(),
                        phase: "awaiting_tool_host".to_owned(),
                        kind: "tool".to_owned(),
                        label: format!("Planner tool: {}", tool_call.name),
                        detail: Some("Waiting for TypeScript host".to_owned()),
                        status: Some("running".to_owned()),
                        payload: Some(tool_call.arguments_json.clone()),
                        latency_ms: None,
                        created_at: now,
                    },
                )?;
                continue;
            }

            let outcome = execute_planner_tool_call(runtime, run, tool_call)?;
            if outcome.had_error {
                had_error = true;
            }
            runtime.chat.persist_event(
                &runtime.store,
                &ChatRunEvent {
                    id: self.next_event_id("planner-tool"),
                    run_id: run.id.clone(),
                    phase: "planning".to_owned(),
                    kind: "tool".to_owned(),
                    label: format!("Planner tool: {}", tool_call.name),
                    detail: Some(outcome.detail),
                    status: Some(if outcome.had_error {
                        "error".to_owned()
                    } else {
                        "done".to_owned()
                    }),
                    payload: None,
                    latency_ms: None,
                    created_at: now_ms(),
                },
            )?;
            messages.push(ChatPlannerMessage {
                role: "tool".to_owned(),
                content: outcome.result_json,
                name: Some(tool_call.name.clone()),
                tool_call_id: Some(tool_call.id.clone()),
                tool_calls: Vec::new(),
            });
        }
        Ok(ToolRoundOutcome {
            had_error,
            external_pending,
        })
    }

    fn next_event_id(&self, prefix: &str) -> String {
        let counter = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}-{}-{counter}", now_ms())
    }
}

struct ToolExecutionOutcome {
    result_json: String,
    detail: String,
    had_error: bool,
}

struct ToolRoundOutcome {
    had_error: bool,
    external_pending: bool,
}

fn build_planner_system_prompt(run: &ChatRun) -> String {
    if run.options.mutations_enabled {
        "You are the Phoenix Canvas planner for a chat run.\nWork inside the provided note and app scope.\nUse tools to inspect the active note, inspect the current selection, highlight candidate ranges, search scoped context, and manage workspace artifacts before the final answer is streamed.\nRead-only tools can run automatically. Proposal tools create diffs or save actions that may require approval.\nNever assume an edit proposal was already applied until a later tool result says so.\nUse proposal tools only when the user is explicitly asking for note edits or rewrites.\nWhen enough information exists, stop calling tools and provide a concise planning summary for the final assistant answer."
            .to_owned()
    } else {
        "You are the Phoenix RLM planner for a chat run.\nWork inside the provided scope.\nUse tools to retrieve notes, search lexical and graph indexes, inspect session state, and manage workspace artifacts.\nNever ask for note edits or mutations.\nStore useful intermediate findings with artifact_put and pin only the most useful artifacts with artifact_pin.\nWhen you have enough information, stop using tools and provide a concise planning summary for the final assistant answer."
            .to_owned()
    }
}

fn build_planner_user_prompt(run: &ChatRun) -> String {
    if run.options.mutations_enabled {
        format!(
            "User request:\n{}\n\nPrepared answer context:\n{}\n\nCanvas mode is enabled for the active note. You may inspect the note, highlight candidate ranges, and create targeted edit proposals when needed. Only use tools when they materially improve the answer or the requested edit.",
            run.user_prompt,
            run.prepared_system_prompt
        )
    } else {
        format!(
            "User request:\n{}\n\nPrepared answer context:\n{}\n\nOnly use tools when they materially improve the answer.",
            run.user_prompt,
            run.prepared_system_prompt
        )
    }
}

fn build_model_request_step(session: &ChatPlannerSession, allow_tools: bool) -> ChatPlannerStep {
    ChatPlannerStep::ModelRequest {
        request: ChatPlannerModelRequest {
            run_id: session.run_id.clone(),
            thread_id: session.thread_id.clone(),
            model: session.model.clone(),
            allow_tools,
            tools: if allow_tools {
                planner_tool_specs(session.mutations_enabled)
            } else {
                Vec::new()
            },
            messages: session.messages.clone(),
        },
    }
}

fn planner_tool_specs(mutations_enabled: bool) -> Vec<ChatPlannerToolSpec> {
    let mut specs = vec![
        ChatPlannerToolSpec {
            name: "scope_describe".to_owned(),
            description: "Describe the active planner scope, session, and workspace budget."
                .to_owned(),
            parameters_json: json!({ "type": "object", "properties": {} }),
        },
        ChatPlannerToolSpec {
            name: "note_list".to_owned(),
            description: "List notes inside the active scope.".to_owned(),
            parameters_json: json!({
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "minimum": 1, "maximum": 25 }
                }
            }),
        },
        ChatPlannerToolSpec {
            name: "note_get".to_owned(),
            description: "Fetch a note in scope, including its body when explicitly requested."
                .to_owned(),
            parameters_json: json!({
                "type": "object",
                "properties": {
                    "noteId": { "type": "string" },
                    "includeBody": { "type": "boolean" }
                },
                "required": ["noteId"]
            }),
        },
        ChatPlannerToolSpec {
            name: "note_read_span".to_owned(),
            description: "Read a precise character span from a note in scope.".to_owned(),
            parameters_json: json!({
                "type": "object",
                "properties": {
                    "noteId": { "type": "string" },
                    "start": { "type": "integer", "minimum": 0 },
                    "end": { "type": "integer", "minimum": 1 }
                },
                "required": ["noteId", "start", "end"]
            }),
        },
        ChatPlannerToolSpec {
            name: "search_lexical".to_owned(),
            description: "Search the lexical retrieval index inside the current scope.".to_owned(),
            parameters_json: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 10 }
                },
                "required": ["query"]
            }),
        },
        ChatPlannerToolSpec {
            name: "search_graph".to_owned(),
            description: "Search the graph retrieval surface inside the current scope.".to_owned(),
            parameters_json: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 10 }
                },
                "required": ["query"]
            }),
        },
        ChatPlannerToolSpec {
            name: "session_state".to_owned(),
            description: "Inspect the current Phoenix UI session document state.".to_owned(),
            parameters_json: json!({ "type": "object", "properties": {} }),
        },
        ChatPlannerToolSpec {
            name: "session_stats".to_owned(),
            description: "Inspect the current Phoenix UI session retrieval statistics.".to_owned(),
            parameters_json: json!({ "type": "object", "properties": {} }),
        },
        ChatPlannerToolSpec {
            name: "artifact_put".to_owned(),
            description: "Persist a run-scoped workspace artifact.".to_owned(),
            parameters_json: json!({
                "type": "object",
                "properties": {
                    "kind": { "type": "string" },
                    "payload": {},
                    "pinned": { "type": "boolean" }
                },
                "required": ["kind", "payload"]
            }),
        },
        ChatPlannerToolSpec {
            name: "artifact_list".to_owned(),
            description: "List the run-scoped workspace artifacts.".to_owned(),
            parameters_json: json!({
                "type": "object",
                "properties": {
                    "pinnedOnly": { "type": "boolean" }
                }
            }),
        },
        ChatPlannerToolSpec {
            name: "artifact_pin".to_owned(),
            description: "Pin or unpin a run-scoped workspace artifact.".to_owned(),
            parameters_json: json!({
                "type": "object",
                "properties": {
                    "key": { "type": "string" },
                    "pinned": { "type": "boolean" }
                },
                "required": ["key"]
            }),
        },
    ];

    if mutations_enabled {
        specs.extend([
            ChatPlannerToolSpec {
                name: "get_active_note_snapshot".to_owned(),
                description: "Read the active note snapshot from the live editor.".to_owned(),
                parameters_json: json!({ "type": "object", "properties": {} }),
            },
            ChatPlannerToolSpec {
                name: "get_selection".to_owned(),
                description: "Read the current live editor selection.".to_owned(),
                parameters_json: json!({ "type": "object", "properties": {} }),
            },
            ChatPlannerToolSpec {
                name: "highlight_range".to_owned(),
                description: "Highlight and reveal a candidate note range in the live editor without editing it.".to_owned(),
                parameters_json: json!({
                    "type": "object",
                    "properties": {
                        "from": { "type": "integer", "minimum": 0 },
                        "to": { "type": "integer", "minimum": 0 }
                    },
                    "required": ["from", "to"]
                }),
            },
            ChatPlannerToolSpec {
                name: "replace_text_proposal".to_owned(),
                description: "Propose replacing a specific range of text in the active note.".to_owned(),
                parameters_json: json!({
                    "type": "object",
                    "properties": {
                        "from": { "type": "integer", "minimum": 0 },
                        "to": { "type": "integer", "minimum": 0 },
                        "replacement": { "type": "string" },
                        "expectedRevision": { "type": "integer" }
                    },
                    "required": ["from", "to", "replacement"]
                }),
            },
            ChatPlannerToolSpec {
                name: "rewrite_block_proposal".to_owned(),
                description: "Propose rewriting one block in the active note.".to_owned(),
                parameters_json: json!({
                    "type": "object",
                    "properties": {
                        "blockIndex": { "type": "integer", "minimum": 0 },
                        "replacement": { "type": "string" },
                        "expectedRevision": { "type": "integer" }
                    },
                    "required": ["blockIndex", "replacement"]
                }),
            },
            ChatPlannerToolSpec {
                name: "insert_text_proposal".to_owned(),
                description: "Propose inserting text at a position in the active note.".to_owned(),
                parameters_json: json!({
                    "type": "object",
                    "properties": {
                        "pos": { "type": "integer", "minimum": 0 },
                        "text": { "type": "string" },
                        "expectedRevision": { "type": "integer" }
                    },
                    "required": ["pos", "text"]
                }),
            },
            ChatPlannerToolSpec {
                name: "save_note_proposal".to_owned(),
                description: "Propose saving the active note after edits are applied.".to_owned(),
                parameters_json: json!({ "type": "object", "properties": {} }),
            },
        ]);
    }

    specs
}

fn external_tool_metadata(name: &str, run: &ChatRun) -> Option<(&'static str, &'static str)> {
    if !run.options.mutations_enabled {
        return None;
    }

    match name {
        "get_active_note_snapshot" | "get_selection" | "highlight_range" => {
            Some(("typescript", "read"))
        }
        "replace_text_proposal"
        | "rewrite_block_proposal"
        | "insert_text_proposal"
        | "save_note_proposal" => Some(("typescript", "proposal")),
        _ => None,
    }
}

fn restore_planner_messages(run: &ChatRun) -> Option<Vec<ChatPlannerMessage>> {
    if run.planner_messages_json.trim().is_empty() {
        return None;
    }
    serde_json::from_str::<Vec<ChatPlannerMessage>>(&run.planner_messages_json)
        .ok()
        .filter(|messages| !messages.is_empty())
}

fn execute_planner_tool_call(
    runtime: &PhoenixRuntime,
    run: &ChatRun,
    tool_call: &ChatPlannerToolCall,
) -> Result<ToolExecutionOutcome, StoreError> {
    let args = serde_json::from_str::<Value>(&tool_call.arguments_json).unwrap_or(Value::Null);
    let result = match tool_call.name.as_str() {
        "scope_describe" => tool_scope_describe(runtime, run)?,
        "note_list" => tool_note_list(runtime, run, &args)?,
        "note_get" => tool_note_get(runtime, run, &args)?,
        "note_read_span" => tool_note_read_span(runtime, run, &args)?,
        "search_lexical" => tool_search_lexical(runtime, run, &args)?,
        "search_graph" => tool_search_graph(runtime, run, &args)?,
        "session_state" => tool_session_state(runtime, run)?,
        "session_stats" => tool_session_stats(runtime, run)?,
        "artifact_put" => tool_artifact_put(runtime, run, &args)?,
        "artifact_list" => tool_artifact_list(runtime, run, &args)?,
        "artifact_pin" => tool_artifact_pin(runtime, run, &args)?,
        other => json!({ "error": format!("Unsupported planner tool: {other}") }),
    };
    let had_error = result.get("error").is_some();
    Ok(ToolExecutionOutcome {
        detail: if had_error {
            result
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("Planner tool failed.")
                .to_owned()
        } else {
            format!("Returned {}", tool_call.name)
        },
        result_json: serde_json::to_string(&result)
            .map_err(|error| StoreError::Query(error.to_string()))?,
        had_error,
    })
}

fn tool_scope_describe(runtime: &PhoenixRuntime, run: &ChatRun) -> Result<Value, StoreError> {
    let session_id = resolve_main_session_id(runtime, run).map(|value| value.0);
    let artifacts = list_run_artifacts(runtime, run)?;
    Ok(json!({
        "runId": run.id,
        "threadId": run.thread_id.0,
        "narrativeId": run.options.narrative_id,
        "folderId": run.options.folder_id,
        "sessionId": session_id,
        "deadlineAt": run.deadline_at,
        "artifactCount": artifacts.len(),
        "artifactBudget": PLANNER_MAX_ARTIFACTS,
    }))
}

fn tool_note_list(
    runtime: &PhoenixRuntime,
    run: &ChatRun,
    args: &Value,
) -> Result<Value, StoreError> {
    let limit = clamp_limit(args.get("limit").and_then(Value::as_u64), 10, 25);
    let notes = runtime.list_note_values(None, false)?;
    let mut filtered = notes
        .into_iter()
        .filter(|note| note_matches_scope(note, run))
        .collect::<Vec<_>>();
    filtered.sort_by(|left, right| {
        note_updated_at(right)
            .cmp(&note_updated_at(left))
            .then_with(|| note_id(left).cmp(&note_id(right)))
    });
    filtered.truncate(limit);
    Ok(json!({
        "notes": filtered,
        "count": filtered.len(),
    }))
}

fn tool_note_get(
    runtime: &PhoenixRuntime,
    run: &ChatRun,
    args: &Value,
) -> Result<Value, StoreError> {
    let note_id = args
        .get("noteId")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if note_id.is_empty() {
        return Ok(json!({ "error": "noteId is required." }));
    }
    let include_body = args
        .get("includeBody")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let Some(note) = runtime.get_note_value(note_id, include_body)? else {
        return Ok(json!({ "error": format!("Note not found: {note_id}") }));
    };
    if !note_matches_scope(&note, run) {
        return Ok(json!({ "error": "Requested note is outside the active scope." }));
    }
    Ok(json!({ "note": note }))
}

fn tool_note_read_span(
    runtime: &PhoenixRuntime,
    run: &ChatRun,
    args: &Value,
) -> Result<Value, StoreError> {
    let note_id = args
        .get("noteId")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let start = args
        .get("start")
        .and_then(Value::as_i64)
        .unwrap_or(0)
        .max(0) as usize;
    let end = args.get("end").and_then(Value::as_i64).unwrap_or(0).max(0) as usize;
    if note_id.is_empty() || end <= start {
        return Ok(json!({ "error": "noteId, start, and end are required." }));
    }
    let Some(note) = runtime.get_note_value(note_id, true)? else {
        return Ok(json!({ "error": format!("Note not found: {note_id}") }));
    };
    if !note_matches_scope(&note, run) {
        return Ok(json!({ "error": "Requested note is outside the active scope." }));
    }
    let body = note
        .get("content")
        .or_else(|| note.get("markdown_content"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    Ok(json!({
        "noteId": note_id,
        "start": start,
        "end": end.min(body.len()),
        "text": slice_clamped(body, start, end),
    }))
}

fn tool_search_lexical(
    runtime: &PhoenixRuntime,
    run: &ChatRun,
    args: &Value,
) -> Result<Value, StoreError> {
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if query.is_empty() {
        return Ok(json!({ "error": "query is required." }));
    }
    let limit = clamp_limit(args.get("limit").and_then(Value::as_u64), 5, 10);
    let result = runtime.query(QueryRequest {
        session_id: resolve_main_session_id(runtime, run),
        query: query.to_owned(),
        scope: planner_scope(run),
        targets: vec![QueryTarget::Chunks],
        limit: Some(limit),
        temporal: None,
        semantic_query_vector: None,
        include_candidate_graph: false,
    })?;
    let document_ids = unique_document_ids(
        result
            .chunk_hits
            .iter()
            .map(|hit| chunk_id_to_document_id(&hit.chunk_id))
            .collect(),
    );
    let notes = runtime.list_note_values_by_ids(&document_ids, false)?;
    let note_map = notes
        .into_iter()
        .filter_map(|note| {
            let id = note_id(&note)?.to_owned();
            Some((id, note))
        })
        .collect::<HashMap<_, _>>();
    let chunk_map = runtime
        .store
        .list_leaf_chunks_for_documents(&document_ids)?
        .into_iter()
        .map(|chunk| (chunk.span_id, chunk.text))
        .collect::<HashMap<_, _>>();
    Ok(json!({
        "hits": result.chunk_hits.into_iter().map(|hit| {
            let note_id = chunk_id_to_document_id(&hit.chunk_id);
            json!({
                "chunkId": hit.chunk_id,
                "noteId": note_id,
                "score": hit.score,
                "title": note_map.get(&note_id).and_then(|note| note.get("title")).and_then(Value::as_str),
                "snippet": chunk_map.get(&hit.chunk_id),
            })
        }).collect::<Vec<_>>(),
        "diagnostics": result.diagnostics,
    }))
}

fn tool_search_graph(
    runtime: &PhoenixRuntime,
    run: &ChatRun,
    args: &Value,
) -> Result<Value, StoreError> {
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if query.is_empty() {
        return Ok(json!({ "error": "query is required." }));
    }
    let limit = clamp_limit(args.get("limit").and_then(Value::as_u64), 5, 10);
    let result = runtime.query(QueryRequest {
        session_id: resolve_main_session_id(runtime, run),
        query: query.to_owned(),
        scope: planner_scope(run),
        targets: vec![QueryTarget::Nodes, QueryTarget::Graph],
        limit: Some(limit),
        temporal: None,
        semantic_query_vector: None,
        include_candidate_graph: false,
    })?;
    let wanted = result
        .node_hits
        .iter()
        .filter_map(|hit| hit.entity_id.as_ref().map(|entity_id| entity_id.0.clone()))
        .collect::<HashSet<_>>();
    let entities = runtime
        .store
        .fetch_rows("entities")?
        .into_iter()
        .filter(|row| {
            row.get("id")
                .and_then(Value::as_str)
                .map(|value| wanted.contains(value))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    let entity_map = entities
        .into_iter()
        .filter_map(|row| {
            let id = row.get("id").and_then(Value::as_str)?.to_owned();
            Some((id, row))
        })
        .collect::<HashMap<_, _>>();
    Ok(json!({
        "hits": result.node_hits.into_iter().map(|hit| {
            let entity = hit
                .entity_id
                .as_ref()
                .and_then(|entity_id| entity_map.get(&entity_id.0));
            json!({
                "entityId": hit.entity_id,
                "score": hit.score,
                "label": entity.and_then(|row| row.get("label")).and_then(Value::as_str),
                "kind": entity.and_then(|row| row.get("kind")).and_then(Value::as_str),
            })
        }).collect::<Vec<_>>(),
        "diagnostics": result.diagnostics,
    }))
}

fn tool_session_state(runtime: &PhoenixRuntime, run: &ChatRun) -> Result<Value, StoreError> {
    let Some(session_id) = resolve_main_session_id(runtime, run) else {
        return Ok(json!({ "error": "No Phoenix UI main session is available." }));
    };
    let state = runtime.graptor.session_state(&runtime.store, &session_id)?;
    serde_json::to_value(state).map_err(|error| StoreError::Query(error.to_string()))
}

fn tool_session_stats(runtime: &PhoenixRuntime, run: &ChatRun) -> Result<Value, StoreError> {
    let Some(session_id) = resolve_main_session_id(runtime, run) else {
        return Ok(json!({ "error": "No Phoenix UI main session is available." }));
    };
    let stats = runtime.graptor.session_stats(&runtime.store, &session_id)?;
    serde_json::to_value(stats).map_err(|error| StoreError::Query(error.to_string()))
}

fn tool_artifact_put(
    runtime: &PhoenixRuntime,
    run: &ChatRun,
    args: &Value,
) -> Result<Value, StoreError> {
    let kind = args
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if kind.is_empty() {
        return Ok(json!({ "error": "kind is required." }));
    }
    let payload = args.get("payload").cloned().unwrap_or(Value::Null);
    let pinned = args.get("pinned").and_then(Value::as_bool).unwrap_or(false);
    if list_run_artifacts(runtime, run)?.len() >= PLANNER_MAX_ARTIFACTS {
        return Ok(json!({ "error": "Workspace artifact budget exhausted." }));
    }
    Ok(json!({
        "artifact": persist_run_artifact(runtime, run, None, kind, payload, pinned)?,
    }))
}

fn tool_artifact_list(
    runtime: &PhoenixRuntime,
    run: &ChatRun,
    args: &Value,
) -> Result<Value, StoreError> {
    let pinned_only = args
        .get("pinnedOnly")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut artifacts = list_run_artifacts(runtime, run)?;
    if pinned_only {
        artifacts.retain(|artifact| artifact.pinned);
    }
    Ok(json!({ "artifacts": artifacts }))
}

fn tool_artifact_pin(
    runtime: &PhoenixRuntime,
    run: &ChatRun,
    args: &Value,
) -> Result<Value, StoreError> {
    let key = args.get("key").and_then(Value::as_str).unwrap_or_default();
    if key.is_empty() {
        return Ok(json!({ "error": "key is required." }));
    }
    let pinned = args.get("pinned").and_then(Value::as_bool).unwrap_or(true);
    let Some(artifact) = set_artifact_pinned(runtime, run, key, pinned)? else {
        return Ok(json!({ "error": format!("Artifact not found: {key}") }));
    };
    Ok(json!({ "artifact": artifact }))
}

fn persist_run_artifact(
    runtime: &PhoenixRuntime,
    run: &ChatRun,
    explicit_key: Option<&str>,
    kind: &str,
    payload: Value,
    pinned: bool,
) -> Result<ChatWorkspaceArtifact, StoreError> {
    let now = now_ms();
    let key = explicit_key
        .map(str::to_owned)
        .unwrap_or_else(|| format!("rlm:{}:{}:{}", run.id, kind, now));
    let narrative_id = run
        .options
        .narrative_id
        .clone()
        .unwrap_or_else(|| "__global__".to_owned());
    let folder_id = run
        .options
        .folder_id
        .clone()
        .unwrap_or_else(|| "__root__".to_owned());
    runtime.store.put_row(
        "workspace_artifacts",
        json!({
            "key": key,
            "thread_id": run.id,
            "narrative_id": narrative_id,
            "folder_id": folder_id,
            "kind": kind,
            "payload": payload,
            "pinned": pinned,
            "produced_by": PLANNER_PRODUCED_BY,
            "created_at": now,
            "updated_at": now,
        }),
    )?;
    Ok(ChatWorkspaceArtifact {
        key,
        run_id: run.id.clone(),
        narrative_id,
        folder_id,
        kind: kind.to_owned(),
        payload,
        pinned,
        produced_by: PLANNER_PRODUCED_BY.to_owned(),
        created_at: now,
        updated_at: now,
    })
}

pub fn list_run_artifacts(
    runtime: &PhoenixRuntime,
    run: &ChatRun,
) -> Result<Vec<ChatWorkspaceArtifact>, StoreError> {
    let rows = runtime
        .store
        .fetch_compact_rows_with_columns("workspace_artifacts", ARTIFACT_COLUMNS)?;
    let mut artifacts = rows
        .into_iter()
        .filter_map(|row| artifact_from_row(&row))
        .filter(|artifact| artifact.run_id == run.id && artifact.produced_by == PLANNER_PRODUCED_BY)
        .collect::<Vec<_>>();
    artifacts.sort_by(|left, right| {
        left.created_at
            .cmp(&right.created_at)
            .then_with(|| left.key.cmp(&right.key))
    });
    Ok(artifacts)
}

pub(crate) fn set_artifact_pinned(
    runtime: &PhoenixRuntime,
    run: &ChatRun,
    key: &str,
    pinned: bool,
) -> Result<Option<ChatWorkspaceArtifact>, StoreError> {
    let Some(mut artifact) = list_run_artifacts(runtime, run)?
        .into_iter()
        .find(|artifact| artifact.key == key)
    else {
        return Ok(None);
    };
    artifact.pinned = pinned;
    artifact.updated_at = now_ms();
    runtime.store.put_row(
        "workspace_artifacts",
        json!({
            "key": artifact.key,
            "thread_id": artifact.run_id,
            "narrative_id": artifact.narrative_id,
            "folder_id": artifact.folder_id,
            "kind": artifact.kind,
            "payload": artifact.payload,
            "pinned": artifact.pinned,
            "produced_by": artifact.produced_by,
            "created_at": artifact.created_at,
            "updated_at": artifact.updated_at,
        }),
    )?;
    Ok(Some(artifact))
}

fn artifact_from_row(row: &CompactRow) -> Option<ChatWorkspaceArtifact> {
    let view = CompactRowView::new(ARTIFACT_COLUMNS, row);
    Some(ChatWorkspaceArtifact {
        key: view.get_str("key")?.to_owned(),
        run_id: view.get_str("thread_id")?.to_owned(),
        narrative_id: view
            .get_str("narrative_id")
            .unwrap_or("__global__")
            .to_owned(),
        folder_id: view.get_str("folder_id").unwrap_or("__root__").to_owned(),
        kind: view.get_str("kind").unwrap_or_default().to_owned(),
        payload: view.get_json("payload").unwrap_or(Value::Null),
        pinned: view.get_bool("pinned").unwrap_or(false),
        produced_by: view.get_str("produced_by").unwrap_or_default().to_owned(),
        created_at: view.get_i64("created_at").unwrap_or_default(),
        updated_at: view.get_i64("updated_at").unwrap_or_default(),
    })
}

fn planner_scope(run: &ChatRun) -> ScopeKey {
    ScopeKey {
        world_id: None,
        narrative_id: run.options.narrative_id.clone(),
        folder_id: run.options.folder_id.clone(),
        folder_path: None,
    }
}

fn resolve_main_session_id(runtime: &PhoenixRuntime, run: &ChatRun) -> Option<SessionId> {
    let rows = runtime.store.fetch_rows("phoenix_sessions").ok()?;
    let desired_narrative = run.options.narrative_id.as_deref();
    let preferred = rows.iter().find(|row| {
        row.get("label").and_then(Value::as_str) == Some("phoenix-ui-main")
            && row
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("active")
                == "active"
            && desired_narrative
                .map(|value| row.get("narrative_id").and_then(Value::as_str) == Some(value))
                .unwrap_or(true)
    });
    preferred
        .or_else(|| {
            rows.iter().find(|row| {
                row.get("label")
                    .and_then(Value::as_str)
                    .map(|label| label.starts_with("phoenix-ui"))
                    .unwrap_or(false)
            })
        })
        .and_then(|row| row.get("session_id").and_then(Value::as_str))
        .map(|value| SessionId(value.to_owned()))
}

fn note_matches_scope(note: &Value, run: &ChatRun) -> bool {
    let narrative_ok = run
        .options
        .narrative_id
        .as_deref()
        .filter(|value| !value.is_empty())
        .map(|value| note.get("narrative_id").and_then(Value::as_str) == Some(value))
        .unwrap_or(true);
    let folder_ok = run
        .options
        .folder_id
        .as_deref()
        .filter(|value| !value.is_empty())
        .map(|value| note.get("folder_id").and_then(Value::as_str) == Some(value))
        .unwrap_or(true);
    narrative_ok && folder_ok
}

fn note_updated_at(note: &Value) -> i64 {
    note.get("updated_at")
        .and_then(Value::as_i64)
        .unwrap_or_default()
}

fn note_id(note: &Value) -> Option<&str> {
    note.get("id").and_then(Value::as_str)
}

fn chunk_id_to_document_id(chunk_id: &str) -> String {
    let separator = chunk_id.find(':').unwrap_or(chunk_id.len());
    chunk_id[..separator].to_owned()
}

fn unique_document_ids(ids: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    ids.into_iter()
        .filter(|id| seen.insert(id.clone()))
        .collect()
}

fn slice_clamped(text: &str, start: usize, end: usize) -> String {
    if start >= end {
        return String::new();
    }
    let clamped_end = end.min(text.len());
    if start >= clamped_end {
        return String::new();
    }
    text[start..clamped_end].to_owned()
}

fn append_section(base: &str, addition: &str) -> String {
    let base = base.trim();
    let addition = addition.trim();
    if base.is_empty() {
        addition.to_owned()
    } else if addition.is_empty() {
        base.to_owned()
    } else {
        format!("{base}\n\n{addition}")
    }
}

fn format_planner_context(response: &str, artifacts: &[ChatWorkspaceArtifact]) -> String {
    let mut sections = Vec::new();
    let response = response.trim();
    if !response.is_empty() {
        sections.push(format!("Planner summary\n\n{response}"));
    }
    let pinned = artifacts
        .iter()
        .filter(|artifact| artifact.pinned)
        .map(format_artifact)
        .collect::<Vec<_>>();
    if !pinned.is_empty() {
        sections.push(format!(
            "Pinned workspace artifacts\n\n{}",
            pinned.join("\n\n")
        ));
    }
    sections.join("\n\n")
}

fn format_artifact(artifact: &ChatWorkspaceArtifact) -> String {
    let payload = if let Some(content) = artifact.payload.get("summary").and_then(Value::as_str) {
        content.to_owned()
    } else if let Some(content) = artifact.payload.get("content").and_then(Value::as_str) {
        content.to_owned()
    } else if artifact.payload.is_string() {
        artifact.payload.as_str().unwrap_or_default().to_owned()
    } else {
        serde_json::to_string_pretty(&artifact.payload)
            .unwrap_or_else(|_| artifact.payload.to_string())
    };
    format!("{} ({})\n{}", artifact.key, artifact.kind, payload)
}

fn planner_evidence_items(
    response: &str,
    artifacts: &[ChatWorkspaceArtifact],
    summary_key: &str,
) -> Vec<EvidenceItem> {
    let mut items = Vec::new();
    if !response.trim().is_empty() {
        items.push(EvidenceItem {
            id: format!("planner-summary:{summary_key}"),
            source: "planner_summary".to_owned(),
            title: Some("Planner summary".to_owned()),
            content: response.trim().to_owned(),
            score: None,
            metadata: None,
        });
    }
    for artifact in artifacts.iter().filter(|artifact| artifact.pinned) {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "artifactKey".to_owned(),
            Value::String(artifact.key.clone()),
        );
        metadata.insert("kind".to_owned(), Value::String(artifact.kind.clone()));
        items.push(EvidenceItem {
            id: format!("planner-artifact:{}", artifact.key),
            source: "planner_artifact".to_owned(),
            title: Some(format!("Pinned artifact: {}", artifact.kind)),
            content: format_artifact(artifact),
            score: None,
            metadata: Some(metadata),
        });
    }
    items
}

fn merge_evidence_json(existing: &str, additions: Vec<EvidenceItem>) -> Result<String, StoreError> {
    let mut evidence = if existing.trim().is_empty() {
        Vec::new()
    } else {
        serde_json::from_str::<Vec<EvidenceItem>>(existing)
            .map_err(|error| StoreError::Query(error.to_string()))?
    };
    evidence.extend(additions);
    serde_json::to_string(&evidence).map_err(|error| StoreError::Query(error.to_string()))
}

fn serialize_messages(messages: &[ChatPlannerMessage]) -> Result<String, serde_json::Error> {
    serde_json::to_string(messages)
}

fn tool_signature(tool_calls: &[ChatPlannerToolCall]) -> String {
    tool_calls
        .iter()
        .map(|call| format!("{}:{}", call.name, call.arguments_json))
        .collect::<Vec<_>>()
        .join("|")
}

fn clamp_limit(value: Option<u64>, default: usize, max: usize) -> usize {
    value
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(default)
        .clamp(1, max)
}

fn deadline_exceeded(run: &ChatRun) -> bool {
    run.deadline_at > 0 && now_ms() > run.deadline_at
}
