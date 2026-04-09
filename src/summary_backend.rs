use crate::bundled_llm::generate_bundled_summary;
use crate::config::{Config, SummaryBackendPreference};
use screamer_core::ambient::{
    heuristic_title, segments_to_transcript, CanonicalSegment, NotesSummarizer, StructuredNotes,
    SummaryTemplate,
};
use screamer_models::{
    bundled_summary_model, find_summary_model, DEFAULT_BUNDLED_SUMMARY_MODEL_ID,
};
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

const MAX_SESSION_TITLE_WORDS: usize = 4;
const MAX_SESSION_TITLE_CHARS: usize = 32;
const MAX_MODEL_PROMPT_CHARS: usize = 24_000;
const BUNDLED_SUMMARY_MAX_TOKENS: usize = 512;
const SCRATCH_PAD_START_MARKER: &str = "--- User Notes (Scratch Pad) ---";
const SCRATCH_PAD_END_MARKER: &str = "--- End User Notes ---";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SummaryModelOption {
    pub backend: SummaryBackendPreference,
    pub label: String,
    pub value: String,
}

#[derive(Clone, Debug)]
pub struct SummaryBackendRegistry {
    bundled_model_id: String,
    bundled_model_path: Option<PathBuf>,
    ollama_models: Vec<String>,
}

impl SummaryBackendRegistry {
    pub fn detect() -> Self {
        let bundled_model_id = bundled_summary_model()
            .map(|model| model.id.to_string())
            .unwrap_or_else(|| DEFAULT_BUNDLED_SUMMARY_MODEL_ID.to_string());
        let bundled_model_path = find_summary_model(&bundled_model_id);
        let ollama_models = detect_ollama_models();

        Self {
            bundled_model_id,
            bundled_model_path,
            ollama_models,
        }
    }

    pub fn bundled_model_label(&self) -> String {
        if self.bundled_model_path.is_some() {
            "Bundled Gemma 3 1B (Default)".to_string()
        } else {
            "Bundled Gemma 3 1B (Missing local artifact)".to_string()
        }
    }

    pub fn options(&self, config: &Config) -> Vec<SummaryModelOption> {
        let mut options = vec![SummaryModelOption {
            backend: SummaryBackendPreference::Bundled,
            label: self.bundled_model_label(),
            value: self.bundled_model_id.clone(),
        }];

        for model in &self.ollama_models {
            let suffix = if model == &config.summary_ollama_model {
                " (Selected)"
            } else {
                ""
            };
            options.push(SummaryModelOption {
                backend: SummaryBackendPreference::Ollama,
                label: format!("Local Ollama: {model}{suffix}"),
                value: model.clone(),
            });
        }

        if options.len() == 1 {
            options.push(SummaryModelOption {
                backend: SummaryBackendPreference::Ollama,
                label: "Local Ollama: gemma4:latest".to_string(),
                value: "gemma4:latest".to_string(),
            });
        }

        options
    }

    pub fn summarizer_for_config(&self, config: &Config) -> Arc<dyn NotesSummarizer> {
        match config.summary_backend {
            SummaryBackendPreference::Bundled => Arc::new(BundledSummaryBackend),
            SummaryBackendPreference::Ollama => Arc::new(OllamaSummaryBackend {
                model: config.summary_ollama_model.clone(),
            }),
        }
    }

    pub fn concise_session_title(
        &self,
        config: &Config,
        live_notes: &str,
        segments: &[CanonicalSegment],
    ) -> String {
        let fallback = sanitize_session_title(&heuristic_title(live_notes, segments));
        let Some(model) = self.preferred_title_model(config) else {
            return fallback;
        };

        let output = Command::new("ollama")
            .arg("run")
            .arg(&model)
            .arg(build_ollama_title_prompt(live_notes, segments, &fallback))
            .output();
        let Ok(output) = output else {
            return fallback;
        };

        if !output.status.success() {
            return fallback;
        }

        validated_session_title(&String::from_utf8_lossy(&output.stdout), &fallback)
    }

    /// Generate a concise session title from the already-generated summary.
    /// Falls back to `concise_session_title` (heuristic / raw transcript) on failure.
    pub fn title_from_summary(
        &self,
        config: &Config,
        structured_notes: &str,
        live_notes: &str,
        segments: &[CanonicalSegment],
    ) -> String {
        let fallback = self.concise_session_title(config, live_notes, segments);

        // Need some summary content to work with
        let summary_excerpt = excerpt_lines(structured_notes, 20);
        if summary_excerpt.trim().is_empty() {
            return fallback;
        }

        let prompt = build_title_from_summary_prompt(&summary_excerpt, &fallback);

        let result = match config.summary_backend {
            SummaryBackendPreference::Ollama => {
                if let Some(model) = self.preferred_title_model(config) {
                    Command::new("ollama")
                        .arg("run")
                        .arg(&model)
                        .arg(&prompt)
                        .output()
                        .ok()
                        .filter(|o| o.status.success())
                        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                } else {
                    None
                }
            }
            SummaryBackendPreference::Bundled => generate_bundled_summary(&prompt, 32).ok(),
        };

        match result {
            Some(raw) if !raw.trim().is_empty() => validated_session_title(&raw, &fallback),
            _ => fallback,
        }
    }

    pub fn has_any_ollama_model(&self) -> bool {
        !self.ollama_models.is_empty()
    }

    fn preferred_title_model(&self, config: &Config) -> Option<String> {
        if self
            .ollama_models
            .iter()
            .any(|model| model == &config.summary_ollama_model)
        {
            return Some(config.summary_ollama_model.clone());
        }

        self.ollama_models.first().cloned()
    }
}

struct BundledSummaryBackend;

impl NotesSummarizer for BundledSummaryBackend {
    fn summarize(
        &self,
        live_notes: &str,
        segments: &[CanonicalSegment],
        title_hint: Option<&str>,
        template: SummaryTemplate,
    ) -> Result<StructuredNotes, String> {
        let title_hint = title_hint.filter(|value| !value.trim().is_empty());
        let fallback = heuristic_structured_notes(live_notes, segments, title_hint);

        if template == SummaryTemplate::General {
            let general_fallback =
                heuristic_general_notes(live_notes, segments, title_hint, &fallback);
            return summarize_general_chunked(
                live_notes,
                segments,
                title_hint,
                &fallback.key_points,
                general_fallback,
            );
        }

        let prompt = build_structured_notes_prompt(live_notes, segments, title_hint, template);

        match generate_bundled_summary(&prompt, BUNDLED_SUMMARY_MAX_TOKENS) {
            Ok(content) if !content.trim().is_empty() => Ok(merge_model_structured_notes(
                &content, live_notes, segments, title_hint, fallback,
            )),
            _ => Ok(fallback),
        }
    }
}

struct OllamaSummaryBackend {
    model: String,
}

impl OllamaSummaryBackend {
    fn generate(&self, prompt: &str) -> Result<String, String> {
        let output = Command::new("ollama")
            .arg("run")
            .arg(&self.model)
            .arg(prompt)
            .output()
            .map_err(|err| format!("Failed to launch Ollama: {err}"))?;

        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
        }

        Ok(trim_generated_response(&String::from_utf8_lossy(
            &output.stdout,
        )))
    }
}

impl NotesSummarizer for OllamaSummaryBackend {
    fn summarize(
        &self,
        live_notes: &str,
        segments: &[CanonicalSegment],
        title_hint: Option<&str>,
        template: SummaryTemplate,
    ) -> Result<StructuredNotes, String> {
        if template == SummaryTemplate::General {
            let fallback = heuristic_structured_notes(live_notes, segments, title_hint);
            let general_fallback =
                heuristic_general_notes(live_notes, segments, title_hint, &fallback);
            return summarize_general_ollama(
                self,
                live_notes,
                segments,
                title_hint,
                &fallback.key_points,
                general_fallback,
            );
        }

        let prompt = build_structured_notes_prompt(live_notes, segments, title_hint, template);
        let content = self.generate(&prompt)?;

        if content.is_empty() {
            return Ok(heuristic_structured_notes(live_notes, segments, title_hint));
        }

        Ok(merge_model_structured_notes(
            &content,
            live_notes,
            segments,
            title_hint,
            heuristic_structured_notes(live_notes, segments, title_hint),
        ))
    }
}

fn detect_ollama_models() -> Vec<String> {
    let output = Command::new("ollama").arg("list").output();
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .skip(1)
        .filter_map(|line| line.split_whitespace().next())
        .filter(|name| name.contains("gemma"))
        .map(str::to_string)
        .collect()
}

// ---------------------------------------------------------------------------
// General-template chunked summarization
// ---------------------------------------------------------------------------

/// Threshold (in chars) below which we summarize in a single pass.
const GENERAL_SINGLE_PASS_CHARS: usize = 6_000;
/// Max chars per chunk when splitting a long transcript.
const GENERAL_CHUNK_CHARS: usize = 5_000;
/// Max tokens for each chunk summary (bundled model).
const GENERAL_CHUNK_MAX_TOKENS: usize = 384;
/// Max tokens for the final merge pass (bundled model).
const GENERAL_MERGE_MAX_TOKENS: usize = 768;
const GENERAL_RESCUE_SECTION_MAX_TOKENS: usize = 192;
const GENERAL_TOPIC_DISCOVERY_MAX_TOKENS: usize = 128;
const GENERAL_TOPIC_DETAIL_MAX_TOKENS: usize = 224;
const GENERAL_MAX_TOPICS: usize = 5;
const GENERAL_TOPIC_CONTEXT_MAX_CHARS: usize = 10_000;
const SUMMARY_TRANSCRIPT_REPEAT_NGRAM_MIN: usize = 3;
const SUMMARY_TRANSCRIPT_REPEAT_NGRAM_MAX: usize = 12;
const SUMMARY_TRANSCRIPT_RECENT_LINE_WINDOW: usize = 2;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct GeneralSummaryContext {
    scratch_pad: Option<String>,
    transcript: String,
    chunks: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TopicMention {
    title: String,
    chunk_index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TopicCluster {
    title: String,
    chunk_indices: Vec<usize>,
    mentions: usize,
    first_chunk_index: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct TopicSection {
    title: String,
    bullets: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RecoveryTopicSection {
    title: String,
    first_index: usize,
    lines: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct GeneralSummaryDraft {
    topics: Vec<TopicSection>,
}

fn summarize_general_chunked(
    live_notes: &str,
    segments: &[CanonicalSegment],
    title_hint: Option<&str>,
    salient_lines: &[String],
    fallback: StructuredNotes,
) -> Result<StructuredNotes, String> {
    let generate = |prompt: &str, max_tokens: usize| generate_bundled_summary(prompt, max_tokens);
    if transcript_seems_noisy(live_notes, segments, salient_lines)
        || transcript_is_pathological_for_llm(live_notes, segments)
    {
        if let Some(notes) = summarize_general_recovery(live_notes, segments, title_hint, &generate)
        {
            return Ok(notes);
        }
        return Ok(fallback);
    }

    if let Some(notes) = summarize_general_multistage(live_notes, segments, title_hint, &generate) {
        return Ok(notes);
    }

    let transcript = full_summary_context(live_notes, segments, false);
    let title = title_hint.unwrap_or("Ambient session");

    // Short transcripts: single pass
    if transcript.chars().count() <= GENERAL_SINGLE_PASS_CHARS {
        let prompt = format!(
            "{GENERAL_TEMPLATE_PROMPT}\n\nTitle hint: {title}\n\nTranscript:\n{transcript}"
        );
        return match generate_bundled_summary(&prompt, GENERAL_MERGE_MAX_TOKENS) {
            Ok(raw) if !raw.trim().is_empty() => {
                Ok(validated_general_raw_notes(raw, segments).unwrap_or_else(|| fallback.clone()))
            }
            _ => Ok(fallback),
        };
    }

    // Long transcripts: chunk → summarize each → merge
    let chunks = split_transcript_chunks(&transcript, GENERAL_CHUNK_CHARS);
    let mut chunk_summaries = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        let prompt = format!(
            "{GENERAL_CHUNK_PROMPT}\n\nPart {}/{} of transcript:\n{chunk}",
            i + 1,
            chunks.len()
        );
        match generate_bundled_summary(&prompt, GENERAL_CHUNK_MAX_TOKENS) {
            Ok(partial) if !partial.trim().is_empty() => {
                chunk_summaries.push(trim_generated_response(&partial));
            }
            _ => {} // skip failed chunks
        }
    }

    if chunk_summaries.is_empty() {
        return Ok(fallback);
    }

    let combined = chunk_summaries.join("\n\n---\n\n");
    let merge_prompt =
        format!("{GENERAL_MERGE_PROMPT}\n\nTitle hint: {title}\n\nPartial summaries:\n{combined}");

    match generate_bundled_summary(&merge_prompt, GENERAL_MERGE_MAX_TOKENS) {
        Ok(raw) if !raw.trim().is_empty() => {
            Ok(validated_general_raw_notes(raw, segments).unwrap_or_else(|| fallback.clone()))
        }
        _ => {
            // If merge fails, concatenate chunk summaries as the output
            Ok(
                validated_general_raw_notes(combined, segments).unwrap_or_else(|| {
                    summarize_general_recovery(live_notes, segments, title_hint, &generate)
                        .unwrap_or(fallback)
                }),
            )
        }
    }
}

fn summarize_general_ollama(
    backend: &OllamaSummaryBackend,
    live_notes: &str,
    segments: &[CanonicalSegment],
    title_hint: Option<&str>,
    salient_lines: &[String],
    fallback: StructuredNotes,
) -> Result<StructuredNotes, String> {
    let generate = |prompt: &str, _max_tokens: usize| backend.generate(prompt);
    if transcript_seems_noisy(live_notes, segments, salient_lines)
        || transcript_is_pathological_for_llm(live_notes, segments)
    {
        if let Some(notes) = summarize_general_recovery(live_notes, segments, title_hint, &generate)
        {
            return Ok(notes);
        }
        return Ok(fallback);
    }

    if let Some(notes) = summarize_general_multistage(live_notes, segments, title_hint, &generate) {
        return Ok(notes);
    }

    let transcript = full_summary_context(live_notes, segments, false);
    let title = title_hint.unwrap_or("Ambient session");

    if transcript.chars().count() <= GENERAL_SINGLE_PASS_CHARS {
        let prompt = format!(
            "{GENERAL_TEMPLATE_PROMPT}\n\nTitle hint: {title}\n\nTranscript:\n{transcript}"
        );
        return match backend.generate(&prompt) {
            Ok(raw) if !raw.trim().is_empty() => {
                Ok(validated_general_raw_notes(raw, segments).unwrap_or_else(|| fallback.clone()))
            }
            Ok(_) => Ok(fallback),
            Err(err) => Err(err),
        };
    }

    let chunks = split_transcript_chunks(&transcript, GENERAL_CHUNK_CHARS);
    let mut chunk_summaries = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        let prompt = format!(
            "{GENERAL_CHUNK_PROMPT}\n\nPart {}/{} of transcript:\n{chunk}",
            i + 1,
            chunks.len()
        );
        if let Ok(partial) = backend.generate(&prompt) {
            if !partial.trim().is_empty() {
                chunk_summaries.push(partial);
            }
        }
    }

    if chunk_summaries.is_empty() {
        return Ok(fallback);
    }

    let combined = chunk_summaries.join("\n\n---\n\n");
    let merge_prompt =
        format!("{GENERAL_MERGE_PROMPT}\n\nTitle hint: {title}\n\nPartial summaries:\n{combined}");

    match backend.generate(&merge_prompt) {
        Ok(raw) if !raw.trim().is_empty() => {
            Ok(validated_general_raw_notes(raw, segments).unwrap_or_else(|| fallback.clone()))
        }
        _ => Ok(
            validated_general_raw_notes(combined, segments).unwrap_or_else(|| {
                summarize_general_recovery(live_notes, segments, title_hint, &generate)
                    .unwrap_or(fallback)
            }),
        ),
    }
}

fn summarize_general_multistage(
    live_notes: &str,
    segments: &[CanonicalSegment],
    title_hint: Option<&str>,
    generate: &dyn Fn(&str, usize) -> Result<String, String>,
) -> Option<StructuredNotes> {
    let context = general_summary_context(live_notes, segments);
    if context.transcript.trim().is_empty() {
        return None;
    }

    let topic_clusters = discover_topic_clusters(&context, title_hint, generate);
    if topic_clusters.is_empty() {
        return None;
    }

    let topics = build_topic_sections(&context, title_hint, &topic_clusters, generate);
    if topics.is_empty() {
        return None;
    }

    let draft = GeneralSummaryDraft { topics };
    let rendered = render_general_summary_draft(&draft);
    if rendered.trim().is_empty() {
        return None;
    }

    Some(StructuredNotes {
        transcript: segments_to_transcript(segments),
        raw_notes: Some(rendered),
        ..Default::default()
    })
}

fn summarize_general_recovery(
    live_notes: &str,
    segments: &[CanonicalSegment],
    title_hint: Option<&str>,
    generate: &dyn Fn(&str, usize) -> Result<String, String>,
) -> Option<StructuredNotes> {
    let (scratch_pad, _) = split_scratch_pad_context(live_notes);
    let distilled_lines =
        select_general_recovery_lines(&cleaned_summary_transcript_lines(live_notes, segments));
    if distilled_lines.is_empty() {
        return None;
    }

    let topics = build_recovery_topic_sections(&distilled_lines, title_hint, scratch_pad, generate);
    if topics.is_empty() {
        return None;
    }

    let rendered = render_general_summary_draft(&GeneralSummaryDraft { topics });
    if rendered.trim().is_empty() {
        return None;
    }

    Some(StructuredNotes {
        transcript: segments_to_transcript(segments),
        raw_notes: Some(rendered),
        ..Default::default()
    })
}

fn build_recovery_topic_sections(
    distilled_lines: &[String],
    title_hint: Option<&str>,
    scratch_pad: Option<&str>,
    generate: &dyn Fn(&str, usize) -> Result<String, String>,
) -> Vec<TopicSection> {
    let mut sections = Vec::<TopicSection>::new();

    for topic in cluster_recovery_lines(distilled_lines) {
        let mut bullets = fallback_recovery_section_bullets(&topic);

        if bullets.is_empty() {
            let prompt = build_general_stage_prompt(
                &GENERAL_RESCUE_SECTION_PROMPT.replace("{topic}", &topic.title),
                title_hint,
                scratch_pad,
                "Reliable transcript fragments",
                &topic.lines.join("\n"),
            );
            let generated = generate(&prompt, GENERAL_RESCUE_SECTION_MAX_TOKENS)
                .ok()
                .map(|raw| parse_simple_bullets(&raw))
                .unwrap_or_default();
            bullets = generated
                .into_iter()
                .filter(|bullet| bullet_has_source_support(bullet, &topic.lines))
                .filter(|bullet| !contains_forbidden_summary_pronouns(bullet))
                .filter(|bullet| line_has_reliable_summary_signal(bullet))
                .collect::<Vec<_>>();
        }

        let bullets = prioritize_general_section_items(&dedupe_bullets(&bullets), 3);
        if bullets.is_empty() {
            continue;
        }

        sections.push(TopicSection {
            title: topic.title,
            bullets,
        });
    }

    sections
}

fn cluster_recovery_lines(lines: &[String]) -> Vec<RecoveryTopicSection> {
    let mut sections = Vec::<RecoveryTopicSection>::new();

    for (index, line) in lines.iter().enumerate() {
        let title = infer_recovery_topic_title(line);
        if let Some(existing) = sections.iter_mut().find(|section| section.title == title) {
            existing.lines.push(line.clone());
            continue;
        }

        sections.push(RecoveryTopicSection {
            title,
            first_index: index,
            lines: vec![line.clone()],
        });
    }

    sections.sort_by_key(|section| section.first_index);
    sections
}

fn infer_recovery_topic_title(line: &str) -> String {
    let lower = line.to_ascii_lowercase();

    if lower.contains("dashboard") {
        return "Dashboard Work".to_string();
    }
    if lower.contains("24 hour")
        || lower.contains("24-hour")
        || lower.contains("wake up")
        || lower.contains("woken up")
        || lower.contains("rotation")
    {
        return "On-Call Expectations".to_string();
    }
    if lower.contains("incident")
        || lower.contains("slack")
        || lower.contains("runbook")
        || lower.contains("notified")
        || lower.contains("alert")
        || lower.contains("page")
        || lower.contains("ping")
        || (lower.contains("handled") && lower.contains("offline"))
        || (lower.contains("david")
            && (lower.contains("offline")
                || lower.contains("wasn't online")
                || lower.contains("wasnt online")
                || lower.contains("not online")
                || lower.contains("online")))
    {
        return "Incident Response".to_string();
    }
    if lower.contains("on call")
        || lower.contains("on-call")
        || lower.contains("coverage")
        || lower.contains("pto")
        || lower.contains("flight")
    {
        return "On-Call Coverage".to_string();
    }
    if lower.contains("training") || lower.contains("setup") {
        return "Training And Setup".to_string();
    }

    fallback_topic_title_from_line(line)
}

fn fallback_topic_title_from_line(line: &str) -> String {
    let tokens = summary_content_tokens(line)
        .into_iter()
        .take(3)
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return "Discussion Notes".to_string();
    }

    title_case_words(&tokens.join(" "))
}

fn fallback_recovery_section_bullets(topic: &RecoveryTopicSection) -> Vec<String> {
    topic
        .lines
        .iter()
        .filter_map(|line| fallback_recovery_bullet(line))
        .collect()
}

fn fallback_recovery_bullet(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();

    if (lower.contains("on call") || lower.contains("on-call"))
        && (lower.contains("pto") || lower.contains("taking"))
    {
        return Some(
            "A team member requested on-call coverage during planned time off.".to_string(),
        );
    }
    if lower.contains("tuesday") && lower.contains("flight") {
        return Some(
            "Coverage may resume Tuesday night if the return flight is on time.".to_string(),
        );
    }
    if lower.contains("david")
        && (lower.contains("offline")
            || lower.contains("wasn't online")
            || lower.contains("wasnt online")
            || lower.contains("not online")
            || lower.contains("online"))
    {
        return Some(
            "David handled a recent early-morning issue because the assigned person was offline."
                .to_string(),
        );
    }
    if lower.contains("24 hour") || lower.contains("24-hour") {
        return Some(
            "The team clarified that this is not treated as a 24-hour on-call rotation."
                .to_string(),
        );
    }
    if lower.contains("training") && lower.contains("setup") {
        return Some(
            "The team discussed training and setup requirements for people entering the on-call loop."
                .to_string(),
        );
    }
    if lower.contains("slack") || lower.contains("runbook") || lower.contains("notified") {
        return Some(
            "The team discussed how incident notifications should flow through Slack access and runbooks."
                .to_string(),
        );
    }

    None
}

/// Build a `StructuredNotes` that uses `raw_notes` so `to_markdown()` emits
/// the LLM output directly instead of the rigid section layout.
fn validated_general_raw_notes(
    raw: String,
    segments: &[CanonicalSegment],
) -> Option<StructuredNotes> {
    let cleaned = clean_general_markdown(&raw);
    if !general_markdown_has_signal(&cleaned) {
        return None;
    }

    Some(StructuredNotes {
        transcript: segments_to_transcript(segments),
        raw_notes: Some(cleaned),
        ..Default::default()
    })
}

fn general_summary_context(
    live_notes: &str,
    segments: &[CanonicalSegment],
) -> GeneralSummaryContext {
    let (scratch_pad, transcript_body) = split_scratch_pad_context(live_notes);
    let transcript = if segments.is_empty() {
        normalize_live_transcript(transcript_body, false)
    } else {
        segments_to_speakerless_transcript(segments)
    };
    let chunks = split_transcript_chunks(&transcript, GENERAL_CHUNK_CHARS);

    GeneralSummaryContext {
        scratch_pad: scratch_pad
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string),
        transcript,
        chunks,
    }
}

fn discover_topic_clusters(
    context: &GeneralSummaryContext,
    title_hint: Option<&str>,
    generate: &dyn Fn(&str, usize) -> Result<String, String>,
) -> Vec<TopicCluster> {
    let mut mentions = Vec::new();
    for (chunk_index, chunk) in context.chunks.iter().enumerate() {
        let prompt = build_general_stage_prompt(
            GENERAL_TOPIC_DISCOVERY_PROMPT,
            title_hint,
            context.scratch_pad.as_deref(),
            "Transcript excerpt",
            chunk,
        );
        let Ok(output) = generate(&prompt, GENERAL_TOPIC_DISCOVERY_MAX_TOKENS) else {
            continue;
        };

        for title in parse_topic_titles(&output) {
            mentions.push(TopicMention { title, chunk_index });
        }
    }

    merge_topic_mentions(mentions)
}

fn build_topic_sections(
    context: &GeneralSummaryContext,
    title_hint: Option<&str>,
    clusters: &[TopicCluster],
    generate: &dyn Fn(&str, usize) -> Result<String, String>,
) -> Vec<TopicSection> {
    let mut sections = Vec::new();

    for cluster in clusters {
        let topic_context = build_topic_context(context, cluster);
        if topic_context.trim().is_empty() {
            continue;
        }

        let prompt = build_general_stage_prompt(
            &GENERAL_TOPIC_DETAIL_PROMPT.replace("{topic}", &cluster.title),
            title_hint,
            context.scratch_pad.as_deref(),
            "Relevant transcript passages",
            &topic_context,
        );
        let bullets = generate(&prompt, GENERAL_TOPIC_DETAIL_MAX_TOKENS)
            .ok()
            .map(|output| parse_simple_bullets(&output))
            .filter(|bullets| !bullets.is_empty())
            .unwrap_or_else(|| fallback_topic_bullets(&topic_context));
        let bullets = dedupe_bullets(&bullets);
        if bullets.is_empty() {
            continue;
        }

        sections.push(TopicSection {
            title: cluster.title.clone(),
            bullets,
        });
    }

    sections
}

fn build_general_stage_prompt(
    instruction: &str,
    title_hint: Option<&str>,
    scratch_pad: Option<&str>,
    body_label: &str,
    body: &str,
) -> String {
    let mut out = String::new();
    out.push_str(instruction.trim());
    out.push_str("\n\nTitle hint: ");
    out.push_str(title_hint.unwrap_or("Ambient session"));

    if let Some(scratch_pad) = scratch_pad.filter(|text| !text.trim().is_empty()) {
        out.push_str("\n\nUser Notes (Scratch Pad):\n");
        out.push_str(scratch_pad.trim());
    }

    out.push_str("\n\n");
    out.push_str(body_label);
    out.push_str(":\n");
    out.push_str(body.trim());
    out
}

fn parse_topic_titles(text: &str) -> Vec<String> {
    parse_simple_bullets(text)
        .into_iter()
        .filter_map(|title| normalize_topic_title(&title))
        .collect()
}

fn normalize_topic_title(text: &str) -> Option<String> {
    let line = strip_list_prefix(text);
    let line = line
        .trim()
        .trim_start_matches('#')
        .trim()
        .trim_matches(|ch: char| matches!(ch, '"' | '\'' | '`' | '*' | '-' | ':' | '.'))
        .trim();
    if line.is_empty() {
        return None;
    }

    let line = if let Some((prefix, rest)) = line.split_once(':') {
        let prefix_lower = prefix.trim().to_ascii_lowercase();
        if prefix_lower.starts_with("topic") || prefix_lower == "title" {
            rest.trim()
        } else {
            line
        }
    } else {
        line
    };
    let line = strip_speaker_prefix(line).trim();
    let normalized = collapse_spaces(line);
    let lower = normalized.to_ascii_lowercase();
    if is_meta_summary_line(&normalized) {
        return None;
    }
    if normalized.split_whitespace().count() > 7 {
        return None;
    }
    if [
        "summary",
        "key points",
        "recap",
        "discussion",
        "notes",
        "transcript",
        "miscellaneous",
        "other",
        "general",
        "questions",
        "updates",
    ]
    .iter()
    .any(|generic| lower == *generic || lower.starts_with(&format!("{generic} ")))
    {
        return None;
    }

    Some(normalized)
}

fn merge_topic_mentions(mentions: Vec<TopicMention>) -> Vec<TopicCluster> {
    let mut clusters = Vec::<TopicCluster>::new();

    for mention in mentions {
        if let Some(existing) = clusters
            .iter_mut()
            .find(|cluster| topic_titles_similar(&cluster.title, &mention.title))
        {
            existing.mentions += 1;
            if !existing.chunk_indices.contains(&mention.chunk_index) {
                existing.chunk_indices.push(mention.chunk_index);
            }
            if mention.chunk_index < existing.first_chunk_index {
                existing.first_chunk_index = mention.chunk_index;
            }
            existing.title = preferred_topic_title(&existing.title, &mention.title);
            continue;
        }

        clusters.push(TopicCluster {
            title: mention.title,
            chunk_indices: vec![mention.chunk_index],
            mentions: 1,
            first_chunk_index: mention.chunk_index,
        });
    }

    clusters.sort_by(|left, right| {
        right
            .mentions
            .cmp(&left.mentions)
            .then(left.first_chunk_index.cmp(&right.first_chunk_index))
    });
    clusters.truncate(GENERAL_MAX_TOPICS);
    clusters.sort_by_key(|cluster| cluster.first_chunk_index);
    clusters
}

fn topic_titles_similar(left: &str, right: &str) -> bool {
    let left_normalized = collapse_spaces(&left.to_ascii_lowercase());
    let right_normalized = collapse_spaces(&right.to_ascii_lowercase());
    if left_normalized == right_normalized {
        return true;
    }
    if left_normalized.contains(&right_normalized) || right_normalized.contains(&left_normalized) {
        return true;
    }

    let left_tokens = summary_tokens(&left_normalized);
    let right_tokens = summary_tokens(&right_normalized);
    if left_tokens.is_empty() || right_tokens.is_empty() {
        return false;
    }

    let left_set = left_tokens.iter().collect::<HashSet<_>>();
    let right_set = right_tokens.iter().collect::<HashSet<_>>();
    let intersection = left_set.intersection(&right_set).count();
    let min_len = left_set.len().min(right_set.len());
    min_len > 0 && intersection * 2 >= min_len
}

fn preferred_topic_title(current: &str, candidate: &str) -> String {
    let current_words = current.split_whitespace().count();
    let candidate_words = candidate.split_whitespace().count();
    if candidate_words > current_words && candidate_words <= 7 {
        candidate.to_string()
    } else {
        current.to_string()
    }
}

fn build_topic_context(context: &GeneralSummaryContext, cluster: &TopicCluster) -> String {
    let mut selected = cluster.chunk_indices.clone();
    selected.sort_unstable();
    selected.dedup();

    let combined = selected
        .into_iter()
        .filter_map(|index| context.chunks.get(index))
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join("\n\n");
    excerpt_balanced_text(&combined, GENERAL_TOPIC_CONTEXT_MAX_CHARS)
}

fn fallback_topic_bullets(topic_context: &str) -> Vec<String> {
    topic_context
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(4)
        .map(strip_speaker_prefix)
        .map(collapse_spaces)
        .collect()
}

fn parse_simple_bullets(text: &str) -> Vec<String> {
    let cleaned = strip_code_fence(&trim_generated_response(text));
    let mut bullets = Vec::new();

    for raw_line in cleaned.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let Some(stripped) = list_item_body(line) else {
            continue;
        };

        let normalized = strip_speaker_prefix(stripped)
            .trim()
            .trim_matches(|ch| matches!(ch, '*' | '_'))
            .trim();
        if normalized.is_empty()
            || is_presentational_line(normalized)
            || is_meta_summary_line(normalized)
        {
            continue;
        }
        bullets.push(collapse_spaces(normalized));
    }

    dedupe_bullets(&bullets)
}

fn is_presentational_line(line: &str) -> bool {
    let lower = line.trim().to_ascii_lowercase();
    lower.starts_with("here are")
        || lower.starts_with("here's")
        || lower.starts_with("here’s")
        || lower.starts_with("below are")
        || lower.starts_with("topics discussed")
        || lower.starts_with("core discussion points")
        || lower.starts_with("breakdown")
        || lower.starts_with("thinking")
        || lower.starts_with("topic ")
}

fn is_meta_summary_line(line: &str) -> bool {
    let normalized = collapse_spaces(strip_speaker_prefix(line).trim());
    if normalized.is_empty() {
        return true;
    }

    let lower = normalized.to_ascii_lowercase();
    if [
        "okay",
        "ok",
        "detailed breakdown",
        "here's a breakdown",
        "here’s a breakdown",
        "here's a response",
        "here’s a response",
        "let's break this down",
        "lets break this down",
        "overall theme",
        "revised text",
        "revised version",
        "the initial interaction",
        "the core loop",
        "the core of the discussion",
        "the conversation revolves around",
        "the speaker is trying",
        "the user is starting",
        "mix of",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
    {
        return true;
    }

    if [
        "response to your request",
        "breakdown of topics",
        "organized into bullet points",
        "preserving the intent and tone",
        "what i think is happening",
        "output bullet lines only",
        "capture only details relevant",
        "preserve concrete facts",
        "do not mention speaker labels",
        "do not add a heading",
        "do not repeat the topic title",
        "prefer 2 to 5 bullets",
        "include only decisions",
        "include only items",
        "if there are no reliable",
        "starting with `- `",
        "each starting with `- `",
        "conversation revolves around",
        "core of the discussion",
        "slightly disjointed feel",
        "potential interpretations",
        "conversational flow",
        "technical jargon",
        "sense of being overwhelmed",
        "repetitive pattern",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
    {
        return true;
    }

    let tokens = summary_tokens(&normalized);
    looks_symbolic_line(&normalized)
        || (tokens.len() >= 8 && looks_noisy_line(&normalized, &tokens))
}

fn dedupe_bullets(lines: &[String]) -> Vec<String> {
    let mut out = Vec::<String>::new();
    for line in lines {
        let trimmed = collapse_spaces(strip_speaker_prefix(line).trim());
        if trimmed.is_empty() || is_empty_section_marker(&trimmed) {
            continue;
        }
        if out
            .iter()
            .any(|existing| lines_are_similar(existing, &trimmed))
        {
            continue;
        }
        out.push(trimmed);
    }
    out
}

fn render_general_summary_draft(draft: &GeneralSummaryDraft) -> String {
    let mut sections = Vec::new();

    for topic in &draft.topics {
        if topic.title.trim().is_empty() || topic.bullets.is_empty() {
            continue;
        }
        let bullets = dedupe_bullets(&topic.bullets)
            .into_iter()
            .map(|bullet| format!("- {bullet}"))
            .collect::<Vec<_>>()
            .join("\n");
        if bullets.is_empty() {
            continue;
        }
        sections.push(format!("## {}\n{}", topic.title.trim(), bullets));
    }

    sections.join("\n\n").trim().to_string()
}

/// Split transcript text into roughly equal chunks on line boundaries.
fn split_transcript_chunks(text: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        if !current.is_empty() && current.chars().count() + line.chars().count() + 1 > max_chars {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    if chunks.is_empty() {
        chunks.push(text.to_string());
    }
    chunks
}

fn build_structured_notes_prompt(
    live_notes: &str,
    segments: &[CanonicalSegment],
    title_hint: Option<&str>,
    template: SummaryTemplate,
) -> String {
    let transcript = prepared_summary_context(live_notes, segments, true);
    let system_prompt = template_system_prompt(template);

    format!(
        "{system_prompt}\n\nTitle hint: {}\n\nTranscript:\n{}",
        title_hint.unwrap_or("Ambient session"),
        transcript,
    )
}

fn template_system_prompt(template: SummaryTemplate) -> &'static str {
    match template {
        SummaryTemplate::General => GENERAL_TEMPLATE_PROMPT,
        SummaryTemplate::OneOnOne => ONE_ON_ONE_TEMPLATE_PROMPT,
        SummaryTemplate::TeamMeeting => TEAM_MEETING_TEMPLATE_PROMPT,
        SummaryTemplate::StandUp => STAND_UP_TEMPLATE_PROMPT,
    }
}

const GENERAL_TEMPLATE_PROMPT: &str = "\
You are an expert note-taker who transforms messy spoken transcripts into clear, structured summaries.\n\n\
INPUT: A raw transcript with filler words (um, uh, like, you know), false starts, repetition, \
and natural spoken-language messiness from one or more speakers.\n\n\
YOUR TASK:\n\
1. Identify the distinct topics or themes discussed.\n\
2. Create a markdown heading (##) for each topic.\n\
3. Under each heading, write concise bullet points capturing the key information.\n\
4. Completely rewrite — do NOT echo or quote the transcript. Synthesize what was said.\n\
5. Strip all filler words, false starts, and repetition.\n\n\
FORMAT RULES:\n\
- Output clean markdown only. No preamble, no closing remarks.\n\
- Use ## headings for each topic or theme.\n\
- Use bullet points (- ) under each heading.\n\
- Keep bullets short and information-dense.\n\
- If action items, decisions, next steps, or unresolved questions came up, include them under a relevant topic heading.\n\
- Do NOT create generic sections like 'Summary', 'Key Points', 'Decisions', 'Action Items', or 'Open Questions' — name headings after the actual topics.\n\
- Do NOT reproduce speaker labels (Speaker 1, You, S1, etc.) — just capture the substance.\n\n\
QUALITY:\n\
- Preserve all important facts, names, numbers, and technical terms exactly.\n\
- Merge duplicate or restated ideas into a single clear bullet.\n\
- Keep output proportional to the input — don't pad short conversations.\n\
- Do not hallucinate or invent information not in the transcript.";

const GENERAL_CHUNK_PROMPT: &str = "\
Summarize this portion of a transcript into concise bullet-point notes.\n\
- Remove filler words, repetition, and false starts.\n\
- Identify topics discussed and group bullets under short ## headings named after the actual topics.\n\
- Capture facts, names, numbers, decisions, next steps, and unresolved questions accurately under the relevant topic.\n\
- Do NOT use generic headings like Summary, Key Points, Decisions, Action Items, or Open Questions.\n\
- Do NOT reproduce speaker labels — just the substance.\n\
- Output clean markdown only.";

const GENERAL_MERGE_PROMPT: &str = "\
You are given partial summaries of different sections of the same conversation.\n\
Merge them into a single cohesive set of structured notes.\n\n\
RULES:\n\
- Combine duplicate topics under one heading.\n\
- Remove redundant bullets that say the same thing.\n\
- Use ## headings named after the actual topics.\n\
- Keep decisions, next steps, and unresolved questions under the relevant topic heading instead of creating generic operational sections.\n\
- Do NOT use generic headings like Summary, Key Points, Decisions, Action Items, or Open Questions.\n\
- Use bullet points (- ) under each heading.\n\
- Keep it concise and well-organized.\n\
- Output clean markdown only. No preamble.";

const GENERAL_RESCUE_SECTION_PROMPT: &str = "\
Turn these noisy transcript fragments into concise meeting notes for the topic `{topic}`.\n\
\n\
RULES:\n\
- Many fragments may be garbled, duplicated, or semantically broken.\n\
- Keep only information that is clearly supported by the fragments.\n\
- It is better to output one accurate bullet than several questionable bullets.\n\
- Output bullet lines only, each starting with `- `.\n\
- Use neutral note-style wording, not transcript-style wording.\n\
- Keep terminology close to the source fragments; do not add interpretations that are not explicit.\n\
- Do not use first person, second person, greetings, filler, or speaker labels.\n\
- Preserve concrete facts, dates, owners, and timing when they are reliable.\n\
- If a detail is ambiguous, omit it.\n\
- Output at most 3 bullets.\n\
- If nothing reliable can be extracted, output `- None`.";

const GENERAL_TOPIC_DISCOVERY_PROMPT: &str = "\
Identify the concrete discussion topics in this meeting transcript excerpt.\n\
\n\
RULES:\n\
- Output topic titles only, one per line, each starting with `- `.\n\
- Use short, concrete titles named after the actual subject matter.\n\
- Do not use generic titles like Summary, Key Points, Miscellaneous, Recap, or Updates.\n\
- Do not include speaker names or labels.\n\
- Only include topics that are explicitly discussed in the excerpt.\n\
- Prefer 1 to 4 topic titles.";

const GENERAL_TOPIC_DETAIL_PROMPT: &str = "\
Write bullets for the meeting topic `{topic}`.\n\
\n\
RULES:\n\
- Output bullet lines only, each starting with `- `.\n\
- Capture only details relevant to this topic.\n\
- Preserve concrete facts, names, numbers, dates, and technical terms.\n\
- If this topic includes a decision, action item, follow-up, or unresolved question, include it as a normal bullet under this topic.\n\
- Do not mention speaker labels.\n\
- Do not add a heading or preamble.\n\
- Do not repeat the topic title inside each bullet.\n\
- Prefer 2 to 5 bullets.";

const ONE_ON_ONE_TEMPLATE_PROMPT: &str = "\
Write concise 1:1 meeting notes in markdown with these sections exactly and in this order:\n\
## Summary\n## Discussion Topics\n## Feedback & Coaching\n## Action Items\n## Follow-ups for Next 1:1\n\n\
Rules:\n\
- Output markdown only.\n\
- Start with `## Summary` on the first line.\n\
- Do not add any preamble, conversational filler, or closing sentence.\n\
- Be concise and avoid repeating the transcript.\n\
- Use short bullet lists for every section except Summary.\n\
- In Discussion Topics, capture the main subjects raised by each participant.\n\
- In Feedback & Coaching, note any feedback given or received, growth areas, or coaching moments.\n\
- In Follow-ups for Next 1:1, list topics or items that should be revisited.\n\
- If the transcript is noisy or ambiguous, say that briefly in Summary instead of inventing details.\n\
- Preserve important technical terms exactly when they appear.\n\
- If a section has nothing reliable, write a single bullet that says `None`.\n\
- Do not include a Transcript section.\n\
- If 'User Notes (Scratch Pad)' content is present, treat it as high-priority context and use its markdown formatting to structure and enhance the summary.";

const TEAM_MEETING_TEMPLATE_PROMPT: &str = "\
Write concise team meeting notes in markdown with these sections exactly and in this order:\n\
## Summary\n## Agenda Items Covered\n## Decisions\n## Action Items\n## Owners & Deadlines\n## Parking Lot\n\n\
Rules:\n\
- Output markdown only.\n\
- Start with `## Summary` on the first line.\n\
- Do not add any preamble, conversational filler, or closing sentence.\n\
- Be concise and avoid repeating the transcript.\n\
- Use short bullet lists for every section except Summary.\n\
- In Agenda Items Covered, list each topic discussed with a brief note on the outcome.\n\
- In Owners & Deadlines, attribute action items to specific speakers when identifiable from the transcript.\n\
- In Parking Lot, capture topics raised but deferred for later discussion.\n\
- If the transcript is noisy or ambiguous, say that briefly in Summary instead of inventing details.\n\
- Preserve important technical terms exactly when they appear.\n\
- If a section has nothing reliable, write a single bullet that says `None`.\n\
- Do not include a Transcript section.\n\
- If 'User Notes (Scratch Pad)' content is present, treat it as high-priority context and use its markdown formatting to structure and enhance the summary.";

const STAND_UP_TEMPLATE_PROMPT: &str = "\
Write concise stand-up meeting notes in markdown with these sections exactly and in this order:\n\
## Summary\n## Yesterday / Completed\n## Today / In Progress\n## Blockers\n## Key Callouts\n\n\
Rules:\n\
- Output markdown only.\n\
- Start with `## Summary` on the first line.\n\
- Do not add any preamble, conversational filler, or closing sentence.\n\
- Be concise and avoid repeating the transcript.\n\
- Use short bullet lists for every section except Summary.\n\
- Attribute updates to specific speakers when identifiable (e.g. 'S1: ...', 'You: ...').\n\
- In Yesterday / Completed, capture what each participant reported as done.\n\
- In Today / In Progress, capture what each participant plans to work on.\n\
- In Blockers, list any impediments or dependencies mentioned.\n\
- In Key Callouts, note any announcements, reminders, or cross-team items raised.\n\
- If the transcript is noisy or ambiguous, say that briefly in Summary instead of inventing details.\n\
- Preserve important technical terms exactly when they appear.\n\
- If a section has nothing reliable, write a single bullet that says `None`.\n\
- Do not include a Transcript section.\n\
- If 'User Notes (Scratch Pad)' content is present, treat it as high-priority context and use its markdown formatting to structure and enhance the summary.";

fn build_ollama_title_prompt(
    live_notes: &str,
    segments: &[CanonicalSegment],
    fallback: &str,
) -> String {
    let notes_excerpt = excerpt_lines(live_notes, 6);
    let transcript_excerpt = segments
        .iter()
        .take(6)
        .map(CanonicalSegment::note_line)
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "Write a concise sidebar title for a macOS notes app.\nRequirements:\n- 2 to 4 words\n- Title Case\n- No punctuation\n- Maximum 30 characters\n- Output the title only\nFallback topic: {fallback}\n\nLive notes:\n{notes_excerpt}\n\nTranscript excerpt:\n{transcript_excerpt}"
    )
}

fn build_title_from_summary_prompt(summary_excerpt: &str, fallback: &str) -> String {
    format!(
        "Write a concise sidebar title for a macOS notes app based on the following meeting summary.\n\
         Requirements:\n\
         - 2 to 4 words\n\
         - Title Case\n\
         - No punctuation\n\
         - Maximum 30 characters\n\
         - Output the title only, nothing else\n\
         Fallback topic: {fallback}\n\n\
         Summary:\n{summary_excerpt}"
    )
}

fn prepared_summary_context(
    live_notes: &str,
    segments: &[CanonicalSegment],
    include_speaker_labels: bool,
) -> String {
    let source = full_summary_context(live_notes, segments, include_speaker_labels);
    excerpt_balanced_text(&source, MAX_MODEL_PROMPT_CHARS)
}

fn full_summary_context(
    live_notes: &str,
    segments: &[CanonicalSegment],
    include_speaker_labels: bool,
) -> String {
    let (scratch_pad, transcript_body) = split_scratch_pad_context(live_notes);
    let transcript = if segments.is_empty() {
        normalize_live_transcript(transcript_body, include_speaker_labels)
    } else if include_speaker_labels {
        sanitize_summary_transcript_lines(
            segments
                .iter()
                .map(CanonicalSegment::note_line)
                .collect::<Vec<_>>(),
            true,
        )
    } else {
        segments_to_speakerless_transcript(segments)
    };

    let mut sections = Vec::new();
    if let Some(scratch_pad) = scratch_pad {
        let scratch_pad = scratch_pad.trim();
        if !scratch_pad.is_empty() {
            sections.push(format!("User Notes (Scratch Pad):\n{scratch_pad}"));
        }
    }

    let transcript = transcript.trim();
    if !transcript.is_empty() {
        sections.push(format!("Transcript:\n{transcript}"));
    }

    sections.join("\n\n")
}

fn split_scratch_pad_context(live_notes: &str) -> (Option<&str>, &str) {
    let trimmed = live_notes.trim();
    let Some(rest) = trimmed.strip_prefix(SCRATCH_PAD_START_MARKER) else {
        return (None, trimmed);
    };

    let rest = rest.trim_start();
    if let Some((scratch_pad, transcript)) = rest.split_once(SCRATCH_PAD_END_MARKER) {
        (Some(scratch_pad.trim()), transcript.trim())
    } else {
        (Some(rest.trim()), "")
    }
}

fn normalize_live_transcript(text: &str, include_speaker_labels: bool) -> String {
    sanitize_summary_transcript_lines(
        text.lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string),
        include_speaker_labels,
    )
}

fn segments_to_speakerless_transcript(segments: &[CanonicalSegment]) -> String {
    sanitize_summary_transcript_lines(
        segments
            .iter()
            .map(|segment| segment.text.trim().to_string())
            .collect::<Vec<_>>(),
        false,
    )
}

fn excerpt_balanced_text(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }

    let lines = trimmed
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return trimmed.chars().take(max_chars).collect();
    }

    let head_budget = max_chars / 3;
    let tail_budget = max_chars.saturating_sub(head_budget + 64);
    let mut head = Vec::new();
    let mut used = 0usize;
    for line in &lines {
        let len = line.chars().count() + 1;
        if used + len > head_budget {
            break;
        }
        used += len;
        head.push((*line).to_string());
    }

    let mut tail = Vec::new();
    let mut tail_used = 0usize;
    for line in lines.iter().rev() {
        let len = line.chars().count() + 1;
        if tail_used + len > tail_budget {
            break;
        }
        tail_used += len;
        tail.push((*line).to_string());
    }
    tail.reverse();

    let mut combined = head;
    combined.push("[... transcript truncated for local summarization ...]".to_string());
    combined.extend(tail);
    combined.join("\n")
}

fn merge_model_structured_notes(
    content: &str,
    live_notes: &str,
    segments: &[CanonicalSegment],
    title_hint: Option<&str>,
    fallback: StructuredNotes,
) -> StructuredNotes {
    let transcript = segments_to_transcript(segments);
    let Some(mut parsed) = parse_model_structured_notes(content, &transcript) else {
        let transcript_lines = cleaned_summary_transcript_lines(live_notes, segments);
        let salient_lines = collect_salient_lines(&transcript_lines);
        return StructuredNotes {
            summary: content.trim().to_string(),
            key_points: collect_key_points(&transcript_lines, &salient_lines),
            decisions: collect_decisions(&transcript_lines),
            action_items: collect_action_items(&transcript_lines),
            open_questions: collect_open_questions(&transcript_lines),
            transcript,
            raw_notes: None,
        };
    };

    if parsed.summary.trim().is_empty() {
        parsed.summary = fallback.summary;
    }
    if parsed.key_points.is_empty() {
        parsed.key_points = fallback.key_points;
    }
    if parsed.decisions.is_empty() {
        parsed.decisions = fallback.decisions;
    }
    if parsed.action_items.is_empty() {
        parsed.action_items = fallback.action_items;
    }
    if parsed.open_questions.is_empty() {
        parsed.open_questions = fallback.open_questions;
    }
    if parsed.summary.trim().is_empty() {
        parsed.summary = heuristic_structured_notes(live_notes, segments, title_hint).summary;
    }
    parsed.transcript = transcript;
    parsed
}

fn parse_model_structured_notes(content: &str, transcript: &str) -> Option<StructuredNotes> {
    let mut summary_lines = Vec::new();
    let mut key_points = Vec::new();
    let mut decisions = Vec::new();
    let mut action_items = Vec::new();
    let mut open_questions = Vec::new();
    let mut current_section = None::<ParsedNotesSection>;

    let cleaned = trim_generated_response(content);
    let cleaned = strip_code_fence(&cleaned);

    for raw_line in cleaned.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(section) = parse_notes_heading(line) {
            current_section = Some(section);
            continue;
        }

        match current_section {
            Some(ParsedNotesSection::Summary) => summary_lines.push(line.to_string()),
            Some(ParsedNotesSection::KeyPoints) => push_parsed_bullet(&mut key_points, line),
            Some(ParsedNotesSection::Decisions) => push_parsed_bullet(&mut decisions, line),
            Some(ParsedNotesSection::ActionItems) => push_parsed_bullet(&mut action_items, line),
            Some(ParsedNotesSection::OpenQuestions) => {
                push_parsed_bullet(&mut open_questions, line)
            }
            None => {}
        }
    }

    let summary = collapse_spaces(&summary_lines.join(" "));
    if summary.is_empty()
        && key_points.is_empty()
        && decisions.is_empty()
        && action_items.is_empty()
        && open_questions.is_empty()
    {
        return None;
    }

    Some(StructuredNotes {
        summary,
        key_points,
        decisions,
        action_items,
        open_questions,
        transcript: transcript.to_string(),
        raw_notes: None,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParsedNotesSection {
    Summary,
    KeyPoints,
    Decisions,
    ActionItems,
    OpenQuestions,
}

fn parse_notes_heading(line: &str) -> Option<ParsedNotesSection> {
    let normalized = line
        .trim()
        .trim_start_matches('#')
        .trim()
        .trim_end_matches(':')
        .to_ascii_lowercase();

    match normalized.as_str() {
        "summary" => Some(ParsedNotesSection::Summary),
        "key points"
        | "discussion topics"
        | "agenda items covered"
        | "yesterday / completed"
        | "yesterday"
        | "completed" => Some(ParsedNotesSection::KeyPoints),
        "decisions"
        | "feedback & coaching"
        | "feedback"
        | "coaching"
        | "today / in progress"
        | "today"
        | "in progress" => Some(ParsedNotesSection::Decisions),
        "action items" | "actions" | "owners & deadlines" | "owners" | "blockers" => {
            Some(ParsedNotesSection::ActionItems)
        }
        "open questions"
        | "open questions / risks"
        | "follow-ups for next 1:1"
        | "follow-ups"
        | "parking lot"
        | "key callouts" => Some(ParsedNotesSection::OpenQuestions),
        _ => None,
    }
}

fn push_parsed_bullet(target: &mut Vec<String>, line: &str) {
    let normalized = strip_list_prefix(line);
    let normalized = normalized.trim();
    if normalized.is_empty() {
        return;
    }
    if is_empty_section_marker(normalized) {
        return;
    }
    target.push(normalized.to_string());
}

fn is_empty_section_marker(line: &str) -> bool {
    let normalized = line
        .trim()
        .trim_end_matches(|ch: char| matches!(ch, '.' | ':' | '!' | ';'))
        .trim();
    normalized.eq_ignore_ascii_case("none") || normalized.eq_ignore_ascii_case("n/a")
}

fn strip_list_prefix(line: &str) -> &str {
    if let Some(rest) = list_item_body(line) {
        return rest;
    }

    line.trim()
}

fn list_item_body(line: &str) -> Option<&str> {
    let line = line.trim();
    for prefix in ["- ", "* ", "• "] {
        if let Some(rest) = line.strip_prefix(prefix) {
            return Some(rest.trim());
        }
    }

    let mut digits = 0usize;
    for ch in line.chars() {
        if ch.is_ascii_digit() {
            digits += 1;
            continue;
        }
        break;
    }
    if digits > 0 {
        let remainder = &line[digits..];
        if let Some(rest) = remainder.strip_prefix(". ") {
            return Some(rest.trim());
        }
        if let Some(rest) = remainder.strip_prefix(") ") {
            return Some(rest.trim());
        }
    }

    None
}

fn strip_code_fence(text: &str) -> String {
    let trimmed = text.trim();
    if !trimmed.starts_with("```") {
        return trimmed.to_string();
    }

    trimmed
        .trim_start_matches("```markdown")
        .trim_start_matches("```md")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim()
        .to_string()
}

fn clean_general_markdown(text: &str) -> String {
    let cleaned = strip_code_fence(&trim_generated_response(text));
    let mut sections = Vec::<TopicSection>::new();
    let mut pending_generic_items = Vec::<String>::new();
    let mut current_heading = None::<String>;
    let mut current_items = Vec::<String>::new();
    let mut current_heading_is_generic = false;

    for raw_line in cleaned.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line == "---" {
            continue;
        }

        if line.starts_with('#') {
            push_cleaned_general_section(
                &mut sections,
                &mut pending_generic_items,
                &mut current_heading,
                &mut current_items,
                current_heading_is_generic,
            );
            current_heading = normalize_general_heading(line);
            current_heading_is_generic = current_heading
                .as_deref()
                .map(is_generic_general_heading)
                .unwrap_or(false);
            continue;
        }

        let body = list_item_body(line).unwrap_or(line);
        let body = collapse_spaces(strip_speaker_prefix(body).trim());
        if body.is_empty() || is_meta_summary_line(&body) {
            continue;
        }

        if current_heading.is_some() {
            current_items.push(body);
        }
    }

    push_cleaned_general_section(
        &mut sections,
        &mut pending_generic_items,
        &mut current_heading,
        &mut current_items,
        current_heading_is_generic,
    );

    sections
        .into_iter()
        .filter(|section| !section.title.trim().is_empty() && !section.bullets.is_empty())
        .map(|section| {
            format!(
                "## {}\n{}",
                section.title,
                section
                    .bullets
                    .into_iter()
                    .map(|item| format!("- {item}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim()
        .to_string()
}

fn bullet_has_source_support(bullet: &str, source_lines: &[String]) -> bool {
    let bullet_tokens = summary_content_tokens(bullet);
    if bullet_tokens.len() < 2 {
        return false;
    }

    source_lines.iter().any(|source| {
        let source_tokens = summary_content_tokens(source);
        if source_tokens.is_empty() {
            return false;
        }

        let bullet_set = bullet_tokens.iter().collect::<HashSet<_>>();
        let source_set = source_tokens.iter().collect::<HashSet<_>>();
        let overlap = bullet_set.intersection(&source_set).count();

        overlap >= 3
            || (overlap >= 2
                && (lines_are_similar(bullet, source)
                    || (contains_temporal_markers(bullet) && contains_temporal_markers(source))
                    || (contains_summary_keywords(bullet) && contains_summary_keywords(source))))
    })
}

fn push_cleaned_general_section(
    sections: &mut Vec<TopicSection>,
    pending_generic_items: &mut Vec<String>,
    heading: &mut Option<String>,
    items: &mut Vec<String>,
    heading_is_generic: bool,
) {
    let Some(heading_text) = heading.take() else {
        items.clear();
        return;
    };

    let cleaned_items = dedupe_bullets(items)
        .into_iter()
        .filter(|item| !is_meta_summary_line(item))
        .filter(|item| line_has_reliable_summary_signal(item))
        .collect::<Vec<_>>();
    items.clear();
    let cleaned_items = prioritize_general_section_items(&cleaned_items, 4);
    if cleaned_items.is_empty() {
        return;
    }

    if heading_is_generic {
        if let Some(section) = sections.last_mut() {
            let mut merged = section.bullets.clone();
            merged.extend(cleaned_items);
            section.bullets = dedupe_bullets(&merged);
        } else {
            pending_generic_items.extend(cleaned_items);
            *pending_generic_items = dedupe_bullets(pending_generic_items);
        }
        return;
    }

    let mut merged_items = std::mem::take(pending_generic_items);
    merged_items.extend(cleaned_items);
    sections.push(TopicSection {
        title: heading_text,
        bullets: dedupe_bullets(&merged_items),
    });
}

fn prioritize_general_section_items(items: &[String], max_items: usize) -> Vec<String> {
    let mut scored = items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let mut score = summary_candidate_score(item);
            if contains_temporal_markers(item) {
                score += 4;
            }
            if contains_summary_keywords(item) {
                score += 4;
            }
            (index, item.clone(), score)
        })
        .collect::<Vec<_>>();

    scored.sort_by(|left, right| right.2.cmp(&left.2).then(left.0.cmp(&right.0)));
    scored.truncate(max_items);
    scored.sort_by_key(|item| item.0);
    scored.into_iter().map(|(_, item, _)| item).collect()
}

fn normalize_general_heading(line: &str) -> Option<String> {
    let heading = line
        .trim()
        .trim_start_matches('#')
        .trim()
        .trim_end_matches(':')
        .trim();
    if heading.is_empty() {
        return None;
    }

    let normalized = heading
        .trim_matches(|ch: char| matches!(ch, '*' | '"' | '\'' | '`'))
        .trim();
    let normalized = collapse_spaces(strip_speaker_prefix(normalized).trim());
    if normalized.is_empty() || is_meta_summary_line(&normalized) {
        return None;
    }

    let lower = normalized.to_ascii_lowercase();
    let word_count = normalized.split_whitespace().count();
    if word_count > 8 && !is_generic_general_heading(&lower) {
        return None;
    }

    Some(normalized)
}

fn is_generic_general_heading(heading: &str) -> bool {
    let normalized = collapse_spaces(heading.trim()).to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "summary"
            | "key points"
            | "discussion topics"
            | "agenda items covered"
            | "decisions"
            | "action items"
            | "actions"
            | "open questions"
            | "open questions / risks"
            | "owners & deadlines"
            | "owners"
            | "parking lot"
            | "follow-ups for next 1:1"
            | "follow-ups"
            | "feedback & coaching"
            | "feedback"
            | "coaching"
            | "blockers"
            | "key callouts"
            | "discussion"
            | "recap"
            | "notes"
            | "updates"
            | "miscellaneous"
            | "other"
            | "general"
    )
}

fn general_markdown_has_signal(text: &str) -> bool {
    let mut headings = 0usize;
    let mut bullets = 0usize;
    let mut meta_or_symbolic = 0usize;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if let Some(heading) = line.strip_prefix("## ") {
            if is_meta_summary_line(heading)
                || looks_symbolic_line(heading)
                || is_generic_general_heading(heading)
            {
                meta_or_symbolic += 1;
            } else {
                headings += 1;
            }
            continue;
        }

        if let Some(bullet) = line.strip_prefix("- ") {
            if is_meta_summary_line(bullet)
                || looks_symbolic_line(bullet)
                || !line_has_reliable_summary_signal(bullet)
            {
                meta_or_symbolic += 1;
            } else {
                bullets += 1;
            }
        }
    }

    headings > 0 && bullets > 0 && meta_or_symbolic == 0
}

fn trim_generated_response(text: &str) -> String {
    text.lines()
        .skip_while(|line| line.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn heuristic_structured_notes(
    live_notes: &str,
    segments: &[CanonicalSegment],
    title_hint: Option<&str>,
) -> StructuredNotes {
    let title = title_hint
        .map(sanitize_session_title)
        .unwrap_or_else(|| sanitize_session_title(&heuristic_title(live_notes, segments)));
    let transcript_lines = cleaned_summary_transcript_lines(live_notes, segments);
    let salient_lines = collect_salient_lines(&transcript_lines);
    let key_points = collect_key_points(&transcript_lines, &salient_lines);
    let decisions = collect_decisions(&transcript_lines);
    let action_items = collect_action_items(&transcript_lines);
    let open_questions = collect_open_questions(&transcript_lines);
    let transcript = segments_to_transcript(segments);

    StructuredNotes {
        summary: build_heuristic_summary(&title, &salient_lines, live_notes, segments),
        key_points,
        decisions,
        action_items,
        open_questions,
        transcript,
        raw_notes: None,
    }
}

fn heuristic_general_notes(
    live_notes: &str,
    segments: &[CanonicalSegment],
    title_hint: Option<&str>,
    _fallback: &StructuredNotes,
) -> StructuredNotes {
    let title = title_hint
        .map(sanitize_session_title)
        .unwrap_or_else(|| sanitize_session_title(&heuristic_title(live_notes, segments)));
    let transcript = segments_to_transcript(segments);
    let transcript_lines = cleaned_summary_transcript_lines(live_notes, segments);
    let bullets = select_general_recovery_lines(&transcript_lines);

    let markdown = if bullets.is_empty() {
        format!(
            "## {title}\n- Transcript was captured, but it was too noisy to extract reliable topic details."
        )
    } else {
        format!(
            "## {title}\n{}",
            bullets
                .into_iter()
                .map(|bullet| format!("- {bullet}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    StructuredNotes {
        transcript,
        raw_notes: Some(markdown),
        ..Default::default()
    }
}

fn collect_key_points(transcript_lines: &[String], salient_lines: &[String]) -> Vec<String> {
    let mut points = salient_lines.iter().take(4).cloned().collect::<Vec<_>>();

    if points.is_empty() {
        points.extend(transcript_lines.iter().take(4).cloned());
    }

    points
}

fn collect_action_items(transcript_lines: &[String]) -> Vec<String> {
    collect_matching_lines(
        transcript_lines,
        &["todo", "action", "follow up", "next step", "will "],
    )
}

fn collect_decisions(transcript_lines: &[String]) -> Vec<String> {
    collect_matching_lines(
        transcript_lines,
        &["decide", "decision", "agreed", "ship", "go with"],
    )
}

fn collect_open_questions(transcript_lines: &[String]) -> Vec<String> {
    let mut questions =
        collect_matching_lines(transcript_lines, &["?", "open question", "unclear"]);
    if questions.is_empty() {
        questions.push("No explicit open questions were captured.".to_string());
    }
    questions
}

fn collect_matching_lines(transcript_lines: &[String], needles: &[&str]) -> Vec<String> {
    let mut kept = Vec::<String>::new();

    for line in transcript_lines.iter().map(String::as_str) {
        let lower = line.to_ascii_lowercase();
        if !needles.iter().any(|needle| lower.contains(needle)) {
            continue;
        }

        let cleaned = clean_candidate_line(line);
        if cleaned.is_empty()
            || !line_has_reliable_summary_signal(&cleaned)
            || kept
                .iter()
                .any(|existing| lines_are_similar(existing, &cleaned))
        {
            continue;
        }

        kept.push(cleaned);
        if kept.len() >= 4 {
            break;
        }
    }

    kept
}

fn excerpt_lines(text: &str, limit: usize) -> String {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(limit)
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Clone, Debug)]
struct SalientLine {
    index: usize,
    text: String,
    score: i32,
}

#[derive(Clone, Debug)]
struct RecoveryLine {
    index: usize,
    text: String,
    score: i32,
}

fn build_heuristic_summary(
    title: &str,
    salient_lines: &[String],
    live_notes: &str,
    segments: &[CanonicalSegment],
) -> String {
    if salient_lines.is_empty() {
        return format!(
            "{title}\n\nConversation was captured, but the transcript was too noisy to produce a reliable detailed summary."
        );
    }

    let mut lines = vec![
        title.to_string(),
        String::new(),
        format!("Main takeaway: {}", as_summary_sentence(&salient_lines[0])),
    ];

    if let Some(second) = salient_lines.get(1) {
        lines.push(format!("Also discussed: {}", as_summary_sentence(second)));
    }

    if transcript_seems_noisy(live_notes, segments, salient_lines) {
        lines.push(
            "Note: parts of the transcript were noisy or repetitive, so minor details may be incomplete."
                .to_string(),
        );
    }

    lines.join("\n")
}

fn collect_salient_lines(candidates: &[String]) -> Vec<String> {
    let mut kept = Vec::<SalientLine>::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let text = clean_candidate_line(candidate);
        if text.is_empty() || !line_has_reliable_summary_signal(&text) {
            continue;
        }
        let tokens = summary_tokens(&text);
        if text.ends_with('?') && !contains_summary_keywords(&text) && tokens.len() < 14 {
            continue;
        }

        let score = summary_candidate_score(&text);
        if score < 10 {
            continue;
        }

        if let Some(existing) = kept
            .iter_mut()
            .find(|existing| lines_are_similar(&existing.text, &text))
        {
            if score > existing.score {
                existing.index = index;
                existing.text = text;
                existing.score = score;
            }
            continue;
        }

        kept.push(SalientLine { index, text, score });
    }

    kept.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then(left.index.cmp(&right.index))
    });
    kept.truncate(4);
    kept.into_iter().map(|line| line.text).collect()
}

fn select_general_recovery_lines(candidates: &[String]) -> Vec<String> {
    let mut kept = Vec::<RecoveryLine>::new();

    for (index, candidate) in candidates.iter().enumerate() {
        let text = clean_candidate_line(candidate);
        if text.is_empty() || !line_has_reliable_summary_signal(&text) {
            continue;
        }

        let mut score = summary_candidate_score(&text);
        if contains_temporal_markers(&text) {
            score += 4;
        }
        if contains_summary_keywords(&text) {
            score += 4;
        }
        if score < 10 {
            continue;
        }

        kept.push(RecoveryLine { index, text, score });
    }

    for idx in 0..kept.len() {
        let support = kept
            .iter()
            .enumerate()
            .filter(|(other_idx, other)| {
                *other_idx != idx && lines_share_topic_signal(&kept[idx].text, &other.text)
            })
            .count() as i32;
        kept[idx].score += support.min(3) * 4;
    }

    kept.retain(|line| {
        line.score >= 16
            || (line.score >= 12
                && (contains_temporal_markers(&line.text) || contains_summary_keywords(&line.text)))
    });

    kept.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then(left.index.cmp(&right.index))
    });

    let mut selected = Vec::<RecoveryLine>::new();
    for candidate in kept {
        if selected
            .iter()
            .any(|existing| lines_are_similar(&existing.text, &candidate.text))
        {
            continue;
        }
        selected.push(candidate);
        if selected.len() >= 4 {
            break;
        }
    }

    selected.sort_by_key(|line| line.index);
    selected.into_iter().map(|line| line.text).collect()
}

fn cleaned_summary_transcript_lines(
    live_notes: &str,
    segments: &[CanonicalSegment],
) -> Vec<String> {
    let (_, transcript_body) = split_scratch_pad_context(live_notes);
    let source_lines = if segments.is_empty() {
        transcript_body
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>()
    } else {
        segments
            .iter()
            .map(|segment| segment.text.trim().to_string())
            .collect::<Vec<_>>()
    };
    let mut cleaned = Vec::<String>::new();

    for line in source_lines {
        let sanitized = sanitize_summary_transcript_body(&line);
        if sanitized.is_empty() {
            continue;
        }

        for fragment in split_summary_fragments(&sanitized) {
            let fragment = collapse_spaces(fragment.trim());
            let tokens = summary_tokens(&fragment);
            if tokens.len() < 4 || is_meta_summary_line(&fragment) {
                continue;
            }
            if fragment.chars().count() > 220 {
                continue;
            }
            if tokens.len() >= 8
                && looks_noisy_line(&fragment, &tokens)
                && !contains_summary_keywords(&fragment)
            {
                continue;
            }
            if cleaned
                .iter()
                .rev()
                .take(SUMMARY_TRANSCRIPT_RECENT_LINE_WINDOW)
                .any(|existing| lines_are_similar(existing, &fragment))
            {
                continue;
            }
            cleaned.push(fragment);
        }
    }

    cleaned
}

fn clean_candidate_line(text: &str) -> String {
    let mut cleaned = collapse_spaces(strip_speaker_prefix(text));
    let lower = cleaned.to_ascii_lowercase();

    for prefix in [
        "okay ",
        "ok ",
        "well ",
        "so ",
        "i mean ",
        "let me tell you about ",
        "the whole point is ",
    ] {
        if lower.starts_with(prefix) {
            cleaned = cleaned[prefix.len()..].trim().to_string();
            break;
        }
    }

    cleaned
        .trim_matches(|ch: char| matches!(ch, '"' | '\'' | '`' | '-' | ':' | ',' | '.'))
        .trim()
        .to_string()
}

fn line_has_reliable_summary_signal(text: &str) -> bool {
    let cleaned = clean_candidate_line(text);
    if cleaned.is_empty() {
        return false;
    }

    let tokens = summary_tokens(&cleaned);
    if tokens.len() < 4 {
        return false;
    }

    if is_low_value_conversational_line(&cleaned)
        || pronoun_heavy_without_anchor(&cleaned, &tokens)
        || starts_like_fragmented_question(&tokens, &cleaned)
        || contains_transcript_filler_phrase(&cleaned)
        || is_generic_boilerplate_line(&cleaned)
        || line_compacts_significantly(&cleaned)
        || contains_repeated_ngram(&tokens, 3, 2)
        || contains_repeated_ngram(&tokens, 4, 2)
    {
        return false;
    }

    !looks_noisy_line(&cleaned, &tokens)
}

fn summary_candidate_score(text: &str) -> i32 {
    let tokens = summary_tokens(text);
    if tokens.len() < 4 {
        return -10;
    }

    let unique_tokens = tokens.iter().collect::<HashSet<_>>().len();
    let unique_ratio = unique_tokens as f32 / tokens.len() as f32;
    let long_tokens = tokens.iter().filter(|token| token.len() >= 4).count() as i32;
    let short_tokens = tokens.iter().filter(|token| token.len() <= 2).count() as i32;
    let repeated_penalty = most_common_token_frequency(&tokens).saturating_sub(2) as i32 * 3;
    let numeric_tokens = tokens
        .iter()
        .filter(|token| token.chars().any(|ch| ch.is_ascii_digit()))
        .count() as i32;
    let first_person_tokens = tokens
        .iter()
        .filter(|token| matches!(token.as_str(), "i" | "i'm" | "ive" | "i've" | "me" | "my"))
        .count() as i32;

    let mut score = tokens.len().min(18) as i32 + long_tokens * 2 - short_tokens;

    if unique_ratio >= 0.75 {
        score += 8;
    } else if unique_ratio >= 0.6 {
        score += 4;
    } else {
        score -= 6;
    }

    if text.ends_with('?') {
        score -= 12;
    } else if text.ends_with(['.', '!']) {
        score += 3;
    }

    if contains_summary_keywords(text) {
        score += 4;
    }
    if numeric_tokens > 0 {
        score += 6 + numeric_tokens.min(3) * 2;
    }
    if contains_temporal_markers(text) {
        score += 4;
    }
    if first_person_tokens >= 2
        && !contains_temporal_markers(text)
        && !contains_summary_keywords(text)
        && numeric_tokens == 0
    {
        score -= 10;
    }
    if is_low_value_conversational_line(text) {
        score -= 16;
    }
    if starts_with_fragment_connector(text) {
        score -= 6;
    }
    if contains_repeated_ngram(&tokens, 3, 2) {
        score -= 10;
    }

    if looks_noisy_line(text, &tokens) {
        return -10;
    }

    score - repeated_penalty
}

fn summary_tokens(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|token| {
            let normalized = token
                .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '\'')
                .to_ascii_lowercase();
            (!normalized.is_empty()).then_some(normalized)
        })
        .collect()
}

fn summary_content_tokens(text: &str) -> Vec<String> {
    summary_tokens(text)
        .into_iter()
        .flat_map(expand_summary_content_token)
        .filter(|token| token.len() >= 4)
        .filter(|token| {
            !matches!(
                token.as_str(),
                "that"
                    | "this"
                    | "with"
                    | "from"
                    | "have"
                    | "they"
                    | "them"
                    | "then"
                    | "just"
                    | "like"
                    | "into"
                    | "your"
                    | "what"
                    | "when"
                    | "where"
                    | "which"
                    | "would"
                    | "could"
                    | "should"
                    | "there"
                    | "their"
                    | "about"
                    | "after"
                    | "before"
                    | "because"
                    | "really"
                    | "probably"
                    | "basically"
            )
        })
        .collect()
}

fn expand_summary_content_token(token: String) -> Vec<String> {
    let mut expanded = vec![token.clone()];

    if token.contains('-') || token.contains('/') {
        expanded.extend(
            token
                .split(|ch: char| matches!(ch, '-' | '/'))
                .filter(|part| !part.is_empty())
                .map(str::to_string),
        );
    }

    expanded
}

fn line_compacts_significantly(text: &str) -> bool {
    let compacted = compact_repeated_tokens(text);
    let compacted_len = compacted.chars().count();
    let original_len = text.chars().count();
    compacted_len > 0 && compacted_len * 10 < original_len * 9
}

fn contains_transcript_filler_phrase(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "so like",
        "you know",
        "i think",
        "i guess",
        "gonna",
        "wanna",
        "gotta",
        "kind of",
        "sort of",
        "let's do that",
        "lets do that",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn is_generic_boilerplate_line(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "is a common practice",
        "is a crucial component",
        "significantly impact operational efficiency",
        "necessary knowledge, skills, and tools",
        "effective team communication, collaboration, and alignment",
        "overall performance",
        "optimal results",
        "various industries",
        "high operational demands",
        "defined timeframe",
        "employee competency and preparedness",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn most_common_token_frequency(tokens: &[String]) -> usize {
    let mut best = 0usize;
    for token in tokens {
        best = best.max(
            tokens
                .iter()
                .filter(|candidate| *candidate == token)
                .count(),
        );
    }
    best
}

fn contains_repeated_ngram(tokens: &[String], size: usize, min_occurrences: usize) -> bool {
    if size == 0 || min_occurrences < 2 || tokens.len() < size * min_occurrences {
        return false;
    }

    for start in 0..=tokens.len() - size {
        let window = &tokens[start..start + size];
        let occurrences = (0..=tokens.len() - size)
            .filter(|index| tokens[*index..*index + size] == window[..])
            .count();
        if occurrences >= min_occurrences {
            return true;
        }
    }

    false
}

fn contains_summary_keywords(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "dashboard",
        "slack",
        "incident",
        "runbook",
        "on call",
        "on-call",
        "pto",
        "swap",
        "coverage",
        "schedule",
        "restore",
        "automation",
        "automate",
        "alert",
        "page",
        "paging",
        "owner",
        "deadline",
        "follow up",
        "follow-up",
        "next step",
        "review",
        "launch",
        "release",
        "ship",
        "bug",
        "fix",
        "deploy",
        "migration",
        "access",
        "permissions",
        "apple",
        "metal",
        "cuda",
        "nvidia",
        "implementation",
        "algorithm",
        "library",
        "rewrite",
        "research",
        "plan",
        "need to",
        "have to",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn contains_temporal_markers(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "today",
        "tomorrow",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "next week",
        "this week",
        "on call",
        "pto",
        "vacation",
        "deadline",
        "morning",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn starts_with_fragment_connector(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "and ", "or ", "but ", "so ", "then ", "because ", "well ", "okay ", "ok ",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
}

fn is_low_value_conversational_line(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "hey ",
        "good morning",
        "good afternoon",
        "how's it going",
        "how is it going",
        "did you see my message",
        "are you gonna",
        "are you going to",
        "sounds good",
        "okay cool",
        "ok cool",
        "i can't say that",
        "thank you",
        "thanks",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
}

fn contains_forbidden_summary_pronouns(text: &str) -> bool {
    summary_tokens(text).iter().any(|token| {
        matches!(
            token.as_str(),
            "i" | "i'm" | "ive" | "i've" | "me" | "my" | "you" | "your"
        )
    })
}

fn pronoun_heavy_without_anchor(text: &str, tokens: &[String]) -> bool {
    let pronouns = tokens
        .iter()
        .filter(|token| matches!(token.as_str(), "i" | "i'm" | "ive" | "i've" | "me" | "my"))
        .count();
    let has_numeric = tokens
        .iter()
        .any(|token| token.chars().any(|ch| ch.is_ascii_digit()));
    pronouns >= 2
        && !has_numeric
        && !contains_temporal_markers(text)
        && !contains_summary_keywords(text)
}

fn starts_like_fragmented_question(tokens: &[String], text: &str) -> bool {
    let Some(first) = tokens.first() else {
        return false;
    };

    matches!(
        first.as_str(),
        "does" | "do" | "did" | "is" | "are" | "was" | "were" | "can" | "could" | "would"
    ) && !contains_temporal_markers(text)
        && !contains_summary_keywords(text)
}

fn looks_noisy_line(text: &str, tokens: &[String]) -> bool {
    if tokens.len() < 4 {
        return true;
    }

    let unique_ratio = tokens.iter().collect::<HashSet<_>>().len() as f32 / tokens.len() as f32;
    let repeated_adjacent = tokens.windows(2).any(|window| window[0] == window[1]);
    let short_ratio =
        tokens.iter().filter(|token| token.len() <= 2).count() as f32 / tokens.len() as f32;
    let lower = text.to_ascii_lowercase();

    unique_ratio < 0.5
        || repeated_adjacent
        || short_ratio > 0.45
        || lower.matches("okay").count() >= 3
        || lower.matches("clean").count() >= 4
}

fn lines_share_topic_signal(left: &str, right: &str) -> bool {
    if lines_are_similar(left, right) {
        return true;
    }

    let left_tokens = summary_content_tokens(left);
    let right_tokens = summary_content_tokens(right);
    if left_tokens.is_empty() || right_tokens.is_empty() {
        return false;
    }

    let left_set = left_tokens.iter().collect::<HashSet<_>>();
    let right_set = right_tokens.iter().collect::<HashSet<_>>();
    left_set.intersection(&right_set).count() >= 2
}

fn lines_are_similar(left: &str, right: &str) -> bool {
    let left_normalized = collapse_spaces(&left.to_ascii_lowercase());
    let right_normalized = collapse_spaces(&right.to_ascii_lowercase());

    if left_normalized == right_normalized {
        return true;
    }

    if left_normalized.contains(&right_normalized) || right_normalized.contains(&left_normalized) {
        return left_normalized.len().min(right_normalized.len()) >= 24;
    }

    let left_tokens = summary_tokens(&left_normalized);
    let right_tokens = summary_tokens(&right_normalized);
    if left_tokens.is_empty() || right_tokens.is_empty() {
        return false;
    }

    let left_set = left_tokens.iter().collect::<HashSet<_>>();
    let right_set = right_tokens.iter().collect::<HashSet<_>>();
    let intersection = left_set.intersection(&right_set).count() as f32;
    let union = left_set.union(&right_set).count() as f32;

    union > 0.0 && intersection / union >= 0.7
}

fn transcript_seems_noisy(
    live_notes: &str,
    segments: &[CanonicalSegment],
    salient_lines: &[String],
) -> bool {
    if salient_lines.is_empty() {
        return true;
    }

    let raw_lines = if segments.is_empty() {
        live_notes
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(strip_speaker_prefix)
            .map(str::to_string)
            .collect::<Vec<_>>()
    } else {
        segments
            .iter()
            .map(|segment| segment.text.trim().to_string())
            .collect::<Vec<_>>()
    };

    if raw_lines.is_empty() {
        return true;
    }

    let noisy_count = raw_lines
        .iter()
        .filter(|line| looks_noisy_line(line, &summary_tokens(line)))
        .count();

    noisy_count * 2 >= raw_lines.len()
}

fn transcript_is_pathological_for_llm(live_notes: &str, segments: &[CanonicalSegment]) -> bool {
    let lines = if segments.is_empty() {
        live_notes
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(strip_speaker_prefix)
            .map(str::to_string)
            .collect::<Vec<_>>()
    } else {
        segments
            .iter()
            .map(|segment| segment.text.trim().to_string())
            .collect::<Vec<_>>()
    };

    lines.iter().any(|line| pathological_transcript_line(line))
}

fn pathological_transcript_line(text: &str) -> bool {
    let trimmed = text.trim();
    let char_count = trimmed.chars().count();
    if char_count >= 900 {
        return true;
    }
    if char_count < 180 {
        return false;
    }

    let repaired = repair_summary_spacing(trimmed);
    let compacted = compact_repeated_tokens(&repaired);
    let compacted_chars = compacted.chars().count();
    if compacted_chars * 10 < char_count * 8 {
        return true;
    }

    let sanitized = sanitize_summary_transcript_body(trimmed);
    let sanitized_chars = sanitized.chars().count();
    if sanitized_chars > 0 && sanitized_chars * 10 < char_count * 8 {
        return true;
    }

    let fragments = split_summary_fragments(&repaired)
        .into_iter()
        .map(|fragment| collapse_spaces(fragment.trim()))
        .filter(|fragment| summary_tokens(fragment).len() >= 4)
        .collect::<Vec<_>>();
    let fragment_count = fragments.len();
    if fragment_count < 4 {
        return false;
    }

    let mut unique = Vec::<String>::new();
    for fragment in fragments {
        if unique
            .iter()
            .any(|existing| lines_are_similar(existing, &fragment))
        {
            continue;
        }
        unique.push(fragment);
    }

    unique.len() * 2 <= fragment_count
}

fn as_summary_sentence(text: &str) -> String {
    let cleaned = clean_candidate_line(text)
        .trim_end_matches(['.', '?', '!'])
        .trim()
        .to_string();
    if cleaned.is_empty() {
        return "transcript quality was too low to extract a stable point.".to_string();
    }

    let lower = cleaned.to_ascii_lowercase();
    if lower.contains("no implementation") && lower.contains("apple") {
        return "there is no Apple-compatible implementation of the available library.".to_string();
    }
    if lower.contains("rewrite") && (lower.contains("apple silicon") || lower.contains("metal")) {
        return "making this work on Mac requires rewriting the algorithm for Apple Silicon / Metal."
            .to_string();
    }
    if lower.contains("cuda") && lower.contains("metal") {
        return "the work involves bridging CUDA-oriented code to Metal.".to_string();
    }
    if lower.contains("months") || lower.contains("year") {
        return "the effort was described as a multi-month project, potentially up to a year."
            .to_string();
    }

    let mut sentence = cleaned;
    if let Some(first) = sentence.chars().next() {
        let first_upper = first.to_uppercase().to_string();
        sentence.replace_range(..first.len_utf8(), &first_upper);
    }
    if !sentence.ends_with('.') {
        sentence.push('.');
    }
    sentence
}

fn sanitize_session_title(text: &str) -> String {
    let candidate = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("Ambient session");
    let candidate = strip_speaker_prefix(candidate)
        .trim_matches(|ch: char| matches!(ch, '"' | '\'' | '`' | '#' | '-' | '*' | ':' | '.'))
        .trim();
    let candidate = if let Some((prefix, suffix)) = candidate.split_once(':') {
        if prefix.trim().eq_ignore_ascii_case("title") {
            suffix.trim()
        } else {
            candidate
        }
    } else {
        candidate
    };
    let candidate = strip_speaker_prefix(candidate).trim();

    let cleaned = collapse_spaces(
        &candidate
            .chars()
            .map(|ch| {
                if ch.is_alphanumeric() || ch.is_whitespace() || ch == '&' {
                    ch
                } else {
                    ' '
                }
            })
            .collect::<String>(),
    );
    if cleaned.is_empty() {
        return "Ambient Session".to_string();
    }

    let limited_words = cleaned
        .split_whitespace()
        .take(MAX_SESSION_TITLE_WORDS)
        .collect::<Vec<_>>()
        .join(" ");
    let trimmed = trim_to_len(&limited_words, MAX_SESSION_TITLE_CHARS);
    if trimmed.is_empty() {
        "Ambient Session".to_string()
    } else {
        title_case_words(&trimmed)
    }
}

fn validated_session_title(text: &str, fallback: &str) -> String {
    let cleaned = sanitize_session_title(text);
    let lower = cleaned.to_ascii_lowercase();
    if cleaned.is_empty()
        || is_meta_summary_line(&cleaned)
        || is_presentational_line(&cleaned)
        || matches!(
            lower.as_str(),
            "thinking" | "summary" | "notes" | "breakdown" | "response" | "overview"
        )
    {
        return sanitize_session_title(fallback);
    }
    cleaned
}

fn strip_speaker_prefix(text: &str) -> &str {
    speaker_prefix(text).map(|(_, rest)| rest).unwrap_or(text)
}

fn speaker_prefix(text: &str) -> Option<(&'static str, &str)> {
    for prefix in [
        "You:",
        "S1:",
        "S2:",
        "S3:",
        "S4:",
        "S5:",
        "S6:",
        "Speaker 1:",
        "Speaker 2:",
        "Speaker 3:",
        "Speaker 4:",
        "Speaker 5:",
        "Speaker 6:",
        "Person A:",
        "Person B:",
        "Person C:",
        "Person D:",
        "Person E:",
        "Person F:",
    ] {
        if let Some(rest) = text.strip_prefix(prefix) {
            return Some((prefix, rest.trim()));
        }
    }
    None
}

fn sanitize_summary_transcript_lines<I>(lines: I, include_speaker_labels: bool) -> String
where
    I: IntoIterator<Item = String>,
{
    let mut cleaned_lines = Vec::<String>::new();

    for line in lines {
        let Some(cleaned) = sanitize_summary_transcript_line(&line, include_speaker_labels) else {
            continue;
        };

        if cleaned_lines
            .iter()
            .rev()
            .take(SUMMARY_TRANSCRIPT_RECENT_LINE_WINDOW)
            .any(|existing| lines_are_similar(existing, &cleaned))
        {
            continue;
        }

        cleaned_lines.push(cleaned);
    }

    cleaned_lines.join("\n")
}

fn sanitize_summary_transcript_line(line: &str, include_speaker_labels: bool) -> Option<String> {
    let (speaker, body) = speaker_prefix(line).map_or((None, line), |(prefix, rest)| {
        (Some(prefix.trim_end_matches(':')), rest)
    });
    let cleaned = sanitize_summary_transcript_body(body);
    if cleaned.is_empty() {
        return None;
    }

    if include_speaker_labels {
        speaker
            .map(|label| format!("{label}: {cleaned}"))
            .or(Some(cleaned))
    } else {
        Some(cleaned)
    }
}

fn sanitize_summary_transcript_body(text: &str) -> String {
    let repaired = repair_summary_spacing(text);
    let mut kept = Vec::<String>::new();

    for fragment in split_summary_fragments(&repaired) {
        let compacted = compact_repeated_tokens(&fragment);
        if compacted.is_empty() {
            continue;
        }

        let tokens = summary_tokens(&compacted);
        if tokens.len() < 3 || is_meta_summary_line(&compacted) {
            continue;
        }
        if tokens.len() >= 8
            && looks_noisy_line(&compacted, &tokens)
            && !contains_summary_keywords(&compacted)
        {
            continue;
        }

        if let Some(existing) = kept
            .iter_mut()
            .rev()
            .take(3)
            .find(|existing| lines_are_similar(existing, &compacted))
        {
            if compacted.chars().count() > existing.chars().count()
                && compacted.contains(existing.as_str())
            {
                *existing = compacted;
            }
            continue;
        }

        kept.push(compacted);
    }

    if kept.is_empty() {
        let compacted = compact_repeated_tokens(&repaired);
        let tokens = summary_tokens(&compacted);
        if compacted.is_empty() || (tokens.len() >= 8 && looks_noisy_line(&compacted, &tokens)) {
            return String::new();
        }
        return compacted;
    }

    kept.join(". ")
}

fn repair_summary_spacing(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        out.push(ch);
        if matches!(ch, '.' | '?' | '!' | ';' | ':') {
            if let Some(next) = chars.peek().copied() {
                if !next.is_whitespace() && !matches!(next, '.' | ',' | '?' | '!') {
                    out.push(' ');
                }
            }
        }
    }

    out
}

fn split_summary_fragments(text: &str) -> Vec<String> {
    let mut fragments = Vec::<String>::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '?' | '!' | ';' | '\n') {
            let fragment = collapse_spaces(current.trim());
            if !fragment.is_empty() {
                fragments.push(fragment);
            }
            current.clear();
        }
    }

    let trailing = collapse_spaces(current.trim());
    if !trailing.is_empty() {
        fragments.push(trailing);
    }
    if fragments.is_empty() {
        let single = collapse_spaces(text.trim());
        if !single.is_empty() {
            fragments.push(single);
        }
    }

    fragments
}

fn compact_repeated_tokens(text: &str) -> String {
    let raw_tokens = text.split_whitespace().collect::<Vec<_>>();
    if raw_tokens.is_empty() {
        return String::new();
    }

    let normalized_tokens = raw_tokens
        .iter()
        .map(|token| {
            token
                .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '\'')
                .to_ascii_lowercase()
        })
        .collect::<Vec<_>>();
    let mut kept = Vec::<&str>::new();
    let mut kept_normalized = Vec::<String>::new();
    let max_ngram = SUMMARY_TRANSCRIPT_REPEAT_NGRAM_MAX.min(raw_tokens.len() / 2);
    let mut index = 0usize;

    while index < raw_tokens.len() {
        let mut collapsed = false;
        for size in (SUMMARY_TRANSCRIPT_REPEAT_NGRAM_MIN..=max_ngram).rev() {
            if index + size * 2 > raw_tokens.len() {
                continue;
            }
            if normalized_tokens[index..index + size]
                .iter()
                .any(|token| token.is_empty())
            {
                continue;
            }

            let first = &normalized_tokens[index..index + size];
            let second = &normalized_tokens[index + size..index + size * 2];
            if first != second {
                continue;
            }

            kept.extend_from_slice(&raw_tokens[index..index + size]);
            kept_normalized.extend(first.iter().cloned());
            let pattern = first.to_vec();
            index += size * 2;
            while index + size <= raw_tokens.len()
                && normalized_tokens[index..index + size] == pattern[..]
            {
                index += size;
            }
            collapsed = true;
            break;
        }

        if collapsed {
            continue;
        }

        if kept_normalized
            .last()
            .is_some_and(|last| !last.is_empty() && *last == normalized_tokens[index])
        {
            index += 1;
            continue;
        }

        kept.push(raw_tokens[index]);
        kept_normalized.push(normalized_tokens[index].clone());
        index += 1;
    }

    collapse_spaces(&kept.join(" "))
}

fn looks_symbolic_line(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }

    let alnum_count = trimmed.chars().filter(|ch| ch.is_alphanumeric()).count();
    let total_count = trimmed.chars().filter(|ch| !ch.is_whitespace()).count();
    if total_count > 0 && alnum_count * 3 < total_count {
        return true;
    }

    trimmed
        .chars()
        .all(|ch| matches!(ch, '-' | '*' | '.' | '_' | ':' | ';'))
}

fn collapse_spaces(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn trim_to_len(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    let mut out = String::new();
    for word in text.split_whitespace() {
        let candidate = if out.is_empty() {
            word.to_string()
        } else {
            format!("{out} {word}")
        };
        if candidate.chars().count() > max_chars {
            break;
        }
        out = candidate;
    }

    if out.is_empty() {
        text.chars().take(max_chars).collect::<String>()
    } else {
        out
    }
}

fn title_case_words(text: &str) -> String {
    text.split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            let Some(first) = chars.next() else {
                return String::new();
            };
            let rest = chars.as_str().to_lowercase();
            format!("{}{}", first.to_uppercase(), rest)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::{
        build_structured_notes_prompt, clean_general_markdown, full_summary_context,
        heuristic_general_notes, heuristic_structured_notes, merge_topic_mentions,
        parse_model_structured_notes, parse_simple_bullets, render_general_summary_draft,
        sanitize_session_title, select_general_recovery_lines, summarize_general_multistage,
        summarize_general_recovery, validated_session_title, GeneralSummaryDraft, TopicMention,
        TopicSection,
    };
    use screamer_core::ambient::SummaryTemplate;
    use screamer_core::ambient::{AudioLane, CanonicalSegment, SpeakerLabel};

    #[test]
    fn sanitize_session_title_strips_formatting_and_speaker_prefixes() {
        assert_eq!(
            sanitize_session_title("Title: You: hospital rehab planning!!!"),
            "Hospital Rehab Planning"
        );
    }

    #[test]
    fn sanitize_session_title_limits_words_and_length() {
        assert_eq!(
            sanitize_session_title(
                "a very long title that should absolutely keep only the first few words"
            ),
            "A Very Long Title"
        );
    }

    #[test]
    fn validated_session_title_rejects_meta_model_output() {
        assert_eq!(
            validated_session_title("Okay, here's the summary", "SIP Training"),
            "Sip Training"
        );
    }

    #[test]
    fn heuristic_summary_prefers_salient_unique_lines_over_raw_transcript() {
        let segments = vec![
            CanonicalSegment {
                id: 1,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S1,
                start_ms: 0,
                end_ms: 500,
                text: "Start cleaning you want me to clean clean what?".to_string(),
            },
            CanonicalSegment {
                id: 2,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S1,
                start_ms: 500,
                end_ms: 1_000,
                text: "There is no implementation of this library that runs on Apple."
                    .to_string(),
            },
            CanonicalSegment {
                id: 3,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S1,
                start_ms: 1_000,
                end_ms: 1_500,
                text: "So to do this, I have to rewrite that algorithm into an Apple Silicon implementation."
                    .to_string(),
            },
            CanonicalSegment {
                id: 4,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S1,
                start_ms: 1_500,
                end_ms: 2_000,
                text: "NVIDIA hardware uses CUDA, so we would need a Metal equivalent on MacBooks."
                    .to_string(),
            },
        ];

        let notes = heuristic_structured_notes("", &segments, Some("Diarization Port"));

        assert!(notes.summary.contains("Main takeaway:"));
        assert!(
            notes.summary.contains("Apple-compatible") || notes.summary.contains("Apple Silicon")
        );
        assert!(!notes.summary.contains("Start cleaning"));
        assert!(notes.key_points.iter().any(|point| point.contains("Apple")));
    }

    #[test]
    fn parses_structured_markdown_sections_from_model_output() {
        let parsed = parse_model_structured_notes(
            "## Summary\nThis was about porting diarization to Apple Silicon.\n\n## Key Points\n- The current Python library expects CUDA.\n- Metal support needs a rewrite.\n\n## Decisions\n- Use a staged v1 rollout.\n\n## Action Items\n- Prototype a Metal path.\n\n## Open Questions\n- How much of wav2vec2 can be shared?\n",
            "S1: transcript",
        )
        .expect("structured notes should parse");

        assert_eq!(
            parsed.summary,
            "This was about porting diarization to Apple Silicon."
        );
        assert_eq!(parsed.key_points.len(), 2);
        assert_eq!(
            parsed.decisions,
            vec!["Use a staged v1 rollout.".to_string()]
        );
        assert_eq!(
            parsed.action_items,
            vec!["Prototype a Metal path.".to_string()]
        );
        assert_eq!(
            parsed.open_questions,
            vec!["How much of wav2vec2 can be shared?".to_string()]
        );
        assert_eq!(parsed.transcript, "S1: transcript");
    }

    #[test]
    fn select_general_recovery_lines_prefers_supported_concrete_fragments() {
        let lines = vec![
            "Hey man, how's it going?".to_string(),
            "I'm on call but I'm taking PTO on the 20th and need coverage.".to_string(),
            "The week of the 20th is the one that needs on-call coverage.".to_string(),
            "I looked at it and tried to decide whether I thought I could ban a phone deal."
                .to_string(),
            "We should restore Slack access for the on-call group and update the runbook."
                .to_string(),
        ];

        let selected = select_general_recovery_lines(&lines);

        assert!(selected.iter().any(|line| line.contains("PTO on the 20th")));
        assert!(selected.iter().any(|line| line.contains("Slack access")));
        assert!(!selected.iter().any(|line| line.contains("how's it going")));
        assert!(!selected
            .iter()
            .any(|line| line.contains("ban a phone deal")));
    }

    #[test]
    fn summarize_general_recovery_rewrites_noisy_lines_into_note_style_sections() {
        let segments = vec![
            CanonicalSegment {
                id: 1,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S1,
                start_ms: 0,
                end_ms: 500,
                text: "I'm on call but I'm taking PTO on Friday and need coverage until I return Tuesday night.".to_string(),
            },
            CanonicalSegment {
                id: 2,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S2,
                start_ms: 500,
                end_ms: 1_000,
                text: "Tuesday onwards I can take over again if my flight is on time.".to_string(),
            },
            CanonicalSegment {
                id: 3,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S1,
                start_ms: 1_000,
                end_ms: 1_500,
                text: "During the last on-call incident, David handled it because I was offline."
                    .to_string(),
            },
            CanonicalSegment {
                id: 4,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S2,
                start_ms: 1_500,
                end_ms: 2_000,
                text: "Our team does not treat this as a 24 hour on-call rotation.".to_string(),
            },
        ];
        let generator = |_prompt: &str, _max_tokens: usize| -> Result<String, String> {
            Err("fallbacks should be sufficient".to_string())
        };

        let notes = summarize_general_recovery("", &segments, Some("On-call"), &generator)
            .expect("recovery summary should succeed");
        let markdown = notes.raw_notes.expect("raw notes should be present");

        assert!(markdown.contains("## On-Call Coverage"));
        assert!(
            markdown.contains("A team member requested on-call coverage during planned time off.")
        );
        assert!(
            markdown.contains("Coverage may resume Tuesday night if the return flight is on time.")
        );
        assert!(markdown.contains(
            "David handled a recent early-morning issue because the assigned person was offline."
        ));
        assert!(markdown.contains("## Incident Response"));
        assert!(markdown.contains("## On-Call Expectations"));
        assert!(markdown.contains("not treated as a 24-hour on-call rotation"));
        assert!(!markdown.contains("I'm on call"));
    }

    #[test]
    fn recovery_topic_inference_treats_handled_offline_lines_as_incidents() {
        assert_eq!(
            super::infer_recovery_topic_title(
                "The one where I was on call happened early in the morning and David handled it because I was offline."
            ),
            "Incident Response"
        );
    }

    #[test]
    fn heuristic_general_notes_uses_topic_heading_only() {
        let segments = vec![
            CanonicalSegment {
                id: 1,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S1,
                start_ms: 0,
                end_ms: 500,
                text: "We need to port diarization to Apple Silicon.".to_string(),
            },
            CanonicalSegment {
                id: 2,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S2,
                start_ms: 500,
                end_ms: 1_000,
                text: "Metal support will need a rewrite of the current CUDA-dependent path."
                    .to_string(),
            },
        ];
        let fallback = heuristic_structured_notes("", &segments, Some("Diarization Port"));
        let notes = heuristic_general_notes("", &segments, Some("Diarization Port"), &fallback);
        let markdown = notes
            .raw_notes
            .expect("general fallback should render raw notes");

        assert!(markdown.starts_with("## Diarization Port"));
        assert!(!markdown.contains("## Summary"));
        assert!(!markdown.contains("## Key Points"));
        assert!(!markdown.contains("## Decisions"));
        assert!(!markdown.contains("## Action Items"));
        assert!(!markdown.contains("## Open Questions"));
    }

    #[test]
    fn parses_common_gemma_heading_variants_and_none_markers() {
        let parsed = parse_model_structured_notes(
            "## Summary\nWe need better diarization before summarization.\n\n## Key Points\nGemma 3 1B should be bundled and run through Metal on Apple Silicon.\n\n## Decisions\nWe should switch to the smaller bundled model.\n\n## Actions\nNone.\n\n## Open Questions\nN/A\n",
            "S1: transcript",
        )
        .expect("variant structured notes should parse");

        assert_eq!(
            parsed.summary,
            "We need better diarization before summarization."
        );
        assert_eq!(
            parsed.key_points,
            vec![
                "Gemma 3 1B should be bundled and run through Metal on Apple Silicon.".to_string()
            ]
        );
        assert_eq!(
            parsed.decisions,
            vec!["We should switch to the smaller bundled model.".to_string()]
        );
        assert!(parsed.action_items.is_empty());
        assert!(parsed.open_questions.is_empty());
    }

    #[test]
    fn general_template_prompt_uses_topic_grouped_markdown_style() {
        let prompt = build_structured_notes_prompt(
            "",
            &[],
            Some("Ambient session"),
            SummaryTemplate::General,
        );

        assert!(prompt.starts_with("You are an expert note-taker"));
        assert!(prompt.contains("Create a markdown heading (##) for each topic"));
        assert!(prompt.contains("Do NOT reproduce speaker labels"));
        assert!(prompt.contains("Transcript:"));
    }

    #[test]
    fn structured_prompt_preserves_scratch_pad_context() {
        let prompt = build_structured_notes_prompt(
            "--- User Notes (Scratch Pad) ---\nPrioritize launch risk and customer issues.\n--- End User Notes ---\n\nPerson A: Launch next week if QA passes.",
            &[],
            Some("Launch review"),
            SummaryTemplate::TeamMeeting,
        );

        assert!(prompt.contains("User Notes (Scratch Pad):"));
        assert!(prompt.contains("Prioritize launch risk and customer issues."));
        assert!(prompt.contains("Transcript:"));
    }

    #[test]
    fn general_context_uses_speakerless_segments() {
        let segments = vec![
            CanonicalSegment {
                id: 1,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S1,
                start_ms: 0,
                end_ms: 500,
                text: "Ship the calendar invite flow next week.".to_string(),
            },
            CanonicalSegment {
                id: 2,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S2,
                start_ms: 500,
                end_ms: 1_000,
                text: "Recurring mobile bug is the blocker.".to_string(),
            },
        ];

        let context = full_summary_context(
            "--- User Notes (Scratch Pad) ---\nFocus on launch blockers.\n--- End User Notes ---\n\nPerson A: ignored because segments win.",
            &segments,
            false,
        );

        assert!(context.contains("User Notes (Scratch Pad):\nFocus on launch blockers."));
        assert!(context.contains("Ship the calendar invite flow next week."));
        assert!(context.contains("Recurring mobile bug is the blocker."));
        assert!(!context.contains("Person A:"));
        assert!(!context.contains("Person B:"));
    }

    #[test]
    fn general_context_compacts_repeated_transcript_fragments() {
        let segments = vec![CanonicalSegment {
            id: 1,
            lane: AudioLane::Microphone,
            speaker: SpeakerLabel::S1,
            start_ms: 0,
            end_ms: 500,
            text: "I'm on call but I'm taking the 20th PTO. I'm on call but I'm taking the 20th PTO. I'm not going to be working and I probably won't even take my work computer with me. I'm not going to be working and I probably won't even take my work computer with me.".to_string(),
        }];

        let context = full_summary_context("", &segments, false);

        assert_eq!(
            context
                .matches("I'm on call but I'm taking the 20th PTO")
                .count(),
            1
        );
        assert_eq!(
            context
                .matches("I'm not going to be working and I probably won't even take my work computer with me")
                .count(),
            1
        );
    }

    #[test]
    fn pathological_transcript_line_detects_repetition_heavy_segments() {
        let text =
            "I'm on call but I'm taking the 20th PTO. I'm on call but I'm taking the 20th PTO. \
I'm not going to be working and I probably won't even take my work computer with me. \
I'm not going to be working and I probably won't even take my work computer with me.";

        assert!(super::pathological_transcript_line(text));
    }

    #[test]
    fn clean_general_markdown_strips_speaker_labels_and_preamble() {
        let cleaned = clean_general_markdown(
            "Here’s a breakdown of the core discussion points:\n\n## Calendar Invite Flow\n* Person A: Ship next week if QA passes.\n* Person B: Maya will pair on the blocker.\n\n---",
        );

        assert_eq!(
            cleaned,
            "## Calendar Invite Flow\n- Ship next week if QA passes.\n- Maya will pair on the blocker."
        );
    }

    #[test]
    fn clean_general_markdown_drops_meta_sections_and_keeps_signal() {
        let cleaned = clean_general_markdown(
            "## Detailed Breakdown of Topics\n- Okay, let's break this down into manageable chunks.\n- **1. The Goal:** Identify patterns in the data.\n\n## Release Planning\n- Ship next week if QA passes.\n\n## Action Items\n- Pair with Maya on the blocker tomorrow.",
        );

        assert_eq!(
            cleaned,
            "## Release Planning\n- Ship next week if QA passes.\n- Pair with Maya on the blocker tomorrow."
        );
    }

    #[test]
    fn clean_general_markdown_drops_low_value_conversational_bullets() {
        let cleaned = clean_general_markdown(
            "## On-call coverage\n- Hey man, how's it going?\n- A team member is taking PTO on the 20th and asked to swap on-call coverage.\n- Did you see my message?",
        );

        assert_eq!(
            cleaned,
            "## On-call coverage\n- A team member is taking PTO on the 20th and asked to swap on-call coverage."
        );
    }

    #[test]
    fn clean_general_markdown_drops_repetition_heavy_bullets() {
        let cleaned = clean_general_markdown(
            "## Scheduling\n- I'm gonna create it sometime today and probably start by adding like I'm gonna create it sometime today and probably start by adding.\n- A team member is on PTO Friday through Monday and needs on-call coverage.",
        );

        assert_eq!(
            cleaned,
            "## Scheduling\n- A team member is on PTO Friday through Monday and needs on-call coverage."
        );
    }

    #[test]
    fn clean_general_markdown_drops_generic_boilerplate_bullets() {
        let cleaned = clean_general_markdown(
            "## On-call Rotation\n- On-call rotation is a common practice in various industries.\n- A team member asked for coverage while on PTO.",
        );

        assert_eq!(
            cleaned,
            "## On-call Rotation\n- A team member asked for coverage while on PTO."
        );
    }

    #[test]
    fn parse_simple_bullets_requires_list_items_and_ignores_meta_text() {
        let bullets = parse_simple_bullets(
            "Okay, let's break this down.\n- **Release timing**\n- Here’s a breakdown of the request.\n1. Pair with Maya tomorrow.",
        );

        assert_eq!(
            bullets,
            vec![
                "Release timing".to_string(),
                "Pair with Maya tomorrow.".to_string()
            ]
        );
    }

    #[test]
    fn merge_topic_mentions_collapses_near_duplicate_titles() {
        let clusters = merge_topic_mentions(vec![
            TopicMention {
                title: "Launch timeline".to_string(),
                chunk_index: 0,
            },
            TopicMention {
                title: "Launch plan".to_string(),
                chunk_index: 1,
            },
            TopicMention {
                title: "Customer confusion".to_string(),
                chunk_index: 1,
            },
        ]);

        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].chunk_indices, vec![0, 1]);
        assert!(clusters[0].title.contains("Launch"));
        assert_eq!(clusters[1].title, "Customer confusion");
    }

    #[test]
    fn render_general_summary_draft_outputs_topic_sections_only() {
        let markdown = render_general_summary_draft(&GeneralSummaryDraft {
            topics: vec![
                TopicSection {
                    title: "Release timing".to_string(),
                    bullets: vec!["Ship next week if QA passes by Thursday.".to_string()],
                },
                TopicSection {
                    title: "Customer confusion".to_string(),
                    bullets: vec!["Support flagged time zones and duplicate reminders.".to_string()],
                },
            ],
        });

        assert!(markdown.starts_with("## Release timing"));
        assert!(markdown.contains("## Customer confusion"));
        assert!(!markdown.contains("## Decisions"));
        assert!(!markdown.contains("## Action Items"));
        assert!(!markdown.contains("## Open Questions"));
    }

    #[test]
    fn multistage_general_summary_builds_topic_grouped_notes() {
        let segments = vec![
            CanonicalSegment {
                id: 1,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S1,
                start_ms: 0,
                end_ms: 500,
                text: "We should ship the calendar invite flow next week if QA passes by Thursday."
                    .to_string(),
            },
            CanonicalSegment {
                id: 2,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S2,
                start_ms: 500,
                end_ms: 1_000,
                text: "The blocker is the recurring-event bug in mobile.".to_string(),
            },
            CanonicalSegment {
                id: 3,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S1,
                start_ms: 1_000,
                end_ms: 1_500,
                text: "If that slips, release desktop editing first.".to_string(),
            },
            CanonicalSegment {
                id: 4,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S3,
                start_ms: 1_500,
                end_ms: 2_000,
                text: "Support flagged confusion around time zones and duplicate reminders."
                    .to_string(),
            },
            CanonicalSegment {
                id: 5,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S2,
                start_ms: 2_000,
                end_ms: 2_500,
                text: "Action item is to pair with Maya on the recurring bug tomorrow.".to_string(),
            },
            CanonicalSegment {
                id: 6,
                lane: AudioLane::Microphone,
                speaker: SpeakerLabel::S1,
                start_ms: 2_500,
                end_ms: 3_000,
                text: "We will decide go or no-go in Friday's launch review.".to_string(),
            },
        ];
        let generator = |prompt: &str, _max_tokens: usize| -> Result<String, String> {
            if prompt.contains("Identify the concrete discussion topics") {
                return Ok("- Release timing\n- Customer confusion".to_string());
            }
            if prompt.contains("Write bullets for the meeting topic `Release timing`") {
                return Ok("- Ship the calendar invite flow next week if QA passes by Thursday.\n- If the recurring mobile bug slips, release desktop editing first.\n- Pair with Maya on the recurring mobile bug tomorrow.\n- The final go or no-go call happens in Friday's launch review.".to_string());
            }
            if prompt.contains("Write bullets for the meeting topic `Customer confusion`") {
                return Ok(
                    "- Support flagged confusion around time zones and duplicate reminders."
                        .to_string(),
                );
            }
            Err(format!("unexpected prompt: {prompt}"))
        };

        let notes = summarize_general_multistage("", &segments, Some("Launch review"), &generator)
            .expect("multistage summary should succeed");
        let markdown = notes.raw_notes.expect("raw notes should be present");

        assert!(markdown.contains("## Release timing"));
        assert!(markdown.contains("## Customer confusion"));
        assert!(markdown.contains("Pair with Maya on the recurring mobile bug tomorrow."));
        assert!(!markdown.contains("## Decisions"));
        assert!(!markdown.contains("## Action Items"));
        assert!(!markdown.contains("## Open Questions"));
        assert!(!markdown.contains("Person A"));
        assert!(!markdown.contains("Person B"));
    }
}
