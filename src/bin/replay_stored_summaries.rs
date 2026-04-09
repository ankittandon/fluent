#[path = "../bundled_llm.rs"]
mod bundled_llm;
#[path = "../config.rs"]
mod config;
#[path = "../session_store.rs"]
mod session_store;
#[path = "../summary_backend.rs"]
mod summary_backend;

use config::Config;
use screamer_core::ambient::{segments_to_transcript, AmbientSessionState};
use session_store::SessionStore;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use summary_backend::SummaryBackendRegistry;

fn main() {
    if let Err(err) = run(std::env::args().skip(1).collect()) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let (session_filter, limit) = parse_args(&args)?;
    let config = Config::load();
    let store = SessionStore::open_default()?;
    let registry = SummaryBackendRegistry::detect();
    let summarizer = registry.summarizer_for_config(&config);

    let sessions = store.list_recent_sessions(limit.unwrap_or(100), None)?;
    let out_dir = default_output_dir()?;
    fs::create_dir_all(&out_dir)
        .map_err(|err| format!("Failed to create replay output directory: {err}"))?;

    let mut report = String::new();
    report.push_str(&format!("output_dir\t{}\n", out_dir.display()));
    report.push_str(
        "id\told_title\tnew_title\told_chars\tnew_chars\told_meta_hits\tnew_meta_hits\tstatus\n",
    );

    for session in sessions {
        if session.state != AmbientSessionState::Completed {
            continue;
        }
        if let Some(filter_id) = session_filter {
            if session.id != filter_id {
                continue;
            }
        }

        let Some(detail) = store.load_session(session.id)? else {
            continue;
        };

        let cleaned_segments = screamer_core::ambient::clean_canonical_segments(&detail.segments);
        let cleaned_live_notes = if cleaned_segments.is_empty() {
            detail.live_notes.clone()
        } else {
            segments_to_transcript(&cleaned_segments)
        };

        let title_hint =
            registry.concise_session_title(&config, &cleaned_live_notes, &cleaned_segments);
        let notes_with_scratch = if detail.scratch_pad.trim().is_empty() {
            cleaned_live_notes.clone()
        } else {
            format!(
                "--- User Notes (Scratch Pad) ---\n{}\n--- End User Notes ---\n\n{}",
                detail.scratch_pad, cleaned_live_notes
            )
        };

        let status;
        let (new_title, new_summary) = match summarizer.summarize(
            &notes_with_scratch,
            &cleaned_segments,
            Some(&title_hint),
            detail.summary_template,
        ) {
            Ok(notes) => {
                let summary = notes.to_markdown();
                let title = registry.title_from_summary(
                    &config,
                    &summary,
                    &cleaned_live_notes,
                    &cleaned_segments,
                );
                status = "ok";
                (title, summary)
            }
            Err(err) => {
                status = "error";
                (detail.title.clone(), format!("## Replay Error\n\n- {err}"))
            }
        };

        let session_dir = out_dir.join(format!("{:03}_{}", session.id, slug(&detail.title)));
        fs::create_dir_all(&session_dir)
            .map_err(|err| format!("Failed to create session replay directory: {err}"))?;
        fs::write(session_dir.join("old_summary.md"), &detail.structured_notes)
            .map_err(|err| format!("Failed to write old summary: {err}"))?;
        fs::write(session_dir.join("new_summary.md"), &new_summary)
            .map_err(|err| format!("Failed to write new summary: {err}"))?;
        fs::write(
            session_dir.join("transcript.md"),
            &detail.transcript_markdown,
        )
        .map_err(|err| format!("Failed to write transcript: {err}"))?;
        fs::write(
            session_dir.join("cleaned_transcript.md"),
            &cleaned_live_notes,
        )
        .map_err(|err| format!("Failed to write cleaned transcript: {err}"))?;

        report.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            detail.id,
            sanitize_tsv(&detail.title),
            sanitize_tsv(&new_title),
            detail.structured_notes.chars().count(),
            new_summary.chars().count(),
            meta_hit_count(&detail.structured_notes),
            meta_hit_count(&new_summary),
            status,
        ));
    }

    let report_path = out_dir.join("report.tsv");
    fs::write(&report_path, report).map_err(|err| format!("Failed to write report: {err}"))?;
    println!("{}", report_path.display());
    Ok(())
}

fn parse_args(args: &[String]) -> Result<(Option<i64>, Option<usize>), String> {
    let mut session_id = None;
    let mut limit = None;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--session" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--session requires a numeric id".to_string())?;
                session_id = Some(
                    value
                        .parse::<i64>()
                        .map_err(|_| format!("Invalid session id: {value}"))?,
                );
                index += 2;
            }
            "--limit" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--limit requires a numeric value".to_string())?;
                limit = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| format!("Invalid limit: {value}"))?,
                );
                index += 2;
            }
            other => {
                return Err(format!(
                    "Unknown argument: {other}. Supported flags: --session <id>, --limit <n>"
                ));
            }
        }
    }

    Ok((session_id, limit))
}

fn default_output_dir() -> Result<PathBuf, String> {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("Clock error: {err}"))?
        .as_secs();
    Ok(std::env::temp_dir().join(format!("screamer-summary-replay-{ts}")))
}

fn slug(text: &str) -> String {
    let lowered = text.to_ascii_lowercase();
    let mut out = String::new();
    let mut last_was_sep = false;
    for ch in lowered.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }
    out.trim_matches('_').chars().take(40).collect()
}

fn sanitize_tsv(text: &str) -> String {
    text.replace('\t', " ").replace('\n', " ")
}

fn meta_hit_count(text: &str) -> usize {
    let lower = text.to_ascii_lowercase();
    [
        "let's break this down",
        "lets break this down",
        "here's a breakdown",
        "here’s a breakdown",
        "here's a response",
        "here’s a response",
        "overall theme",
        "response to your request",
        "organized into bullet points",
        "what i think is happening",
        "detailed breakdown",
    ]
    .iter()
    .filter(|needle| lower.contains(**needle))
    .count()
}
