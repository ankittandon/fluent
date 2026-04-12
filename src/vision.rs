use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const HELPER_BINARY_NAME: &str = "screamer_vision_helper";
const VISION_MAX_TOKENS: &str = "192";
const MEDIA_MARKER: &str = "<__media__>";
const SCREEN_HELPER_SYSTEM_PROMPT: &str = "\
You are Screamer's screen buddy. The screenshot is from a product, website, \
or web app, and the user is asking how to use it. Be concise because your \
answer will be spoken aloud. Answer in at most three short sentences. Give \
the exact next step when you can. Do not use markdown, bullets, lists, or \
long explanations. Start with the answer itself; do not prefix it with \
\"Answer:\" or \"This is the answer.\" If the screenshot is unclear, say what \
you can see and ask one short follow-up question.";

/// Run a vision query: pass the user's transcribed question + screenshot to the
/// multimodal Gemma 3 4B model and return the response.
pub fn ask_about_screen(prompt: &str, screenshot_path: &Path) -> Result<String, String> {
    let helper_path = find_helper_path()
        .ok_or_else(|| format!("Vision helper not found: {HELPER_BINARY_NAME}"))?;
    let model_prompt = build_screen_helper_prompt(prompt);

    let screenshot_str = screenshot_path
        .to_str()
        .ok_or("Screenshot path contains invalid UTF-8")?;

    let mut child = Command::new(&helper_path)
        .arg("--image")
        .arg(screenshot_str)
        .arg("--max-tokens")
        .arg(VISION_MAX_TOKENS)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| {
            format!(
                "Failed to launch vision helper at {}: {err}",
                helper_path.display()
            )
        })?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(model_prompt.as_bytes())
            .map_err(|err| format!("Failed to send prompt to vision helper: {err}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|err| format!("Failed to wait for vision helper: {err}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            format!(
                "Vision helper exited with status {}",
                output
                    .status
                    .code()
                    .map(|code| code.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            )
        } else {
            stderr
        });
    }

    Ok(clean_model_response(&String::from_utf8_lossy(
        &output.stdout,
    )))
}

fn build_screen_helper_prompt(user_question: &str) -> String {
    format!(
        "{SCREEN_HELPER_SYSTEM_PROMPT}\n\nUser question: {}",
        user_question.trim()
    )
}

fn clean_model_response(response: &str) -> String {
    let mut normalized = response
        .lines()
        .map(str::trim)
        .filter(|line| {
            !line.is_empty()
                && *line != "<start_of_image>"
                && *line != "<end_of_image>"
                && *line != MEDIA_MARKER
        })
        .collect::<Vec<_>>()
        .join(" ");

    while normalized.contains("  ") {
        normalized = normalized.replace("  ", " ");
    }

    strip_answer_prefix(&normalized).to_string()
}

fn strip_answer_prefix(text: &str) -> &str {
    let trimmed = text.trim();
    let lower = trimmed.to_ascii_lowercase();
    for prefix in [
        "answer:",
        "final answer:",
        "the answer is:",
        "this is the answer:",
        "here's the answer:",
        "here is the answer:",
    ] {
        if lower.starts_with(prefix) {
            return trimmed[prefix.len()..].trim_start();
        }
    }
    trimmed
}

fn find_helper_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let mut candidates = Vec::new();

    if let Some(parent) = exe.parent() {
        candidates.push(parent.join(HELPER_BINARY_NAME));
        if parent.ends_with("deps") {
            if let Some(debug_or_release_dir) = parent.parent() {
                candidates.push(debug_or_release_dir.join(HELPER_BINARY_NAME));
            }
        }
    }

    // Check inside .app bundle
    if let Some(parent) = exe.parent() {
        if parent.file_name().map(|n| n == "MacOS").unwrap_or(false) {
            candidates.push(parent.join(HELPER_BINARY_NAME));
        }
    }

    candidates.into_iter().find(|candidate| candidate.is_file())
}

#[cfg(test)]
mod tests {
    use super::{build_screen_helper_prompt, clean_model_response};

    #[test]
    fn screen_helper_prompt_keeps_spoken_answers_short() {
        let prompt = build_screen_helper_prompt("How do I change this setting?");

        assert!(prompt.contains("spoken aloud"));
        assert!(prompt.contains("at most three short sentences"));
        assert!(prompt.contains("do not prefix"));
        assert!(prompt.contains("User question: How do I change this setting?"));
    }

    #[test]
    fn cleans_image_markers_and_answer_prefix_for_tts() {
        let response = "<start_of_image>\nAnswer: Click Share, then Copy Link.";

        assert_eq!(
            clean_model_response(response),
            "Click Share, then Copy Link."
        );
    }
}
