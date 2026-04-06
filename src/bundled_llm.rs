use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const HELPER_BINARY_NAME: &str = "screamer_summary_helper";

pub fn generate_bundled_summary(prompt: &str, max_tokens: usize) -> Result<String, String> {
    let helper_path = bundled_helper_path()
        .ok_or_else(|| format!("Bundled summary helper not found: {HELPER_BINARY_NAME}"))?;

    let mut child = Command::new(&helper_path)
        .arg("--max-tokens")
        .arg(max_tokens.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| {
            format!(
                "Failed to launch bundled summary helper at {}: {err}",
                helper_path.display()
            )
        })?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(prompt.as_bytes())
            .map_err(|err| format!("Failed to send prompt to summary helper: {err}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|err| format!("Failed to wait for summary helper: {err}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            format!(
                "Summary helper exited with status {}",
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

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn bundled_helper_path() -> Option<PathBuf> {
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

    if let Some(bundle_macos_dir) = bundle_macos_dir_from_exe(&exe) {
        candidates.push(bundle_macos_dir.join(HELPER_BINARY_NAME));
    }

    candidates.into_iter().find(|candidate| candidate.is_file())
}

fn bundle_macos_dir_from_exe(exe: &Path) -> Option<PathBuf> {
    let parent = exe.parent()?;
    (parent.file_name()? == "MacOS").then(|| parent.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::bundle_macos_dir_from_exe;
    use std::path::Path;

    #[test]
    fn detects_macos_bundle_directory_from_app_binary() {
        let path = Path::new("/tmp/Screamer.app/Contents/MacOS/Screamer");
        let macos_dir = bundle_macos_dir_from_exe(path).expect("should resolve bundle macos dir");
        assert_eq!(macos_dir, Path::new("/tmp/Screamer.app/Contents/MacOS"));
    }
}
