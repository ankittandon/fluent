use kokoro_micro::TtsEngine;
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Read, Write};

const DEFAULT_VOICE: &str = "af_sky";
const DEFAULT_SPEED: f32 = 1.08;
const DEFAULT_GAIN: f32 = 1.0;
const DEFAULT_LANG: &str = "en";

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    configure_onnx_runtime_dylib();
    if args.server {
        return run_server(args);
    }

    let mut text = String::new();
    io::stdin()
        .read_to_string(&mut text)
        .map_err(|err| format!("Failed to read TTS text from stdin: {err}"))?;

    let output_path = args
        .output_path
        .as_deref()
        .ok_or("Missing required --output argument")?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| format!("Failed to create TTS runtime: {err}"))?;
    let mut engine =
        runtime.block_on(TtsEngine::with_paths(&args.model_path, &args.voices_path))?;

    synthesize_to_file(
        &mut engine,
        text.trim(),
        &args.voice,
        args.speed,
        args.gain,
        &args.lang,
        output_path,
    )
}

fn configure_onnx_runtime_dylib() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }

    for candidate in onnx_runtime_dylib_candidates() {
        if candidate.exists() {
            std::env::set_var("ORT_DYLIB_PATH", candidate);
            return;
        }
    }
}

fn onnx_runtime_dylib_candidates() -> Vec<std::path::PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("libonnxruntime.dylib"));
        }
    }
    candidates.push(
        std::path::PathBuf::from("models")
            .join("tts")
            .join("onnxruntime")
            .join("libonnxruntime.dylib"),
    );
    candidates
}

fn run_server(args: Args) -> Result<(), String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| format!("Failed to create TTS runtime: {err}"))?;
    let mut engine =
        runtime.block_on(TtsEngine::with_paths(&args.model_path, &args.voices_path))?;

    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line.map_err(|err| format!("Failed to read TTS request: {err}"))?;
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<TtsRequest>(&line) {
            Ok(request) => handle_request(&mut engine, request),
            Err(err) => TtsResponse {
                id: 0,
                ok: false,
                error: Some(format!("Invalid TTS request: {err}")),
            },
        };

        serde_json::to_writer(&mut stdout, &response)
            .map_err(|err| format!("Failed to write TTS response: {err}"))?;
        stdout
            .write_all(b"\n")
            .map_err(|err| format!("Failed to finish TTS response: {err}"))?;
        stdout
            .flush()
            .map_err(|err| format!("Failed to flush TTS response: {err}"))?;
    }

    Ok(())
}

fn handle_request(engine: &mut TtsEngine, request: TtsRequest) -> TtsResponse {
    let result = synthesize_to_file(
        engine,
        request.text.trim(),
        request.voice.as_deref().unwrap_or(DEFAULT_VOICE),
        request.speed.unwrap_or(DEFAULT_SPEED),
        request.gain.unwrap_or(DEFAULT_GAIN),
        request.lang.as_deref().unwrap_or(DEFAULT_LANG),
        &request.output,
    );

    match result {
        Ok(()) => TtsResponse {
            id: request.id,
            ok: true,
            error: None,
        },
        Err(err) => TtsResponse {
            id: request.id,
            ok: false,
            error: Some(err),
        },
    }
}

fn synthesize_to_file(
    engine: &mut TtsEngine,
    text: &str,
    voice: &str,
    speed: f32,
    gain: f32,
    lang: &str,
    output_path: &str,
) -> Result<(), String> {
    if text.is_empty() {
        return Err("TTS request contained empty text.".to_string());
    }

    let audio = engine.synthesize_with_options(text, Some(voice), speed, gain, Some(lang))?;
    if audio.is_empty() {
        return Err("TTS synthesis returned no audio.".to_string());
    }

    engine.save_wav(output_path, &audio)
}

#[derive(Deserialize)]
struct TtsRequest {
    id: u64,
    text: String,
    output: String,
    voice: Option<String>,
    speed: Option<f32>,
    gain: Option<f32>,
    lang: Option<String>,
}

#[derive(Serialize)]
struct TtsResponse {
    id: u64,
    ok: bool,
    error: Option<String>,
}

struct Args {
    model_path: String,
    voices_path: String,
    output_path: Option<String>,
    voice: String,
    speed: f32,
    gain: f32,
    lang: String,
    server: bool,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut model_path = None;
        let mut voices_path = None;
        let mut output_path = None;
        let mut voice = DEFAULT_VOICE.to_string();
        let mut speed = DEFAULT_SPEED;
        let mut gain = DEFAULT_GAIN;
        let mut lang = DEFAULT_LANG.to_string();
        let mut server = false;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--server" => {
                    server = true;
                }
                "--model" => {
                    model_path = Some(
                        args.next()
                            .ok_or_else(|| "Missing value for --model".to_string())?,
                    );
                }
                "--voices" => {
                    voices_path = Some(
                        args.next()
                            .ok_or_else(|| "Missing value for --voices".to_string())?,
                    );
                }
                "--output" => {
                    output_path = Some(
                        args.next()
                            .ok_or_else(|| "Missing value for --output".to_string())?,
                    );
                }
                "--voice" => {
                    voice = args
                        .next()
                        .ok_or_else(|| "Missing value for --voice".to_string())?;
                }
                "--speed" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "Missing value for --speed".to_string())?;
                    speed = value
                        .parse()
                        .map_err(|err| format!("Invalid --speed value `{value}`: {err}"))?;
                }
                "--gain" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "Missing value for --gain".to_string())?;
                    gain = value
                        .parse()
                        .map_err(|err| format!("Invalid --gain value `{value}`: {err}"))?;
                }
                "--lang" => {
                    lang = args
                        .next()
                        .ok_or_else(|| "Missing value for --lang".to_string())?;
                }
                other => return Err(format!("Unknown argument: {other}")),
            }
        }

        Ok(Self {
            model_path: model_path.ok_or("Missing required --model argument")?,
            voices_path: voices_path.ok_or("Missing required --voices argument")?,
            output_path,
            voice,
            speed,
            gain,
            lang,
            server,
        })
    }
}
