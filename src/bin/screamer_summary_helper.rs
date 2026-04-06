use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::{LlamaModelParams, LlamaSplitMode};
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::TokenToStringError;
use screamer_models::{find_summary_model, DEFAULT_BUNDLED_SUMMARY_MODEL_ID};
use std::io::{self, Read};
use std::num::NonZeroU32;

const DEFAULT_SESSION_CONTEXT_TOKENS: u32 = 8_192;
const DEFAULT_BATCH_TOKENS: usize = 512;

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let max_tokens = parse_max_tokens(std::env::args().skip(1))?;
    let mut prompt = String::new();
    io::stdin()
        .read_to_string(&mut prompt)
        .map_err(|err| format!("Failed to read prompt from stdin: {err}"))?;
    if prompt.trim().is_empty() {
        return Err("Summary helper received an empty prompt.".to_string());
    }

    let content = generate_summary(prompt.trim(), max_tokens)?;
    print!("{content}");
    Ok(())
}

fn parse_max_tokens(mut args: impl Iterator<Item = String>) -> Result<usize, String> {
    while let Some(arg) = args.next() {
        if arg == "--max-tokens" {
            let value = args
                .next()
                .ok_or_else(|| "Missing value for --max-tokens".to_string())?;
            return value
                .parse::<usize>()
                .map_err(|err| format!("Invalid --max-tokens value `{value}`: {err}"));
        }
    }

    Ok(384)
}

fn recommended_thread_count() -> u32 {
    let available = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(4);

    available.saturating_sub(1).clamp(2, 8) as u32
}

fn generate_summary(prompt: &str, max_tokens: usize) -> Result<String, String> {
    let Some(model_path) = find_summary_model(DEFAULT_BUNDLED_SUMMARY_MODEL_ID) else {
        return Err(
            "Bundled Gemma summary model not found. Expected models/summary/gemma-3-1b-it-q4_k_m.gguf."
                .to_string(),
        );
    };

    let mut backend =
        LlamaBackend::init().map_err(|err| format!("Failed to init llama backend: {err}"))?;
    backend.void_logs();

    let gpu_layers = if backend.supports_gpu_offload() {
        u32::MAX
    } else {
        0
    };
    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(gpu_layers)
        .with_split_mode(LlamaSplitMode::None)
        .with_main_gpu(0)
        .with_use_mmap(backend.supports_mmap())
        .with_use_mlock(false);
    let model =
        LlamaModel::load_from_file(&backend, &model_path, &model_params).map_err(|err| {
            format!(
                "Failed to load bundled Gemma model at {}: {err}",
                model_path.display()
            )
        })?;

    let rendered_prompt = render_prompt(&model, prompt)?;
    let prompt_tokens = model
        .str_to_token(&rendered_prompt, AddBos::Never)
        .map_err(|err| format!("Failed to tokenize summary prompt: {err}"))?;

    let threads = recommended_thread_count() as i32;
    let context_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(DEFAULT_SESSION_CONTEXT_TOKENS))
        .with_n_batch(DEFAULT_BATCH_TOKENS as u32)
        .with_n_ubatch(DEFAULT_BATCH_TOKENS as u32)
        .with_n_threads(threads)
        .with_n_threads_batch(threads)
        .with_offload_kqv(true);
    let mut context = model
        .new_context(&backend, context_params)
        .map_err(|err| format!("Failed to create Gemma context: {err}"))?;

    decode_prompt(&mut context, &prompt_tokens)?;

    let mut generated = String::new();
    let mut position = prompt_tokens.len() as i32;
    let mut token_batch = LlamaBatch::new(1, 1);
    for _ in 0..max_tokens {
        let next_token = context.token_data_array().sample_token_greedy();
        if model.is_eog_token(next_token) || next_token == model.token_eos() {
            break;
        }

        generated.push_str(&decode_token_piece(&model, next_token)?);

        token_batch.clear();
        token_batch
            .add(next_token, position, &[0], true)
            .map_err(|err| format!("Failed to append generated token to batch: {err}"))?;
        context
            .decode(&mut token_batch)
            .map_err(|err| format!("Failed to decode generated token: {err}"))?;
        position += 1;
    }

    Ok(generated.trim().to_string())
}

fn render_prompt(model: &LlamaModel, prompt: &str) -> Result<String, String> {
    let message = LlamaChatMessage::new("user".to_string(), prompt.to_string())
        .map_err(|err| format!("Failed to build Gemma chat message: {err}"))?;

    let template = model
        .chat_template(None)
        .map_err(|err| format!("Failed to load Gemma chat template: {err}"))?;
    model
        .apply_chat_template(&template, &[message], true)
        .map_err(|err| format!("Failed to apply Gemma chat template: {err}"))
}

fn decode_prompt(
    context: &mut llama_cpp_2::context::LlamaContext<'_>,
    tokens: &[LlamaToken],
) -> Result<(), String> {
    if tokens.is_empty() {
        return Err("Gemma prompt tokenization returned no tokens.".to_string());
    }

    let mut batch = LlamaBatch::new(DEFAULT_BATCH_TOKENS, 1);
    let total = tokens.len();
    let mut processed = 0usize;

    while processed < total {
        batch.clear();
        for (offset, token) in tokens[processed..total.min(processed + DEFAULT_BATCH_TOKENS)]
            .iter()
            .enumerate()
        {
            let absolute = processed + offset;
            let is_last = absolute + 1 == total;
            batch
                .add(*token, absolute as i32, &[0], is_last)
                .map_err(|err| format!("Failed to add prompt token to batch: {err}"))?;
        }

        context
            .decode(&mut batch)
            .map_err(|err| format!("Failed to decode prompt batch: {err}"))?;
        processed += DEFAULT_BATCH_TOKENS;
    }

    Ok(())
}

fn decode_token_piece(model: &LlamaModel, token: LlamaToken) -> Result<String, String> {
    let mut buffer_size = 8usize;
    loop {
        match model.token_to_piece_bytes(token, buffer_size, false, None) {
            Ok(bytes) => return Ok(String::from_utf8_lossy(&bytes).to_string()),
            Err(TokenToStringError::InsufficientBufferSpace(needed)) => {
                buffer_size = usize::try_from((-needed).max(8)).unwrap_or(32);
            }
            Err(err) => return Err(format!("Failed to decode generated token: {err}")),
        }
    }
}
