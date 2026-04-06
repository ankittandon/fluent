const SPEECH_RECOGNITION_CTOR_NAMES = ['SpeechRecognition', 'webkitSpeechRecognition'];

function resolveSpeechRecognitionCtor() {
  if (typeof window === 'undefined') {
    return null;
  }

  for (const name of SPEECH_RECOGNITION_CTOR_NAMES) {
    if (typeof window[name] === 'function') {
      return {
        ctor: window[name],
        name,
      };
    }
  }

  return null;
}

function createEmitter() {
  const listeners = new Map();

  return {
    emit(event, payload) {
      const handlers = listeners.get(event);
      if (!handlers) {
        return;
      }

      for (const handler of [...handlers]) {
        handler(payload);
      }
    },
    on(event, handler) {
      let handlers = listeners.get(event);
      if (!handlers) {
        handlers = new Set();
        listeners.set(event, handlers);
      }
      handlers.add(handler);

      return () => {
        handlers.delete(handler);
        if (handlers.size === 0) {
          listeners.delete(event);
        }
      };
    },
    clear() {
      listeners.clear();
    },
  };
}

function buildUnsupportedError() {
  return new Error(
    'Screamer web is not supported in this browser because SpeechRecognition is unavailable.'
  );
}

export function normalizeTranscript(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s']/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function getScreamerBrowserSupport() {
  const resolved = resolveSpeechRecognitionCtor();
  return {
    isSupported: Boolean(resolved),
    recognitionCtorName: resolved?.name ?? null,
  };
}

export function createCommandMatcher(commands, options = {}) {
  const minimumScore = options.minimumScore ?? 0.75;
  const compiled = (commands || []).map((command) => ({
    ...command,
    normalizedPhrases: (command.phrases || []).map((phrase) => ({
      raw: phrase,
      normalized: normalizeTranscript(phrase),
    })),
  }));

  return (input) => {
    const normalizedInput = normalizeTranscript(input);
    if (!normalizedInput) {
      return null;
    }

    let bestMatch = null;

    for (const command of compiled) {
      for (const phrase of command.normalizedPhrases) {
        if (!phrase.normalized) {
          continue;
        }

        let score = 0;
        if (normalizedInput === phrase.normalized) {
          score = 1;
        } else if (normalizedInput.startsWith(`${phrase.normalized} `) || normalizedInput.endsWith(` ${phrase.normalized}`)) {
          score = 0.92;
        } else if (normalizedInput.includes(phrase.normalized)) {
          score = 0.82;
        } else if (phrase.normalized.includes(normalizedInput)) {
          score = 0.76;
        }

        if (score < minimumScore) {
          continue;
        }

        if (!bestMatch || score > bestMatch.score) {
          bestMatch = {
            command: {
              id: command.id,
              phrases: command.phrases,
              payload: command.payload,
            },
            phrase: phrase.raw,
            score,
            transcript: input,
          };
        }
      }
    }

    return bestMatch;
  };
}

export function createScreamerBrowserClient(options = {}) {
  const emitter = createEmitter();
  const support = getScreamerBrowserSupport();
  let recognition = null;
  let state = 'idle';
  let destroyed = false;
  let activeSessionId = 0;

  function setState(nextState) {
    if (state === nextState) {
      return;
    }

    const previousState = state;
    state = nextState;
    emitter.emit('statechange', { previousState, state: nextState });
  }

  function teardownRecognition() {
    if (!recognition) {
      return;
    }

    recognition.onstart = null;
    recognition.onend = null;
    recognition.onerror = null;
    recognition.onresult = null;
    recognition = null;
  }

  function ensureRecognition(sessionId) {
    const resolved = resolveSpeechRecognitionCtor();
    if (!resolved) {
      throw buildUnsupportedError();
    }

    const instance = new resolved.ctor();
    instance.continuous = options.continuous ?? false;
    instance.interimResults = options.interimResults ?? true;
    instance.lang = options.lang ?? 'en-US';
    instance.maxAlternatives = 1;

    instance.onstart = () => {
      if (sessionId !== activeSessionId || destroyed) {
        return;
      }
      setState('listening');
    };

    instance.onresult = (event) => {
      if (sessionId !== activeSessionId || destroyed) {
        return;
      }

      let finalText = '';
      let interimText = '';

      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        const transcript = result?.[0]?.transcript ?? '';
        if (!transcript) {
          continue;
        }

        if (result.isFinal) {
          finalText += transcript;
        } else {
          interimText += transcript;
        }
      }

      const normalizedFinal = finalText.trim();
      const normalizedInterim = interimText.trim();

      if (normalizedInterim) {
        emitter.emit('partial', {
          text: normalizedInterim,
          sessionId,
        });
      }

      if (normalizedFinal) {
        emitter.emit('final', {
          text: normalizedFinal,
          normalizedText: normalizeTranscript(normalizedFinal),
          sessionId,
        });
      }
    };

    instance.onerror = (event) => {
      if (sessionId !== activeSessionId || destroyed) {
        return;
      }

      const detail = {
        code: event.error ?? 'unknown',
        message: event.message || `Speech recognition failed with code "${event.error ?? 'unknown'}".`,
        sessionId,
      };

      setState('idle');
      emitter.emit('error', detail);
    };

    instance.onend = () => {
      if (sessionId !== activeSessionId || destroyed) {
        teardownRecognition();
        return;
      }

      setState('idle');
      teardownRecognition();
    };

    recognition = instance;
    return instance;
  }

  return {
    getState() {
      return state;
    },
    getSupport() {
      return support;
    },
    on(event, handler) {
      return emitter.on(event, handler);
    },
    async start() {
      if (destroyed) {
        throw new Error('Cannot start Screamer web after destroy().');
      }
      if (!support.isSupported) {
        throw buildUnsupportedError();
      }
      if (state === 'starting' || state === 'listening') {
        return;
      }

      activeSessionId += 1;
      const sessionId = activeSessionId;
      const instance = ensureRecognition(sessionId);
      setState('starting');
      instance.start();
    },
    stop() {
      if (!recognition || (state !== 'starting' && state !== 'listening')) {
        return;
      }

      setState('stopping');
      recognition.stop();
    },
    abort() {
      if (!recognition) {
        return;
      }

      setState('idle');
      recognition.abort();
      teardownRecognition();
    },
    destroy() {
      destroyed = true;
      if (recognition) {
        recognition.abort();
      }
      teardownRecognition();
      emitter.clear();
      setState('idle');
    },
  };
}
