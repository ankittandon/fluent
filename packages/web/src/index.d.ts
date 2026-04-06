export type ScreamerBrowserState = 'idle' | 'starting' | 'listening' | 'stopping';

export type ScreamerSupport = {
  isSupported: boolean;
  recognitionCtorName: string | null;
};

export type ScreamerStateChangeEvent = {
  previousState: ScreamerBrowserState;
  state: ScreamerBrowserState;
};

export type ScreamerPartialEvent = {
  text: string;
  sessionId: number;
};

export type ScreamerFinalEvent = {
  text: string;
  normalizedText: string;
  sessionId: number;
};

export type ScreamerErrorEvent = {
  code: string;
  message: string;
  sessionId: number;
};

export type ScreamerCommandDefinition<TPayload = unknown> = {
  id: string;
  phrases: string[];
  payload?: TPayload;
};

export type ScreamerCommandMatch<TPayload = unknown> = {
  command: ScreamerCommandDefinition<TPayload>;
  phrase: string;
  score: number;
  transcript: string;
};

export type ScreamerBrowserOptions = {
  lang?: string;
  continuous?: boolean;
  interimResults?: boolean;
};

export interface ScreamerBrowserClient {
  getState(): ScreamerBrowserState;
  getSupport(): ScreamerSupport;
  on(event: 'statechange', handler: (event: ScreamerStateChangeEvent) => void): () => void;
  on(event: 'partial', handler: (event: ScreamerPartialEvent) => void): () => void;
  on(event: 'final', handler: (event: ScreamerFinalEvent) => void): () => void;
  on(event: 'error', handler: (event: ScreamerErrorEvent) => void): () => void;
  start(): Promise<void>;
  stop(): void;
  abort(): void;
  destroy(): void;
}

export declare function normalizeTranscript(value: string): string;
export declare function getScreamerBrowserSupport(): ScreamerSupport;
export declare function createCommandMatcher<TPayload = unknown>(
  commands: ScreamerCommandDefinition<TPayload>[],
  options?: {
    minimumScore?: number;
  }
): (input: string) => ScreamerCommandMatch<TPayload> | null;
export declare function createScreamerBrowserClient(
  options?: ScreamerBrowserOptions
): ScreamerBrowserClient;
