export interface OpenAIProviderOptions {
  /** API key (required) */
  apiKey: string
  /** Default model name, can be overridden per-request (default: 'gpt-4o-mini') */
  model?: string
  /** API base URL for OpenAI-compatible endpoints (default: 'https://api.openai.com/v1') */
  baseURL?: string
  /** Enable debug logging (default: false) */
  debug?: boolean
}

export interface ChatRequest {
  messages?: Array<{
    role: string
    content: string | unknown[]
    name?: string
    callId?: string
    toolCalls?: Array<{ name: string; args: Record<string, unknown>; callId?: string }>
    reasoningContent?: string
    reasoning_content?: string
  }>
  tools?: Array<{ name: string; description?: string; schema?: unknown }>
  toolChoice?: 'auto' | 'required' | { name: string }
  parallel?: boolean
  /** Override the default model for this request */
  model?: string
  [key: string]: any
}

export interface ChatResponse {
  text: string
  usage?: { input: number; output: number; total: number }
  toolCalls?: Array<{ name: string; args: Record<string, unknown>; callId?: string }> | null
  reasoningContent?: string
  reasoning_content?: string
}

export interface StreamChunk {
  type?: 'delta' | 'usage'
  delta?: string
  usage?: { input: number; output: number; total: number }
}

export interface OpenAIProvider {
  name: string
  defaultModel: string
  /** Base URL of the API endpoint */
  baseURL: string
  chat(request: ChatRequest): Promise<ChatResponse>
  stream(request: ChatRequest): AsyncIterable<StreamChunk>
}

/** Create an OpenAI provider instance */
export function openai(options: OpenAIProviderOptions): OpenAIProvider

/**
 * Vercel AI SDK compatible factory:
 *   const openai = createOpenAI({ baseURL, apiKey })
 *   streamText({ model: openai("gpt-4o"), ... })
 */
export function createOpenAI(options: Omit<OpenAIProviderOptions, 'model'>): (modelName: string) => OpenAIProvider
