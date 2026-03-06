// @sdjz/ai-sdk-provider-openai v0.1.0 — OpenAI provider factory

// ─── Zod → JSON Schema 转换 ───
function toJsonSchema(schema) {
  if (!schema) return {}
  // Already JSON Schema (has type field) -> return as-is
  if (schema.type) return schema
  // Zod object (has _def property)
  if (schema._def) return zodToJsonSchema(schema)
  // Fallback: try to serialize
  try { return JSON.parse(JSON.stringify(schema)) } catch { return {} }
}

function zodToJsonSchema(z) {
  if (!z || !z._def) return {}
  const def = z._def
  const typeName = def.typeName

  if (typeName === 'ZodObject') {
    const shape = def.shape?.() || def.shape || {}
    const properties = {}
    const required = []
    for (const [key, val] of Object.entries(shape)) {
      properties[key] = zodToJsonSchema(val)
      // ZodOptional 不加到 required
      if (val?._def?.typeName !== 'ZodOptional') {
        required.push(key)
      }
    }
    const result = { type: 'object', properties }
    if (required.length > 0) result.required = required
    return result
  }
  if (typeName === 'ZodString') return { type: 'string' }
  if (typeName === 'ZodNumber') return { type: 'number' }
  if (typeName === 'ZodBoolean') return { type: 'boolean' }
  if (typeName === 'ZodArray') {
    return { type: 'array', items: zodToJsonSchema(def.type) }
  }
  if (typeName === 'ZodOptional') {
    return zodToJsonSchema(def.innerType)
  }
  if (typeName === 'ZodDefault') {
    return zodToJsonSchema(def.innerType)
  }
  if (typeName === 'ZodEnum') {
    return { type: 'string', enum: def.values }
  }
  if (typeName === 'ZodLiteral') {
    return { type: typeof def.value, enum: [def.value] }
  }
  if (typeName === 'ZodUnion') {
    return { oneOf: (def.options || []).map(o => zodToJsonSchema(o)) }
  }
  if (typeName === 'ZodNullable') {
    const inner = zodToJsonSchema(def.innerType)
    return { ...inner, nullable: true }
  }
  // Fallback
  return {}
}

function getReasoningContent(message) {
  if (typeof message?.reasoningContent === 'string') return message.reasoningContent
  if (typeof message?.reasoning_content === 'string') return message.reasoning_content
  return undefined
}

function applyRequestExtras(body, request, reservedKeys) {
  for (const [key, value] of Object.entries(request || {})) {
    if (reservedKeys.has(key) || value === undefined) continue
    body[key] = value
  }
}

function stringifyContent(content) {
  return typeof content === 'string' ? content : JSON.stringify(content)
}

function mapMessage(message) {
  const reasoningContent = getReasoningContent(message)

  if (message.role === 'assistant' && message.toolCalls && message.toolCalls.length > 0) {
    return {
      role: 'assistant',
      content: message.content || null,
      ...(reasoningContent ? { reasoning_content: reasoningContent } : {}),
      tool_calls: message.toolCalls.map(tc => ({
        id: tc.callId || `call_${Math.random().toString(36).slice(2, 10)}`,
        type: 'function',
        function: {
          name: tc.name,
          arguments: typeof tc.args === 'string' ? tc.args : JSON.stringify(tc.args || {}),
        },
      })),
    }
  }

  if (message.role === 'tool') {
    return {
      role: 'tool',
      tool_call_id: message.callId || message.tool_call_id || 'unknown',
      content: stringifyContent(message.content),
    }
  }

  return {
    role: message.role,
    content: stringifyContent(message.content),
    ...(reasoningContent ? { reasoning_content: reasoningContent } : {}),
    ...(message.name ? { name: message.name } : {}),
  }
}

function applyTools(body, tools, toolChoice) {
  if (!tools || tools.length === 0) return

  body.tools = tools.map(t => ({
    type: 'function',
    function: {
      name: t.name,
      description: t.description || '',
      parameters: toJsonSchema(t.schema),
    },
  }))

  if (toolChoice) {
    if (typeof toolChoice === 'string') {
      body.tool_choice = toolChoice
    } else if (toolChoice.name) {
      body.tool_choice = { type: 'function', function: { name: toolChoice.name } }
    }
  }
}

function buildChatBody(request, model, options = {}) {
  const body = {
    model,
    messages: (request?.messages || []).map(mapMessage),
  }

  if (request?.maxTokens) {
    body.max_tokens = request.maxTokens
  }

  applyTools(body, request?.tools, request?.toolChoice)

  if (options.stream === true) {
    body.stream = true
    body.stream_options = { include_usage: true }
  }

  applyRequestExtras(
    body,
    request,
    new Set(['messages', 'tools', 'toolChoice', 'parallel', 'model', 'maxTokens', 'stream', 'stream_options']),
  )

  return body
}

function toUsage(usage) {
  return usage ? {
    input: usage.prompt_tokens || 0,
    output: usage.completion_tokens || 0,
    total: usage.total_tokens || 0,
  } : undefined
}

function parseToolArgs(rawArgs) {
  if (!rawArgs) return {}
  try {
    return JSON.parse(rawArgs)
  } catch {
    return {}
  }
}

function mergeToolCallDelta(accumulator, parts) {
  for (const part of parts || []) {
    const index = typeof part.index === 'number' ? part.index : accumulator.length
    if (!accumulator[index]) {
      accumulator[index] = {
        id: '',
        name: '',
        arguments: '',
      }
    }

    if (part.id) accumulator[index].id = part.id
    if (part.function?.name) accumulator[index].name = part.function.name
    if (typeof part.function?.arguments === 'string') {
      accumulator[index].arguments += part.function.arguments
    }
  }
}

function finalizeToolCalls(toolCalls) {
  const normalized = toolCalls
    .filter(Boolean)
    .map(tc => ({
      name: tc.name,
      args: parseToolArgs(tc.arguments),
      callId: tc.id,
    }))
    .filter(tc => tc.name)

  return normalized.length > 0 ? normalized : null
}

function ensureToolReasoningCompatibility(result, request, baseURL) {
  const needsCompatibility = Boolean(
    result?.toolCalls &&
    result.toolCalls.length > 0 &&
    request?.reasoning?.enabled !== false &&
    baseURL !== 'https://api.openai.com/v1',
  )

  if (!needsCompatibility || result.reasoningContent || result.reasoning_content) {
    return result
  }

  const fallbackReasoning = (result.text || '').trim() || 'Compatibility placeholder for missing reasoning_content in assistant tool call.'

  return {
    ...result,
    reasoningContent: fallbackReasoning,
    reasoning_content: fallbackReasoning,
  }
}

async function collectStreamChatResponse(resp) {
  const reader = resp.body?.getReader?.()
  if (!reader) throw new Error('Stream response body is not readable')

  const decoder = new TextDecoder()
  const toolCalls = []
  let buffer = ''
  let text = ''
  let reasoningContent = ''
  let usage

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || !trimmed.startsWith('data: ')) continue
      const data = trimmed.slice(6)
      if (data === '[DONE]') {
        return {
          text,
          usage,
          toolCalls: finalizeToolCalls(toolCalls),
          ...(reasoningContent ? { reasoningContent, reasoning_content: reasoningContent } : {}),
        }
      }

      let parsed
      try {
        parsed = JSON.parse(data)
      } catch {
        continue
      }

      if (parsed.usage) {
        usage = toUsage(parsed.usage)
      }

      const delta = parsed.choices?.[0]?.delta
      if (!delta) continue

      if (typeof delta.content === 'string') {
        text += delta.content
      }

      if (typeof delta.reasoning_content === 'string') {
        reasoningContent += delta.reasoning_content
      }

      if (Array.isArray(delta.tool_calls)) {
        mergeToolCallDelta(toolCalls, delta.tool_calls)
      }
    }
  }

  return {
    text,
    usage,
    toolCalls: finalizeToolCalls(toolCalls),
    ...(reasoningContent ? { reasoningContent, reasoning_content: reasoningContent } : {}),
  }
}

export function openai(options) {
  const apiKey = options.apiKey
  const defaultModel = options.model || 'gpt-4o-mini'
  const baseURL = (options.baseURL || 'https://api.openai.com/v1').replace(/\/+$/, '')
  const debug = options.debug || false

  // Debug logging with colors
  const log = debug ? (type, ...args) => {
    const colors = { req: '\x1b[36m', ok: '\x1b[32m', err: '\x1b[31m', reset: '\x1b[0m' }
    const prefix = type === 'req' ? `${colors.req}->${colors.reset}`
      : type === 'ok' ? `${colors.ok}ok${colors.reset}`
      : `${colors.err}-${colors.reset}`
    console.log(`${prefix} [ai-sdk:openai]`, ...args)
  } : () => {}

  return {
    name: `openai:${defaultModel}`,
    defaultModel,
    baseURL,

    async chat(request, signal) {
      const messages = request?.messages || []
      const tools = request?.tools
      // Support per-request model override
      const model = request?.model || defaultModel
      const shouldUseStreamForTools = Boolean(
        tools &&
        tools.length > 0 &&
        request?.reasoning?.enabled !== false &&
        baseURL !== 'https://api.openai.com/v1',
      )
      const body = buildChatBody(request, model, { stream: shouldUseStreamForTools })

      let resp
      const startTime = Date.now()
      log('req', `chat model=${model} messages=${messages.length}${tools ? ` tools=${tools.length}` : ''}${shouldUseStreamForTools ? ' mode=stream-tools' : ''}`)
      try {
        resp = await fetch(`${baseURL}/chat/completions`, {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            'authorization': `Bearer ${apiKey}`,
          },
          body: JSON.stringify(body),
          signal,
        })
      } catch (fetchErr) {
        log('err', `chat failed: ${fetchErr.message} (${Date.now() - startTime}ms)`)
        // 网络层错误（DNS、连接超时、SSL 等）
        const cause = fetchErr.cause || {}
        const code = cause.code || cause.errno || ''
        const hint = code === 'ENOTFOUND' ? ' (DNS resolution failed - check network/proxy)'
          : code === 'ECONNREFUSED' ? ' (connection refused - is the server running?)'
          : code === 'ETIMEDOUT' ? ' (connection timed out)'
          : code === 'CERT_HAS_EXPIRED' ? ' (SSL certificate expired)'
          : ''
        const err = new Error(`[${model}] Network error: ${fetchErr.message}${hint} - URL: ${baseURL}`)
        err.name = 'NetworkError'
        err.cause = fetchErr
        err.code = code
        throw err
      }

      if (!resp.ok) {
        const text = await resp.text().catch(() => '')
        const requestId = resp.headers.get('x-request-id') || 'unknown'

        // Try to parse JSON error body
        let errorDetail = text
        let errorCode = null
        let errorType = null
        try {
          const errJson = JSON.parse(text)
          if (errJson.error) {
            errorDetail = errJson.error.message || text
            errorCode = errJson.error.code || null
            errorType = errJson.error.type || null
          }
        } catch {}

        const baseMsg = `[${model}] ${errorDetail}`
        const debugInfo = `(status=${resp.status}, request_id=${requestId}${errorCode ? `, code=${errorCode}` : ''}${errorType ? `, type=${errorType}` : ''})`

        if (resp.status === 429) {
          const retryAfter = parseInt(resp.headers.get('retry-after') || '0', 10)
          const err = new Error(`Rate limited: ${baseMsg} ${debugInfo}`)
          err.name = 'RateLimitError'
          err.retryAfter = retryAfter || null
          err.requestId = requestId
          throw err
        }
        if (resp.status === 401 || resp.status === 403) {
          const err = new Error(`Auth error: ${baseMsg} ${debugInfo}`)
          err.name = 'AuthError'
          err.requestId = requestId
          err.status = resp.status
          throw err
        }
        if (resp.status === 400) {
          const err = new Error(`Bad request: ${baseMsg} ${debugInfo}`)
          err.name = 'ModelError'
          err.requestId = requestId
          throw err
        }
        if (resp.status === 404) {
          const err = new Error(`Model not found: ${baseMsg} ${debugInfo}`)
          err.name = 'ModelError'
          err.requestId = requestId
          throw err
        }
        if (resp.status === 413) {
          const err = new Error(`Context too long: ${baseMsg} ${debugInfo}`)
          err.name = 'ContextLengthError'
          err.requestId = requestId
          throw err
        }
        if (resp.status >= 500) {
          const err = new Error(`Server error: ${baseMsg} ${debugInfo}`)
          err.name = 'NetworkError'
          err.requestId = requestId
          err.status = resp.status
          throw err
        }

        const err = new Error(`API error: ${baseMsg} ${debugInfo}`)
        err.requestId = requestId
        err.status = resp.status
        throw err
      }

      if (shouldUseStreamForTools) {
        const result = ensureToolReasoningCompatibility(
          await collectStreamChatResponse(resp),
          request,
          baseURL,
        )
        const latency = Date.now() - startTime
        log('ok', `chat(stream) ${latency}ms tokens=${result.usage?.total || 'N/A'}${result.toolCalls ? ` tools=${result.toolCalls.length}` : ''}${result.reasoningContent ? ` reasoning=${result.reasoningContent.length}` : ''}`)
        return result
      }

      const json = await resp.json()
      const choice = json.choices?.[0]
      const message = choice?.message
      const reasoningContent = getReasoningContent(message)

      const toolCalls = message?.tool_calls?.map(tc => ({
        name: tc.function?.name,
        args: JSON.parse(tc.function?.arguments || '{}'),
        callId: tc.id,
      })) || null

      const latency = Date.now() - startTime
      log('ok', `chat ${latency}ms tokens=${json.usage?.total_tokens || 'N/A'}${toolCalls ? ` tools=${toolCalls.length}` : ''}`)

      return ensureToolReasoningCompatibility({
        text: message?.content || '',
        usage: toUsage(json.usage),
        toolCalls: toolCalls && toolCalls.length > 0 ? toolCalls : null,
        ...(reasoningContent ? { reasoningContent, reasoning_content: reasoningContent } : {}),
      }, request, baseURL)
    },

    async *stream(request, signal) {
      const messages = request?.messages || []
      const model = request?.model || defaultModel

      const body = {
        ...buildChatBody(request, model, { stream: true }),
      }

      let resp
      const startTime = Date.now()
      log('req', `stream model=${model} messages=${messages.length}`)
      try {
        resp = await fetch(`${baseURL}/chat/completions`, {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            'authorization': `Bearer ${apiKey}`,
          },
          body: JSON.stringify(body),
          signal,
        })
      } catch (fetchErr) {
        log('err', `stream failed: ${fetchErr.message} (${Date.now() - startTime}ms)`)
        const cause = fetchErr.cause || {}
        const code = cause.code || cause.errno || ''
        const hint = code === 'ENOTFOUND' ? ' (DNS resolution failed - check network/proxy)'
          : code === 'ECONNREFUSED' ? ' (connection refused)'
          : code === 'ETIMEDOUT' ? ' (connection timed out)'
          : ''
        const err = new Error(`[${model}] Stream network error: ${fetchErr.message}${hint} - URL: ${baseURL}`)
        err.name = 'NetworkError'
        err.cause = fetchErr
        err.code = code
        throw err
      }

      if (!resp.ok) {
        const text = await resp.text().catch(() => '')
        const requestId = resp.headers.get('x-request-id') || 'unknown'

        let errorDetail = text
        try {
          const errJson = JSON.parse(text)
          if (errJson.error) errorDetail = errJson.error.message || text
        } catch {}

        const err = new Error(`[${model}] Stream error: ${errorDetail} (status=${resp.status}, request_id=${requestId})`)
        err.requestId = requestId
        err.status = resp.status
        if (resp.status === 429) err.name = 'RateLimitError'
        else if (resp.status === 401 || resp.status === 403) err.name = 'AuthError'
        else if (resp.status >= 500) err.name = 'NetworkError'
        throw err
      }

      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed || !trimmed.startsWith('data: ')) continue
          const data = trimmed.slice(6)
          if (data === '[DONE]') return

          try {
            const parsed = JSON.parse(data)

            // Parse usage (SSE last message usually contains usage)
            if (parsed.usage) {
              yield {
                type: 'usage',
                usage: {
                  input: parsed.usage.prompt_tokens || 0,
                  output: parsed.usage.completion_tokens || 0,
                  total: parsed.usage.total_tokens || 0,
                },
              }
            }

            const delta = parsed.choices?.[0]?.delta?.content
            if (delta) yield { type: 'delta', delta }
          } catch (parseErr) {
            // SSE parse error - log for debugging
            if (data.trim().length > 0) {
              console.warn('[ai-sdk:openai:stream] failed to parse SSE chunk:', data.slice(0, 200))
            }
          }
        }
      }
    },
  }
}

// Vercel AI SDK compatible createOpenAI:
// const openai = createOpenAI({ baseURL, apiKey })
// streamText({ model: openai("gpt-4o"), ... })
export function createOpenAI(options) {
  const apiKey = options.apiKey
  const baseURL = (options.baseURL || 'https://api.openai.com/v1').replace(/\/+$/, '')

  // Returns a callable function: openai("model-name") returns provider
  const factory = (modelName) => {
    return openai({ apiKey, baseURL, model: modelName })
  }

  factory.apiKey = apiKey
  factory.baseURL = baseURL

  return factory
}
