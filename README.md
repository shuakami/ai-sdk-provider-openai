# @sdjz/ai-sdk-provider-openai

OpenAI provider for `@sdjz/ai-sdk`.

## Installation

```bash
npm install @sdjz/ai-sdk @sdjz/ai-sdk-provider-openai
```

## Usage

```ts
import { createAI } from '@sdjz/ai-sdk'
import { openai } from '@sdjz/ai-sdk-provider-openai'

const provider = openai({
  apiKey: process.env.OPENAI_API_KEY!,
  model: 'gpt-4o-mini',
})

const ai = createAI({ provider })
const result = await ai.chat('Hello')

console.log(result.text)
```

## Custom Base URL

```ts
const provider = openai({
  apiKey: process.env.OPENAI_API_KEY!,
  baseURL: 'https://your-proxy.example.com/v1',
  model: 'gpt-4o-mini',
})
```

## License

MIT
