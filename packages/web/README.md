# `@screamer/web`

Experimental Screamer browser SDK for embedding voice control in web apps.

## Status

This first package focuses on the developer experience and event model.

- It exposes a Screamer-style push-to-talk client API.
- It currently uses the browser speech recognition engine where available.
- It is designed so the backend can later be swapped for a Screamer-managed web inference runtime.

## Example

```js
import { createScreamerBrowserClient, createCommandMatcher } from '@screamer/web';

const client = createScreamerBrowserClient();
const matchCommand = createCommandMatcher([
  { id: 'open-news', phrases: ['open news', 'go to news'] },
]);

client.on('final', ({ text }) => {
  const match = matchCommand(text);
  if (match?.command.id === 'open-news') {
    window.location.href = '/news';
  }
});

await client.start();
```
