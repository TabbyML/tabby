import { createThreadFromIframe, createThreadFromInsideIframe } from '@quilted/threads';

function createClient(target) {
  return createThreadFromIframe(target);
}
function createServer(api) {
  return createThreadFromInsideIframe({
    expose: {
      init: api.init,
      sendMessage: api.sendMessage
    }
  });
}

export { createClient, createServer };
