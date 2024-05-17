import { createThreadFromIframe, createThreadFromInsideIframe } from '@quilted/threads';

function createClient(target, api) {
  return createThreadFromIframe(target, {
    expose: {
      navigate: api.navigate
    }
  });
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
