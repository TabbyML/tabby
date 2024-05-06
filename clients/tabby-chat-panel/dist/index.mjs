import { createEndpoint } from '@remote-ui/rpc';

function createClient(endpoint) {
  return createEndpoint(endpoint);
}
function createServer(endpoint, api) {
  const server = createEndpoint(endpoint);
  server.expose({
    init: api.init
  });
  return server;
}

export { createClient, createServer };
