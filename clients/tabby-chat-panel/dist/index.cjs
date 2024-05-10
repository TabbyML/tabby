'use strict';

const rpc = require('@remote-ui/rpc');

function createClient(endpoint) {
  return rpc.createEndpoint(endpoint);
}
function createServer(endpoint, api) {
  const server = rpc.createEndpoint(endpoint);
  server.expose({
    init: api.init
  });
  return server;
}

exports.createClient = createClient;
exports.createServer = createServer;
