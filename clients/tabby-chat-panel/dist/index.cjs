'use strict';

const threads = require('@quilted/threads');

function createClient(target, api) {
  return threads.createThreadFromIframe(target, {
    expose: {
      navigate: api.navigate
    }
  });
}
function createServer(api) {
  return threads.createThreadFromInsideIframe({
    expose: {
      init: api.init,
      sendMessage: api.sendMessage
    }
  });
}

exports.createClient = createClient;
exports.createServer = createServer;
