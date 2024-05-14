'use strict';

const threads = require('@quilted/threads');

function createClient(target) {
  return threads.createThreadFromIframe(target);
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
