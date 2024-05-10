'use strict';

const rpc = require('@remote-ui/rpc');
const react = require('react');
const index = require('./index.cjs');

function useClient(iframeRef) {
  return react.useMemo(() => {
    if (iframeRef.current)
      return index.createClient(rpc.fromIframe(iframeRef.current));
  }, [iframeRef.current]);
}
function useServer(api) {
  return react.useMemo(() => {
    return index.createServer(rpc.fromInsideIframe(), api);
  }, []);
}

exports.useClient = useClient;
exports.useServer = useServer;
