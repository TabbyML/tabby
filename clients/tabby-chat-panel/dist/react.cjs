'use strict';

const react = require('react');
const index = require('./index.cjs');
require('@quilted/threads');

function useClient(iframeRef, api) {
  const clientRef = react.useRef(null);
  react.useEffect(() => {
    if (iframeRef.current && !clientRef.current) {
      clientRef.current = index.createClient(iframeRef.current, api);
    }
  }, [iframeRef.current]);
  return react.useMemo(() => clientRef.current, [clientRef.current]);
}
function useServer(api) {
  const serverRef = react.useRef(null);
  react.useEffect(() => {
    const isInIframe = window.self !== window.top;
    if (isInIframe && !serverRef.current)
      serverRef.current = index.createServer(api);
  }, []);
  return react.useMemo(() => serverRef.current, [serverRef.current]);
}

exports.useClient = useClient;
exports.useServer = useServer;
