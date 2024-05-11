'use strict';

const react = require('react');
const threads = require('@quilted/threads');
const index = require('./index.cjs');

function useClient(iframeRef) {
  return react.useMemo(() => {
    if (iframeRef.current)
      return index.createClient(threads.createThreadFromIframe, iframeRef.current);
  }, [iframeRef.current]);
}
function useServer(api) {
  const [isInIframe, setIsInIframe] = react.useState(false);
  react.useEffect(() => {
    setIsInIframe(window.self !== window.top);
  }, []);
  return react.useMemo(() => {
    if (isInIframe) {
      return index.createServer(threads.createThreadFromInsideIframe, api);
    }
  }, [isInIframe]);
}

exports.useClient = useClient;
exports.useServer = useServer;
