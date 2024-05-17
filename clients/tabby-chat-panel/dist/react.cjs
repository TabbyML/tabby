'use strict';

const react = require('react');
const index = require('./index.cjs');
require('@quilted/threads');

function useClient(iframeRef) {
  return react.useMemo(() => {
    if (iframeRef.current) {
      return index.createClient(iframeRef.current);
    }
  }, [iframeRef.current]);
}
function useServer(api) {
  const [isInIframe, setIsInIframe] = react.useState(false);
  const serverRef = react.useRef(null);
  react.useEffect(() => {
    const isInIframe2 = window.self !== window.top;
    if (isInIframe2 && !serverRef.current)
      serverRef.current = index.createServer(api);
    setIsInIframe(isInIframe2);
  }, []);
  return react.useMemo(() => {
    return serverRef.current;
  }, [isInIframe]);
}

exports.useClient = useClient;
exports.useServer = useServer;
