'use strict';

const react = require('react');
const index = require('./index.cjs');
require('@quilted/threads');

function useClient(iframeRef) {
  return react.useMemo(() => {
    if (iframeRef.current)
      return index.createClient(iframeRef.current);
  }, [iframeRef.current]);
}
function useServer(api) {
  const [isInIframe, setIsInIframe] = react.useState(false);
  react.useEffect(() => {
    setIsInIframe(window.self !== window.top);
  }, []);
  return react.useMemo(() => {
    if (isInIframe)
      return index.createServer(api);
  }, [isInIframe]);
}

exports.useClient = useClient;
exports.useServer = useServer;
