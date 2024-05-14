import { useMemo, useState, useEffect } from 'react';
import { createClient, createServer } from './index.mjs';
import '@quilted/threads';

function useClient(iframeRef) {
  return useMemo(() => {
    if (iframeRef.current)
      return createClient(iframeRef.current);
  }, [iframeRef.current]);
}
function useServer(api) {
  const [isInIframe, setIsInIframe] = useState(false);
  useEffect(() => {
    setIsInIframe(window.self !== window.top);
  }, []);
  return useMemo(() => {
    if (isInIframe)
      return createServer(api);
  }, [isInIframe]);
}

export { useClient, useServer };
