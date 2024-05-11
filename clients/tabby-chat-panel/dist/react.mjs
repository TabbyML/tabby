import { useMemo, useState, useEffect } from 'react';
import { createThreadFromIframe, createThreadFromInsideIframe } from '@quilted/threads';
import { createClient, createServer } from './index.mjs';

function useClient(iframeRef) {
  return useMemo(() => {
    if (iframeRef.current)
      return createClient(createThreadFromIframe, iframeRef.current);
  }, [iframeRef.current]);
}
function useServer(api) {
  const [isInIframe, setIsInIframe] = useState(false);
  useEffect(() => {
    setIsInIframe(window.self !== window.top);
  }, []);
  return useMemo(() => {
    if (isInIframe) {
      return createServer(createThreadFromInsideIframe, api);
    }
  }, [isInIframe]);
}

export { useClient, useServer };
