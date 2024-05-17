import { useMemo, useState, useRef, useEffect } from 'react';
import { createClient, createServer } from './index.mjs';
import '@quilted/threads';

function useClient(iframeRef) {
  return useMemo(() => {
    if (iframeRef.current) {
      return createClient(iframeRef.current);
    }
  }, [iframeRef.current]);
}
function useServer(api) {
  const [isInIframe, setIsInIframe] = useState(false);
  const serverRef = useRef(null);
  useEffect(() => {
    const isInIframe2 = window.self !== window.top;
    if (isInIframe2 && !serverRef.current)
      serverRef.current = createServer(api);
    setIsInIframe(isInIframe2);
  }, []);
  return useMemo(() => {
    return serverRef.current;
  }, [isInIframe]);
}

export { useClient, useServer };
