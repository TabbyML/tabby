import { useRef, useEffect, useMemo } from 'react';
import { createClient, createServer } from './index.mjs';
import '@quilted/threads';

function useClient(iframeRef, api) {
  const clientRef = useRef(null);
  useEffect(() => {
    if (iframeRef.current && !clientRef.current) {
      clientRef.current = createClient(iframeRef.current, api);
    }
  }, [iframeRef.current]);
  return useMemo(() => clientRef.current, [clientRef.current]);
}
function useServer(api) {
  const serverRef = useRef(null);
  useEffect(() => {
    const isInIframe = window.self !== window.top;
    if (isInIframe && !serverRef.current)
      serverRef.current = createServer(api);
  }, []);
  return useMemo(() => serverRef.current, [serverRef.current]);
}

export { useClient, useServer };
