import { useRef, useEffect } from 'react';
import { createClient, createServer } from './index.mjs';
import '@quilted/threads';

function useClient(iframeRef, api) {
  const clientRef = useRef(null);
  useEffect(() => {
    if (iframeRef.current && !clientRef.current) {
      clientRef.current = createClient(iframeRef.current, api);
    }
  }, [iframeRef.current]);
  return clientRef.current;
}
function useServer(api) {
  const serverRef = useRef(null);
  useEffect(() => {
    const isInIframe = window.self !== window.top;
    if (isInIframe && !serverRef.current)
      serverRef.current = createServer(api);
  }, []);
  return serverRef.current;
}

export { useClient, useServer };
