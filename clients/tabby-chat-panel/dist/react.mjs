import { fromIframe, fromInsideIframe } from '@remote-ui/rpc';
import { useMemo } from 'react';
import { createClient, createServer } from './index.mjs';

function useClient(iframeRef) {
  return useMemo(() => {
    if (iframeRef.current)
      return createClient(fromIframe(iframeRef.current));
  }, [iframeRef.current]);
}
function useServer(api) {
  return useMemo(() => {
    return createServer(fromInsideIframe(), api);
  }, [api]);
}

export { useClient, useServer };
