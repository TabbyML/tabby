import { RefObject } from 'react';
import { ClientApi, ServerApi } from './index.mjs';

declare function useClient(iframeRef: RefObject<HTMLIFrameElement>, api: ClientApi): ServerApi | null;
declare function useServer(api: ServerApi): ClientApi | null;

export { useClient, useServer };
