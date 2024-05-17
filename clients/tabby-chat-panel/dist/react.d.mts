import { RefObject } from 'react';
import { Thread } from '@quilted/threads';
import { Api } from './index.mjs';

declare function useClient(iframeRef: RefObject<HTMLIFrameElement>): Api | undefined;
declare function useServer(api: Api): Thread<Record<string, never>> | null;

export { useClient, useServer };
