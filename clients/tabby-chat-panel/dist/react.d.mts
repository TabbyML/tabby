import * as _quilted_threads from '@quilted/threads';
import { RefObject } from 'react';
import { Api } from './index.mjs';

declare function useClient(iframeRef: RefObject<HTMLIFrameElement>): _quilted_threads.Thread<Record<string, never>> | undefined;
declare function useServer(api: Api): _quilted_threads.Thread<Record<string, never>> | undefined;

export { useClient, useServer };
