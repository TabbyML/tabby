import * as _remote_ui_rpc from '@remote-ui/rpc';
import { RefObject } from 'react';
import { Api } from './index.cjs';

declare function useClient(iframeRef: RefObject<HTMLIFrameElement>): _remote_ui_rpc.Endpoint<Api> | undefined;
declare function useServer(api: Api): _remote_ui_rpc.Endpoint<unknown>;

export { useClient, useServer };
