import { createThreadFromIframe } from 'tabby-threads'
import { createClient as createClientFromThread } from './thread'
import type { ClientApi } from './client'
import type { ServerApi, ServerApiList } from './server'

export async function createClient(iframe: HTMLIFrameElement, api: ClientApi): Promise<ServerApiList> {
  const thread = createThreadFromIframe<ClientApi, ServerApi>(iframe, { expose: api })
  return await createClientFromThread(thread)
}
