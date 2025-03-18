import type { Thread } from 'tabby-threads'
import * as semver from 'semver'
import type { ClientApi } from './client'
import { type ServerApi, type ServerApiList, serverApiVersionList } from './server'

export type CreateThreadFunction<Self, Target> = (expose: Self) => Thread<Target>

export async function createServer(thread: Thread<ClientApi>): Promise<ClientApi> {
  const methods = await thread._requestMethods() as (keyof ClientApi)[]

  const clientApi: Record<string, any> = {}
  for (const method of methods) {
    clientApi[method] = thread[method]
  }
  return clientApi as ClientApi
}

function isCompatible(current: string, target: string): boolean {
  const currentSemver = semver.coerce(current)
  const targetSemver = semver.coerce(target)
  if (!currentSemver || !targetSemver) {
    return false
  }
  return semver.gte(currentSemver, targetSemver)
}

export async function createClient(thread: Thread<ServerApi>): Promise<ServerApiList> {
  const methods = await thread._requestMethods() as (keyof ServerApi)[]

  const serverApi: Record<string, any> = {}
  for (const method of methods) {
    serverApi[method] = thread[method]
  }

  const serverVersion = await thread.getVersion()
  const serverApiList: Record<string, any> = {}
  for (const version of serverApiVersionList) {
    if (isCompatible(serverVersion, version)) {
      serverApiList[version] = serverApi
    }
  }
  return serverApiList as ServerApiList
}
