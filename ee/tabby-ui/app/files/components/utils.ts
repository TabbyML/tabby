import { isNil, keyBy, map } from 'lodash-es'

import {
  RepositoryKind,
  RepositoryListQuery
} from '@/lib/gql/generates/graphql'
import fetcher from '@/lib/tabby/fetcher'
import { ResolveEntriesResponse, TFile } from '@/lib/types'

export type ViewMode = 'tree' | 'blob'
type RepositoryItem = RepositoryListQuery['repositoryList'][0]

function getProviderVariantFromKind(kind: RepositoryKind) {
  return kind.toLowerCase().replaceAll('_', '')
}

function resolveRepositoryInfoFromPath(path: string | undefined): {
  repositoryKind?: RepositoryKind
  repositoryName?: string
  basename?: string
  repositorySpecifier?: string
  // todo
  viewMode?: string
  rev?: string
} {
  const emptyResult = {}
  if (!path) return emptyResult
  const pathSegments = path.split('/')
  const repositoryKindStr = pathSegments[0]

  if (!repositoryKindStr) {
    return emptyResult
  }

  if (repositoryKindStr === 'git') {
    if (pathSegments.length < 3) return emptyResult

    // e.g.  git/tabby/tree/main/ee/tabby-ui
    const repositoryName = pathSegments[1]
    return {
      repositoryKind: RepositoryKind.Git,
      repositoryName,
      viewMode: pathSegments[2],
      basename: pathSegments.slice(4).join('/').replace(/\/?$/, ''),
      repositorySpecifier: `git/${repositoryName}`,
      rev: pathSegments[3]
    }
  } else if (
    ['github', 'gitlab', 'githubselfhosted', 'gitlabselfhosted'].includes(
      repositoryKindStr
    )
  ) {
    // e.g.  /github/TabbyML/tabby/tree/main/ee/tabby-ui
    if (pathSegments.length < 4) return emptyResult
    let kind: RepositoryKind = RepositoryKind.Github
    switch (repositoryKindStr) {
      case 'github':
        kind = RepositoryKind.Github
        break
      case 'gitlab':
        kind = RepositoryKind.Gitlab
        break
      case 'githubselfhosted':
        kind = RepositoryKind.GithubSelfHosted
        break
      case 'gitlabselfhosted':
        kind = RepositoryKind.GitlabSelfHosted
        break
    }
    const repositoryName = [pathSegments[1], pathSegments[2]].join('/')

    return {
      repositoryKind: kind,
      repositoryName,
      basename: pathSegments.slice(5).join('/').replace(/\/?$/, ''),
      repositorySpecifier: `${getProviderVariantFromKind(
        kind
      )}/${repositoryName}`,
      viewMode: pathSegments[3],
      rev: pathSegments[4]
    }
  }
  return emptyResult
}

function resolveFileNameFromPath(path: string) {
  if (!path) return ''
  const pathSegments = path.split('/')
  return pathSegments[pathSegments.length - 1]
}

function getDirectoriesFromBasename(
  basename: string | undefined,
  isDir?: boolean
): string[] {
  if (isNil(basename)) return []

  let result = ['']
  const pathSegments = basename.split('/')
  // if path points to a file, the dirs do not include the path itself
  const len = isDir ? pathSegments.length : pathSegments.length - 1
  for (let i = 0; i < len; i++) {
    result.push(pathSegments.slice(0, i + 1).join('/'))
  }
  return result
}

async function fetchEntriesFromPath(
  path: string | undefined,
  repository: RepositoryListQuery['repositoryList'][0] | undefined
) {
  if (!path || !repository) return []

  const { basename, rev, viewMode } = resolveRepositoryInfoFromPath(path)
  // array of dir basename that do not include the repo name.
  const directoryPaths = getDirectoriesFromBasename(basename, viewMode === 'tree')
  // fetch all dirs from path
  const requests: Array<() => Promise<ResolveEntriesResponse>> =
    directoryPaths.map(
      dir => () =>
        fetcher(
          `/repositories/${getProviderVariantFromKind(repository.kind)}/${
            repository.id
          }/rev/${rev ?? 'main'}/${encodeURIComponentIgnoringSlash(dir)}`
        ).catch(e => [])
    )
  const entries = await Promise.all(requests.map(fn => fn()))
  let result: TFile[] = []
  for (let entry of entries) {
    if (entry?.entries?.length) {
      result = [...result, ...entry.entries]
    }
  }
  return result
}

function resolveRepoSpecifierFromRepoInfo(
  repo:
    | { kind: RepositoryKind | undefined; name: string | undefined }
    | undefined
) {
  if (repo?.kind && repo?.name) {
    return `${getProviderVariantFromKind(repo.kind)}/${repo.name}`
  }

  return undefined
}

function repositoryList2Map(repos: RepositoryListQuery['repositoryList']) {
  return keyBy(repos, o => `${getProviderVariantFromKind(o.kind)}/${o.name}`)
}

function repositoryMap2List(
  repoMap: Record<string, RepositoryItem>
): RepositoryListQuery['repositoryList'] {
  const list = map(repoMap, v => v)
  list.sort((a, b) => {
    return a.name.localeCompare(b.name)
  })
  return list
}

function encodeURIComponentIgnoringSlash(str: string) {
  return str
    .split('/')
    .map(part => encodeURIComponent(part))
    .join('/')
}

function resolveRepoRef(ref: string | undefined): {
  kind?: 'branch' | 'tag'
  name?: string
  ref: string
} {
  if (!ref)
    return {
      ref: ''
    }

  const regx = /refs\/(\w+)\/(.*)/
  const match = ref.match(regx)
  if (match) {
    const kind = match[1] === 'tags' ? 'tag' : 'branch'
    return {
      kind,
      name: encodeURIComponent(match[2]),
      ref
    }
  }
  return {
    ref
  }
}

function getDefaultRepoRef(refs: string[]) {
  let mainRef: string | undefined
  let masterRef: string | undefined
  let firstHeadRef: string | undefined
  let firstTagRef: string | undefined
  for (const ref of refs) {
    if (ref === 'refs/heads/main') {
      mainRef = ref
    } else if (ref === 'refs/heads/master') {
      masterRef = ref
    } else if (!firstHeadRef && ref.startsWith('refs/heads/')) {
      firstHeadRef = ref
    } else if (!firstTagRef && ref.startsWith('refs/tags/')) {
      firstTagRef = ref
    }
  }
  return mainRef || masterRef || firstHeadRef || firstTagRef
}

// todo encode & decode
function generateEntryPath(
  repo:
    | { kind: RepositoryKind | undefined; name: string | undefined }
    | undefined,
  rev: string | undefined,
  basename: string,
  kind: 'dir' | 'file'
) {
  const specifier = resolveRepoSpecifierFromRepoInfo(repo)
  // todo use 'main' as fallback
  const finalRev = rev ?? 'main'
  return `${specifier}/${
    kind === 'dir' ? 'tree' : 'blob'
  }/${finalRev}/${basename}`
}

function toEntryRequestUrl(
  repo: RepositoryItem | undefined,
  rev: string | undefined,
  basename: string | undefined
): string | null {
  const repoId = repo?.id
  const kind = repo?.kind
  if (!repoId || !kind || !rev) return null

  const activeRepoIdentity = `${getProviderVariantFromKind(kind)}/${repoId}`

  return `/repositories/${activeRepoIdentity}/rev/${rev}/${encodeURIComponentIgnoringSlash(
    basename ?? ''
  )}`
}

export {
  resolveRepoSpecifierFromRepoInfo,
  resolveFileNameFromPath,
  getDirectoriesFromBasename,
  fetchEntriesFromPath,
  resolveRepositoryInfoFromPath,
  repositoryList2Map,
  repositoryMap2List,
  encodeURIComponentIgnoringSlash,
  getProviderVariantFromKind,
  resolveRepoRef,
  getDefaultRepoRef,
  generateEntryPath,
  toEntryRequestUrl
}
