import { isNil, keyBy } from 'lodash-es'

import {
  RepositoryKind,
  RepositoryListQuery
} from '@/lib/gql/generates/graphql'
import fetcher from '@/lib/tabby/fetcher'
import { ResolveEntriesResponse, TFile } from '@/lib/types'

function getProviderVariantFromKind(kind: RepositoryKind) {
  return kind.toLowerCase().replaceAll('_', '')
}

function resolveRepositoryInfoFromPath(path: string | undefined): {
  repositoryKind?: RepositoryKind
  repositoryName?: string
  basename?: string
  repositorySpecifier?: string
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

    // e.g.  git/tabby/main/ee/tabby-ui
    const repositoryName = pathSegments[1]
    return {
      repositoryKind: RepositoryKind.Git,
      repositoryName,
      basename: pathSegments.slice(3).join('/'),
      repositorySpecifier: `git/${repositoryName}`,
      rev: pathSegments[2]
    }
  } else if (
    ['github', 'gitlab', 'githubselfhosted', 'gitlabselfhosted'].includes(
      repositoryKindStr
    )
  ) {
    // e.g.  /github/TabbyML/tabby/main/ee/tabby-ui
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
      basename: pathSegments.slice(4).join('/'),
      repositorySpecifier: `${getProviderVariantFromKind(
        kind
      )}/${repositoryName}`,
      rev: pathSegments[3]
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

  const { basename, rev } = resolveRepositoryInfoFromPath(path)
  // array of dir basename that do not include the repo name.
  const directoryPaths = getDirectoriesFromBasename(basename)
  // fetch all dirs from path
  const requests: Array<() => Promise<ResolveEntriesResponse>> =
    directoryPaths.map(
      dir => () =>
        fetcher(
          encodeURIComponentIgnoringSlash(
            `/repositories/${getProviderVariantFromKind(repository.kind)}/${
              repository.id
            }/rev/${encodeURIComponent(rev ?? '')}/${dir}`
          )
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

function encodeURIComponentIgnoringSlash(str: string) {
  return str
    .split('/')
    .map(part => encodeURIComponent(part))
    .join('/')
}

function resolveRepoRef(ref: string): {
  kind?: 'branch' | 'tag'
  name?: string
  ref: string
} {
  const regx = /refs\/(\w+)\/(.*)/
  const match = ref.match(regx)
  if (match) {
    const kind = match[1] === 'tags' ? 'tag' : 'branch'
    return {
      kind,
      name: match[2],
      ref
    }
  }
  return {
    ref
  }
}

export {
  resolveRepoSpecifierFromRepoInfo,
  resolveFileNameFromPath,
  getDirectoriesFromBasename,
  fetchEntriesFromPath,
  resolveRepositoryInfoFromPath,
  repositoryList2Map,
  encodeURIComponentIgnoringSlash,
  getProviderVariantFromKind,
  resolveRepoRef
}
