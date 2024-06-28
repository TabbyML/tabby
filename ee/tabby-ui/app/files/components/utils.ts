import { isNil, keyBy, map, trimEnd } from 'lodash-es'

import {
  RepositoryKind,
  RepositoryListQuery
} from '@/lib/gql/generates/graphql'

export type ViewMode = 'tree' | 'blob'
type RepositoryItem = RepositoryListQuery['repositoryList'][0]

export enum CodeBrowserError {
  INVALID_URL = 'INVALID_URL',
  NOT_FOUND = 'NOT_FOUND',
  REPOSITORY_NOT_FOUND = 'REPOSITORY_NOT_FOUND',
  REPOSITORY_SYNC_FAILED = 'REPOSITORY_SYNC_FAILED'
}

const repositoryKindStrList = Object.keys(RepositoryKind).map(kind =>
  getProviderVariantFromKind(kind as RepositoryKind)
)

function getProviderVariantFromKind(kind: RepositoryKind) {
  return kind.toLowerCase().replaceAll('_', '')
}

function resolveRepositoryInfoFromPath(path: string | undefined): {
  repositoryKind?: RepositoryKind
  repositoryName?: string
  basename?: string
  repositorySpecifier?: string
  viewMode?: string
  rev?: string
} {
  const emptyResult = {}

  if (!path) return emptyResult
  const separatorIndex = path.indexOf('/-/')

  const pathSegments = path.split('/')
  const repositoryKindStr = pathSegments[0]
  const isValidRepositoryKind =
    repositoryKindStrList.includes(repositoryKindStr)

  if (!isValidRepositoryKind || separatorIndex === -1) {
    return emptyResult
  }

  let kind: RepositoryKind = RepositoryKind.Git
  switch (repositoryKindStr) {
    case 'git':
      kind = RepositoryKind.Git
      break
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
  let basename: string | undefined
  let viewMode: ViewMode | undefined
  let rev: string | undefined

  const treeSeparatorIndex = path.indexOf('/-/tree/')
  const blobSeparatorIndex = path.indexOf('/-/blob/')

  if (treeSeparatorIndex > -1) {
    viewMode = 'tree'
    const temp = path.slice(treeSeparatorIndex + '/-/tree/'.length)
    const tempSegments = temp.split('/')
    rev = tempSegments[0]
    basename = trimEnd(tempSegments.slice(1).join('/'), '/')
  }

  if (blobSeparatorIndex > -1) {
    viewMode = 'blob'
    const temp = path.slice(blobSeparatorIndex + '/-/blob/'.length)
    const tempSegments = temp.split('/')
    rev = tempSegments[0]
    basename = trimEnd(tempSegments.slice(1).join('/'), '/')
  }

  const repositorySpecifier = path.split('/-/')[0]
  const repositoryName = repositorySpecifier.split('/').slice(1).join('/')

  return {
    repositorySpecifier: path.split('/-/')[0],
    repositoryName,
    repositoryKind: kind,
    rev: !isNil(rev) ? decodeURIComponent(rev) : undefined,
    viewMode,
    basename: !isNil(basename) ? decodeURIComponent(basename) : undefined
  }
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
  name: string
  ref: string
} {
  if (!ref)
    return {
      name: '',
      ref: ''
    }

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
    name: '',
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

function generateEntryPath(
  repo:
    | { kind: RepositoryKind | undefined; name: string | undefined }
    | undefined,
  rev: string | undefined,
  basename: string,
  kind: 'dir' | 'file'
) {
  const specifier = resolveRepoSpecifierFromRepoInfo(repo)
  return `${specifier}/-/${
    kind === 'file' ? 'blob' : 'tree'
  }/${encodeURIComponent(rev ?? '')}/${encodeURIComponentIgnoringSlash(
    basename ?? ''
  )}`
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

  return `/repositories/${activeRepoIdentity}/rev/${encodeURIComponent(
    rev
  )}/${encodeURIComponentIgnoringSlash(basename ?? '')}`
}

function parseLineFromSearchParam(line: string | undefined): {
  start?: number
  end?: number
} {
  if (!line) return {}
  const [startStr, endStr] = line.split('-')
  const startNumber = parseInt(startStr)
  const endNumber = parseInt(endStr)
  return {
    start: Number.isNaN(startNumber) ? undefined : startNumber,
    end: Number.isNaN(endNumber) ? undefined : endNumber
  }
}

// hash will be like #L10 or #L10-L20
function parseLineNumberFromHash(hash: string | undefined): {
  start?: number
  end?: number
} | null {
  const regex = /^#L(\d+)(?:-L(\d+))?/
  if (!hash) return null
  const match = regex.exec(hash)
  if (!match) return null

  const [, startStr, endStr] = match
  const startNumber = parseInt(startStr)
  const endNumber = parseInt(endStr)
  return {
    start: Number.isNaN(startNumber) ? undefined : startNumber,
    end: Number.isNaN(endNumber) ? undefined : endNumber
  }
}

export {
  resolveRepoSpecifierFromRepoInfo,
  resolveFileNameFromPath,
  getDirectoriesFromBasename,
  resolveRepositoryInfoFromPath,
  repositoryList2Map,
  repositoryMap2List,
  encodeURIComponentIgnoringSlash,
  getProviderVariantFromKind,
  resolveRepoRef,
  getDefaultRepoRef,
  generateEntryPath,
  toEntryRequestUrl,
  parseLineFromSearchParam,
  parseLineNumberFromHash
}
