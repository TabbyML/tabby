import { NextPage } from 'next'
import { findIndex } from 'lodash-es'

import GithubPage from './components/provider-list'
import { REPOSITORY_KIND_METAS } from './constants'

type Params = {
  kind: string
}

export function generateStaticParams() {
  return REPOSITORY_KIND_METAS.map(item => ({ kind: item.enum.toLowerCase() }))
}

const IntegrateGitPage: NextPage<{ params: Params }> = ({ params }) => {
  const kindIndex = findIndex(
    REPOSITORY_KIND_METAS,
    item => item.enum.toLocaleLowerCase() === params.kind?.toLocaleLowerCase()
  )
  const kind =
    kindIndex > -1
      ? REPOSITORY_KIND_METAS[kindIndex].enum
      : REPOSITORY_KIND_METAS[0].enum

  return <GithubPage kind={kind} />
}

export default IntegrateGitPage
