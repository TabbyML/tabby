import { NextPage } from 'next'
import { findIndex } from 'lodash-es'

import GithubPage from './components/provider-list'
import { PROVIDER_KIND_METAS } from '../constants'

type Params = {
  kind: string
}

export function generateStaticParams() {
  return PROVIDER_KIND_METAS.map(item => ({ kind: item.name }))
}

const IntegrateGitPage: NextPage<{ params: Params }> = ({ params }) => {
  const kindIndex = findIndex(
    PROVIDER_KIND_METAS,
    item => item.name === params.kind?.toLocaleLowerCase()
  )
  const kind =
    kindIndex > -1
      ? PROVIDER_KIND_METAS[kindIndex].enum
      : PROVIDER_KIND_METAS[0].enum

  return <GithubPage kind={kind} />
}

export default IntegrateGitPage
