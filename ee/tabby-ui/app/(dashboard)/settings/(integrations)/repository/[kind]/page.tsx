import { NextPage } from 'next'
import { findIndex } from 'lodash-es'

import { IntegrationKind } from '@/lib/gql/generates/graphql'
import { SubHeader } from '@/components/sub-header'

import { PROVIDER_KIND_METAS } from '../constants'
import RepositoryProvidersPage from './components/provider-list'

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

  let kindText = 'Git'
  if (
    kind === IntegrationKind.Github ||
    kind === IntegrationKind.GithubSelfHosted
  ) {
    kindText = 'GitHub'
  }
  if (
    kind === IntegrationKind.Gitlab ||
    kind === IntegrationKind.GitlabSelfHosted
  ) {
    kindText = 'GitLab'
  }

  return (
    <>
      <SubHeader externalLink="https://tabby.tabbyml.com/blog/2023/10/16/repository-context-for-code-completion">
        Connect to {kindText} as a provider, and select repositories from this
        provider to serve as context, thereby improving the performance of large
        language models
      </SubHeader>
      <RepositoryProvidersPage kind={kind} />
    </>
  )
}

export default IntegrateGitPage
