import { findIndex } from 'lodash-es'

import { IntegrationKind } from '@/lib/gql/generates/graphql'
import { SubHeader } from '@/components/sub-header'

import { PROVIDER_KIND_METAS } from '../constants'

type Params = { kind: string }

export function generateStaticParams() {
  return PROVIDER_KIND_METAS.map(item => ({ kind: item.name }))
}

export default function IntegrationProviderLayout({
  children,
  params
}: {
  children: React.ReactNode
  params: Params
}) {
  const kindIndex = findIndex(
    PROVIDER_KIND_METAS,
    item => item.name === params.kind?.toLocaleLowerCase()
  )
  const kind =
    kindIndex > -1
      ? PROVIDER_KIND_METAS[kindIndex].enum
      : PROVIDER_KIND_METAS[0].enum

  return (
    <>
      <IntegrationHeader kind={kind} />
      {children}
    </>
  )
}

function IntegrationHeader({ kind }: { kind: IntegrationKind }) {
  let text = ''

  switch (kind) {
    case IntegrationKind.Github:
      text =
        'Connect to GitHub as a provider, and select repositories from this provider to serve as context, thereby improving the performance of large language models'
      break
    case IntegrationKind.GithubSelfHosted:
      text =
        'Connect to Self-Hosted GitHub as a provider, and select repositories from this provider to serve as context, thereby improving the performance of large language models'
      break
    case IntegrationKind.Gitlab:
      text =
        'Connect to GitLab as a provider, and select repositories from this provider to serve as context, thereby improving the performance of large language models'
      break
    case IntegrationKind.GitlabSelfHosted:
      text =
        'Connect to Self-Hosted GitLab as a provider, and select repositories from this provider to serve as context, thereby improving the performance of large language models'
      break
  }

  return (
    <SubHeader externalLink="https://tabby.tabbyml.com/blog/2023/10/16/repository-context-for-code-completion">
      {text}
    </SubHeader>
  )
}
