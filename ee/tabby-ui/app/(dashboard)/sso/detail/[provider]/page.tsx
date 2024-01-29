import type { NextPage } from 'next'
import { OAUTH_PROVIDERS } from "@/lib/constant"
import { OAuthProvider } from '@/lib/gql/generates/graphql'

type Params = {
  provider: OAuthProvider
}

const PARAMS_TO_ENUM_ARR: Array<{name: string, enum: OAuthProvider}> = [
  {
    name: 'github',
    enum: OAuthProvider.Github
  },
  {
    name: 'google',
    enum: OAuthProvider.Google
  },
]

export function generateStaticParams() {
  return OAUTH_PROVIDERS.map(provider => {
    const p = PARAMS_TO_ENUM_ARR.find(x => x.enum === provider)
    return { provider: p!.name }
  })
}

const SSODetail: NextPage<{ params: Params }> = ({ params }) => {

  return (
    <div>
      detail
      <div>{params.provider}</div>
    </div>
  )
}

export default SSODetail