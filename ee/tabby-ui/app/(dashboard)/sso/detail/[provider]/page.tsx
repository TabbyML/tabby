import type { NextPage } from 'next'
import { OAUTH_PROVIDERS, OauthProvider } from "@/lib/constant"

type Params = {
  provider: OauthProvider
}

export function generateStaticParams() {
  return OAUTH_PROVIDERS.map(provider => ({ provider }))
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