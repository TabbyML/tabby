'use client'

import React from 'react'
import { useRouter } from 'next/navigation'
import { isNil, pickBy } from 'lodash-es'
import { useQuery } from 'urql'

import { OAuthProvider } from '@/lib/gql/generates/graphql'
import { oauthCredential } from '@/lib/tabby/query'
import LoadingWrapper from '@/components/loading-wrapper'
import { ListSkeleton } from '@/components/skeleton'

import OAuthCredentialForm from '../../../components/oauth-credential-form'
import { SSOTypeRadio } from '../../../components/sso-type-radio'

interface OAuthCredentialDetailProps
  extends React.HTMLAttributes<HTMLDivElement> {
  provider: OAuthProvider
}

const OAuthCredentialDetail: React.FC<OAuthCredentialDetailProps> = ({
  provider
}) => {
  const router = useRouter()
  const [{ data, fetching }] = useQuery({
    query: oauthCredential,
    variables: { provider }
  })

  const credential = data?.oauthCredential

  const defaultValues = React.useMemo(() => {
    if (!credential) return undefined
    return pickBy(credential, v => !isNil(v))
  }, [credential])

  const onSubmitSuccess = () => {
    router.push('/settings/sso')
  }

  return (
    <div>
      <LoadingWrapper
        loading={fetching}
        fallback={<ListSkeleton className="mt-2" />}
      >
        <SSOTypeRadio value="oauth" readonly />
        <OAuthCredentialForm
          defaultValues={defaultValues}
          onSuccess={onSubmitSuccess}
          className="mt-6"
          defaultProvider={provider}
        />
      </LoadingWrapper>
    </div>
  )
}

export { OAuthCredentialDetail }
