'use client'

import React from 'react'
import { useRouter } from 'next/navigation'
import { isNil, pickBy } from 'lodash-es'
import { useQuery } from 'urql'

import { OAuthProvider } from '@/lib/gql/generates/graphql'
import { ListSkeleton } from '@/components/skeleton'

import OAuthCredentialForm from './oauth-credential-form'
import { oauthCredential } from './oauth-credential-list'

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
      {fetching ? (
        <div>
          <ListSkeleton />
        </div>
      ) : (
        <OAuthCredentialForm
          provider={provider}
          defaultValues={defaultValues}
          onSuccess={onSubmitSuccess}
        />
      )}
    </div>
  )
}

export { OAuthCredentialDetail }
