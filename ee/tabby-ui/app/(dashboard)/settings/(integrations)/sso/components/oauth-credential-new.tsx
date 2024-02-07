'use client'

import React from 'react'
import { useRouter } from 'next/navigation'

import { OAuthProvider } from '@/lib/gql/generates/graphql'

import OAuthCredentialForm from './oauth-credential-form'
import { SSOHeader } from './sso-header'

interface NewOAuthCredentialProps
  extends React.HTMLAttributes<HTMLDivElement> {}

const NewOAuthCredential: React.FC<NewOAuthCredentialProps> = () => {
  const router = useRouter()
  const onSuccess = () => {
    router.push(`/settings/sso`)
  }

  return (
    <>
      <SSOHeader />
      <OAuthCredentialForm
        provider={OAuthProvider.Github}
        isNew
        onSuccess={onSuccess}
      />
    </>
  )
}

export { NewOAuthCredential }
