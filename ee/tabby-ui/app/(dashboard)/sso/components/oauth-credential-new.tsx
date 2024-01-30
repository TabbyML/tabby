'use client'

import React from 'react'
import { useRouter } from 'next/navigation'

import { CardTitle } from '@/components/ui/card'

import OAuthCredentialForm from './oauth-credential-form'

interface NewOAuthCredentialProps
  extends React.HTMLAttributes<HTMLDivElement> {}

const NewOAuthCredential: React.FC<NewOAuthCredentialProps> = () => {
  const router = useRouter()
  const onSuccess = () => {
    router.push(`/sso`)
  }

  return (
    <>
      <CardTitle className="mb-6">Create Credential</CardTitle>
      <OAuthCredentialForm onSuccess={onSuccess} />
    </>
  )
}

export { NewOAuthCredential }
