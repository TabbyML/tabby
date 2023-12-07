'use client'

import { graphql } from '@/lib/gql/generates'
import { UserAuthForm } from './user-auth-form'
import { useSearchParams } from 'next/navigation'
import { useGraphQLQuery } from '@/lib/tabby-gql-client'

export const getIsAdminInitialized = graphql(/* GraphQL */ `
  query GetIsAdminInitialized {
    isAdminInitialized
  }
`)

export default function Signup() {
  const { data } = useGraphQLQuery(getIsAdminInitialized)
  const title = data?.isAdminInitialized
    ? 'Create an account'
    : 'Create an admin account'

  const searchParams = useSearchParams()
  const invitationCode = searchParams.get('invitationCode') || undefined

  return (
    <div className="space-y-6 w-[350px]">
      <div className="flex flex-col space-y-2 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        <p className="text-sm text-muted-foreground">
          Fill form below to create your account
        </p>
      </div>
      <UserAuthForm invitationCode={invitationCode} />
    </div>
  )
}
