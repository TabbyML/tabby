'use client'

import fetcher from '@/lib/tabby/fetcher'
import UserSignInForm from './user-signin-form'
import useSWRImmutable from 'swr/immutable'

export default function Signin() {

  const { data } = useSWRImmutable('/oauth/providers', fetcher)

  return (
    <div className="w-[350px] space-y-6">
      <div className="flex flex-col space-y-2 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">Sign In</h1>
        <p className="text-sm text-muted-foreground">
          Enter credentials to login to your account
        </p>
      </div>
      <UserSignInForm />
      
      {/* todo third party login */}
      <a href={`http://localhost:8080/oauth/signin?provider=github`}>Signin with github</a>
    </div>
  )
}
