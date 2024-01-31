'use client'

import { useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import useSWRImmutable from 'swr/immutable'

import { saveAuthToken } from '@/lib/tabby/auth'
import fetcher from '@/lib/tabby/fetcher'
import { IconGithub, IconGoogle } from '@/components/ui/icons'

import UserSignInForm from './user-signin-form'

export default function Signin() {
  const searchParams = useSearchParams()
  const errorMessage = searchParams.get('error_message')
  const accessToken = searchParams.get('access_token')
  const refreshToken = searchParams.get('refresh_token')

  const { data }: { data?: string[] } = useSWRImmutable(
    '/oauth/providers',
    fetcher
  )

  useEffect(() => {
    if (errorMessage) return
    if (accessToken && refreshToken) {
      saveAuthToken({ accessToken, refreshToken })
    }
  }, [searchParams])

  return (
    <>
      <div className="w-[350px] space-y-6">
        <div className="flex flex-col space-y-2 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">Sign In</h1>
          <p className="text-sm text-muted-foreground">
            Enter credentials to login to your account
          </p>
        </div>
        <UserSignInForm />
      </div>

      {!!data?.length && (
        <div className="relative flex py-5 items-center w-[350px] mt-10">
          <div className="flex-grow border-t "></div>
          <span className="flex-shrink mx-4 text-sm text-muted-foreground">
            Or Signin with
          </span>
          <div className="flex-grow border-t "></div>
        </div>
      )}
      <div className="flex items-center gap-6 mx-auto">
        {data?.includes('github') && (
          <a href={`http://localhost:8080/oauth/signin?provider=github`}>
            <IconGithub className="w-8 h-8" />
          </a>
        )}
        {data?.includes('google') && (
          <a href={`http://localhost:8080/oauth/signin?provider=google`}>
            <IconGoogle className="w-8 h-8" />
          </a>
        )}
      </div>
      {!!errorMessage && (
        <div className="text-destructive mt-4">{errorMessage}</div>
      )}
    </>
  )
}
