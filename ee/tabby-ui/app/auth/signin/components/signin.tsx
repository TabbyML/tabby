'use client'

import { useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import useSWRImmutable from 'swr/immutable'

import { useSignIn } from '@/lib/tabby/auth'
import fetcher from '@/lib/tabby/fetcher'
import { IconGithub, IconGoogle, IconSpinner } from '@/components/ui/icons'

import UserSignInForm from './user-signin-form'

export default function Signin() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const errorMessage = searchParams.get('error_message')
  const accessToken = searchParams.get('access_token')
  const refreshToken = searchParams.get('refresh_token')

  const shouldAutoSignin = !!accessToken && !!refreshToken
  const displayLoading = shouldAutoSignin && !errorMessage

  const signin = useSignIn()
  const { data }: { data?: string[] } = useSWRImmutable(
    shouldAutoSignin ? null : '/oauth/providers',
    fetcher
  )

  useEffect(() => {
    if (errorMessage) return
    if (accessToken && refreshToken) {
      signin({ accessToken, refreshToken }).then(() => router.replace('/'))
    }
  }, [searchParams])

  if (displayLoading) {
    return <IconSpinner className="h-8 w-8 animate-spin" />
  }

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
        <div className="relative mt-10 flex w-[350px] items-center py-5">
          <div className="grow border-t "></div>
          <span className="mx-4 shrink text-sm text-muted-foreground">
            Or Signin with
          </span>
          <div className="grow border-t "></div>
        </div>
      )}
      <div className="mx-auto flex items-center gap-6">
        {data?.includes('github') && (
          <a href={`/oauth/signin?provider=github`}>
            <IconGithub className="h-8 w-8" />
          </a>
        )}
        {data?.includes('google') && (
          <a href={`/oauth/signin?provider=google`}>
            <IconGoogle className="h-8 w-8" />
          </a>
        )}
      </div>
      {!!errorMessage && (
        <div className="mt-4 text-destructive">{errorMessage}</div>
      )}
    </>
  )
}
