'use client'

import { useEffect } from 'react'
import Link from 'next/link'
import useSWRImmutable from 'swr/immutable'

import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { useAllowSelfSignup } from '@/lib/hooks/use-server-info'
import { useSignIn } from '@/lib/tabby/auth'
import fetcher from '@/lib/tabby/fetcher'
import { IconGithub, IconGoogle, IconSpinner } from '@/components/ui/icons'

import UserSignInForm from './user-signin-form'

export default function SigninSection() {
  const { router, searchParams } = useRouterStuff()
  const allowSelfSignup = useAllowSelfSignup()
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
      <div className="w-[350px] space-y-4">
        <div className="flex flex-col space-y-2 text-center mt-2">
          <h1 className="text-2xl font-semibold tracking-tight">Sign In</h1>
          <p className="text-sm text-muted-foreground">
            Enter credentials to login to your account
          </p>
        </div>
        <UserSignInForm />
        {allowSelfSignup && (
          <div className="text-center text-sm">
            Don’t have an accout?
            <Link
              href="/auth/signin?mode=signup"
              className="font-semibold text-primary hover:underline ml-1"
            >
              Create an account
            </Link>
          </div>
        )}
      </div>

      {!!data?.length && (
        <div className="relative mt-4 flex w-[350px] items-center py-5">
          <div className="grow border-t "></div>
          <span className="mx-4 shrink text-sm text-muted-foreground">
            Or Sign In with
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
