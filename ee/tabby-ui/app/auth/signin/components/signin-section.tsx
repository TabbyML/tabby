'use client'

import { useEffect } from 'react'
import Link from 'next/link'
import { findIndex } from 'lodash-es'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { AuthProviderKind } from '@/lib/gql/generates/graphql'
import useRouterStuff from '@/lib/hooks/use-router-stuff'
import {
  useAllowSelfSignup,
  useIsDisablePasswordLogin,
  useIsFetchingServerInfo
} from '@/lib/hooks/use-server-info'
import { useSession, useSignIn } from '@/lib/tabby/auth'
import { cn } from '@/lib/utils'
import { Card, CardContent } from '@/components/ui/card'
import {
  IconGithub,
  IconGitLab,
  IconGoogle,
  IconSpinner
} from '@/components/ui/icons'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

import LdapSignInForm from './ldap-signin-form'
import UserSignInForm from './user-signin-form'

const authProvidersQuery = graphql(/* GraphQL */ `
  query authProviders {
    authProviders {
      kind
    }
  }
`)

export default function SigninSection() {
  const { router, searchParams } = useRouterStuff()
  const isFetchingServerInfo = useIsFetchingServerInfo()
  const allowSelfSignup = useAllowSelfSignup()
  const isDisablePasswordLogin = useIsDisablePasswordLogin()
  const signin = useSignIn()
  const errorMessage = searchParams.get('error_message')
  const accessToken = searchParams.get('access_token')
  const refreshToken = searchParams.get('refresh_token')
  const passwordForceRender =
    searchParams.get('passwordSignIn')?.toString() === 'true'
  const shouldAutoSignin = !!accessToken && !!refreshToken

  const [{ data, fetching: fetchingAuthProviders }] = useQuery({
    query: authProvidersQuery,
    pause: shouldAutoSignin
  })
  const authProviders = data?.authProviders

  const enableGithubOauth =
    findIndex(authProviders, x => x.kind === AuthProviderKind.OauthGithub) > -1
  const enableGitlabOauth =
    findIndex(authProviders, x => x.kind === AuthProviderKind.OauthGitlab) > -1
  const enableGoogleOauth =
    findIndex(authProviders, x => x.kind === AuthProviderKind.OauthGoogle) > -1
  const enable3POauth =
    enableGithubOauth || enableGitlabOauth || enableGoogleOauth
  const enableLdapAuth =
    findIndex(authProviders, x => x.kind === AuthProviderKind.Ldap) > -1
  const passwordSigninVisible = passwordForceRender || !isDisablePasswordLogin
  const tabsDefaultValue = passwordSigninVisible ? 'standard' : 'ldap'
  const formVisible = passwordSigninVisible || enableLdapAuth
  const tabListVisible = passwordSigninVisible && enableLdapAuth

  useEffect(() => {
    if (errorMessage) return
    if (accessToken && refreshToken) {
      signin({ accessToken, refreshToken })
    }
  }, [searchParams])

  const { status } = useSession()
  useEffect(() => {
    if (status === 'authenticated') {
      router.replace('/')
    }
  }, [status])

  const displayLoading =
    isFetchingServerInfo ||
    fetchingAuthProviders ||
    (shouldAutoSignin && !errorMessage)

  if (displayLoading) {
    return <IconSpinner className="h-8 w-8 animate-spin" />
  }

  return (
    <>
      <div className="w-[350px] space-y-6">
        <div className="flex flex-col space-y-2 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">Sign In</h1>
          {formVisible && (
            <p className="text-sm text-muted-foreground">
              Enter {!passwordSigninVisible && enableLdapAuth ? 'LDAP' : ''}{' '}
              credentials to login to your account
            </p>
          )}
        </div>
        <Card
          className={cn('bg-background', {
            'border-0 shadow-0': !tabListVisible
          })}
        >
          <CardContent
            className={cn('pt-4', {
              'p-0': !tabListVisible
            })}
          >
            {formVisible && (
              <Tabs defaultValue={tabsDefaultValue}>
                {tabListVisible && (
                  <TabsList className="mb-2">
                    <TabsTrigger value="standard">Standard</TabsTrigger>
                    <TabsTrigger value="ldap">LDAP</TabsTrigger>
                  </TabsList>
                )}
                <TabsContent value="standard">
                  <UserSignInForm />
                </TabsContent>
                <TabsContent value="ldap">
                  <LdapSignInForm />
                </TabsContent>
              </Tabs>
            )}
          </CardContent>
        </Card>
        {allowSelfSignup && (
          <div className="text-center text-sm">
            Don’t have an accout?
            <Link
              href="/auth/signin?mode=signup"
              className="ml-1 font-semibold text-primary hover:underline"
            >
              Create an account
            </Link>
          </div>
        )}
      </div>

      {enable3POauth && (
        <div className="relative mt-4 flex w-[350px] items-center py-5">
          <div className="grow border-t "></div>
          <span className="mx-4 shrink text-sm text-muted-foreground">
            {formVisible ? 'Or' : ''} Sign In with
          </span>
          <div className="grow border-t "></div>
        </div>
      )}
      <div className="mx-auto flex items-center gap-8">
        {enableGithubOauth && (
          <a href={`/oauth/signin?provider=github`}>
            <IconGithub className="h-8 w-8" />
          </a>
        )}
        {enableGoogleOauth && (
          <a href={`/oauth/signin?provider=google`}>
            <IconGoogle className="h-8 w-8" />
          </a>
        )}
        {enableGitlabOauth && (
          <a href={`/oauth/signin?provider=gitlab`}>
            <IconGitLab className="h-8 w-8" />
          </a>
        )}
      </div>
      {!!errorMessage && (
        <div className="mt-4 text-destructive">{errorMessage}</div>
      )}
    </>
  )
}
