'use client'

import React from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

import {
  useAllowSelfSignup,
  useIsEmailConfigured
} from '@/lib/hooks/use-server-info'

import ResetPasswordRequestSection from './reset-password-request-section'
import SelfRegisterSection from './self-signup-section'
import SigninSection from './signin-section'

export default function Signin() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const mode = searchParams.get('mode')?.toString()
  const isEmailConfigured = useIsEmailConfigured()
  const allowSelfSignup = useAllowSelfSignup()

  React.useEffect(() => {
    const isParamsInvalid =
      (isEmailConfigured === false && mode === 'reset') ||
      (allowSelfSignup === false && mode === 'signup')
    if (isParamsInvalid) {
      router.replace('/auth/signin')
    }
  }, [mode, isEmailConfigured, allowSelfSignup])

  if (mode === 'reset') {
    return <ResetPasswordRequestSection />
  }

  if (mode === 'signup') {
    return <SelfRegisterSection />
  }

  return <SigninSection />
}
