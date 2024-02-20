import { Metadata } from 'next'

import { ResetPasswordRequestPage } from './components/reset-password-request-page'

export const metadata: Metadata = {
  title: 'Reset password'
}

export default function Page() {
  return <ResetPasswordRequestPage />
}
