import { Metadata } from 'next'

import ResetPasswordPage from './components/reset-password-page'

export const metadata: Metadata = {
  title: 'Reset password'
}

export default function Page() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center">
      <ResetPasswordPage />
    </div>
  )
}
