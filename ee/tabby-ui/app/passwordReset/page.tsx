import { Metadata } from 'next'

import PasswordReset from './components/passwordReset'

export const metadata: Metadata = {
  title: 'Reset password'
}

export default function Page() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center">
      <PasswordReset />
    </div>
  )
}
