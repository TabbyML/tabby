import { Metadata } from 'next'

import { Mail } from './components/mail'

export const metadata: Metadata = {
  title: 'Mail Delivery'
}

export default function MailPage() {
  return <Mail />
}
