import { Metadata } from 'next'

import Subscription from './components/subscription'

export const metadata: Metadata = {
  title: 'Subscription'
}

export default function SubscriptionPage() {
  return <Subscription />
}
