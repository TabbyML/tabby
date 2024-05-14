import { Metadata } from 'next'

import Chats from '../components/chats-mock'

export const metadata: Metadata = {
  title: 'Playground'
}

export default function IndexPage() {
  return (
    <main className="flex flex-1 flex-col">
      <Chats />
    </main>
  )
}
