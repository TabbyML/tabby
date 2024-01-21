import { Metadata } from 'next'

import Chats from './components/chats'

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
