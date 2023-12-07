import { Metadata } from 'next'
import Chats from './components/chats'
import { Header } from '@/components/header'

export const metadata: Metadata = {
  title: 'Playground'
}

export default function IndexPage() {
  return (
    <>
      <Header />
      <main className="bg-muted/50 flex flex-1 flex-col">
        <Chats />
      </main>
    </>
  )
}
