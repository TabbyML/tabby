import { nanoid } from '@/lib/utils'
import { Chat } from '@/components/chat'
import { Metadata } from 'next'
 
export const metadata: Metadata = {
  title: 'Playground',
}

export default function IndexPage() {
  const id = nanoid()

  return <Chat id={id} />
}
