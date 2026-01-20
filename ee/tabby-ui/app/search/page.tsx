import { Metadata } from 'next'
import { redirect } from 'next/navigation'

export const metadata: Metadata = {
  title: 'Search'
}

export default function IndexPage() {
  redirect('/')
}
