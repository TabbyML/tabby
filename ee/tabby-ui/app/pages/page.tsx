import { Metadata } from 'next'
import { redirect } from 'next/navigation'

export const metadata: Metadata = {
  title: 'Pages'
}

export default function Pages() {
  redirect('/')
}
