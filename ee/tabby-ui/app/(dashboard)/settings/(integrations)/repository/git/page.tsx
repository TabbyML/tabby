import { Metadata } from 'next'

import Git from './components/git'

export const metadata: Metadata = {
  title: 'Git Repositories'
}

export default function Generic() {
  return <Git />
}
