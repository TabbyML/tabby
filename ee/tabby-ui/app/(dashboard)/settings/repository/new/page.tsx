import { Metadata } from 'next'

import { NewRepository } from './components/new-page'

export const metadata: Metadata = {
  title: 'New Repository'
}

export default function IndexPage() {
  return (
    <div className="p-6">
      <NewRepository />
    </div>
  )
}
