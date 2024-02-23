import { Metadata } from 'next'

import General from './components/general'

export const metadata: Metadata = {
  title: 'General'
}

export default function GeneralSettings() {
  // todo abstract settings-layout after email was merged
  return (
    <div className="p-6">
      <General />
    </div>
  )
}
