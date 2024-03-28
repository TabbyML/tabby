import { Metadata } from 'next'

import FeatureList from './components/feature-list'

export const metadata: Metadata = {
  title: 'Experiment Flags'
}

export default function IndexPage() {
  return (
    <div className="mx-auto flex max-w-xl flex-col gap-3">
      <h3 className="scroll-m-20 text-2xl font-semibold tracking-tight">
        Experiment Flags
      </h3>
      <FeatureList />
    </div>
  )
}
