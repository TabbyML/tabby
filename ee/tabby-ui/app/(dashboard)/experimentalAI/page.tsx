import { Metadata } from 'next'

import ExperimentalAIOption from './components/Option'

export const metadata: Metadata = {
  title: 'Experimental AI'
}

export default function IndexPage() {
  return (
    <div className="mx-auto flex max-w-xl flex-col gap-3">
      <h3 className="scroll-m-20 text-2xl font-semibold tracking-tight">
        Experimental AI
      </h3>

      <ExperimentalAIOption
        title="Quick Action Bar"
        description="Enable Quick Action Bar to display a convenient toolbar when you select code, offering options to explain the code, add unit tests, and more." />
    </div>
  )
}
