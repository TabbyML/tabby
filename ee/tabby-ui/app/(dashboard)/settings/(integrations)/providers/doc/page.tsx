import { Metadata } from 'next'

import { CardContent } from '@/components/ui/card'

import CustomDocument from './components/custom-doc'
import PresetDocument from './components/preset-doc'

export const metadata: Metadata = {
  title: 'Document'
}

export default function DocumentProviderPage() {
  return (
    <>
      <CardContent className="pl-0">
        <CustomDocument />
      </CardContent>
      <div className="h-16" />
      <CardContent className="pl-0">
        <PresetDocument />
      </CardContent>
    </>
  )
}
