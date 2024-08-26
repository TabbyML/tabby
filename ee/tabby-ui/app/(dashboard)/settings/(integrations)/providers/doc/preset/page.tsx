import { Metadata } from 'next'

import PresetDocument from '../components/preset-doc'

export const metadata: Metadata = {
  title: 'Preset Document'
}

export default function PresetDocumentPage() {
  return <PresetDocument />
}
