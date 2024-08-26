import { Metadata } from 'next'

import CustomDocument from '../components/custom-doc'

export const metadata: Metadata = {
  title: 'Custom Document'
}

export default function CustomDocumentPage() {
  return <CustomDocument />
}
