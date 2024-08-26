import { Metadata } from 'next'

import CreateCustomDocument from '../../components/create-custom-doc'

export const metadata: Metadata = {
  title: 'Create Custom Document'
}

export default function CreateCustomDocumentPage() {
  return <CreateCustomDocument />
}
