import { CardContent } from '@/components/ui/card'
import { SubHeader } from '@/components/sub-header'

import NotionDocument from './components/notion-doc'

export default function DocumentProviderPage() {
  return (
    <>
      
      <CardContent className="pl-0 pt-8 xl:pb-32">
        <SubHeader>
          <p>
            Notion Database Integration
          </p>
        </SubHeader>
        <NotionDocument />
      </CardContent>
    </>
  )
}
