import { CardContent } from '@/components/ui/card'
import { SubHeader } from '@/components/sub-header'

import CustomDocument from './components/custom-doc'
import PresetDocument from './components/preset-doc'

export default function DocumentProviderPage() {
  return (
    <>
      <CardContent className="pl-0">
        <SubHeader>
          <p>
            Documents are a critical source for engineering knowledge. Tabby
            provides an easy way to include these documents when interacting
            with LLM in chat interfaces (e.g., Answer Engine, Chat Panel, etc.).
            Simply press the @ button in the chat interface and select the
            document you wish to include.
          </p>
        </SubHeader>
        <PresetDocument />
      </CardContent>
      <CardContent className="pl-0 pt-8 xl:pb-32">
        <SubHeader>
          <p>
            You can also include your own developer documents here. Please
            ensure that the URLs are accessible from the Tabby server to
            guarantee successful crawling.
          </p>
        </SubHeader>
        <CustomDocument />
      </CardContent>
    </>
  )
}
