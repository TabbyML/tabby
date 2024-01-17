import dedent from 'dedent'

import { SupportedLanguage } from '../grammars'
import type { QueryName } from '../queries'

export const goQueries = {
    [SupportedLanguage.Go]: {
        singlelineTriggers: dedent`
            (struct_type (field_declaration_list ("{") @block_start)) @trigger
            (interface_type ("{") @block_start) @trigger
        `,
        intents: '',
        documentableNodes: '',
    },
} satisfies Partial<Record<SupportedLanguage, Record<QueryName, string>>>
