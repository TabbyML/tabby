import dedent from 'dedent'

import { SupportedLanguage } from '../grammars'
import type { QueryName } from '../queries'

/**
 * Incomplete code cases to cover:
 *
 * 1. call_expression: example(
 * 2. formal_parameters: function example(
 * 3. import_statement: import react
 * 4. lexical_declaration: const foo =
 *
 * The capture group name ending with "!" means this capture group does not require
 * a specific cursor position to match.
 *
 * TODO: try/catch, members, if/else, loops, etc.
 * Tracking: https://github.com/sourcegraph/cody/issues/1456
 */
const JS_INTENTS_QUERY = dedent`
    ; Cursor dependent intents
    ;--------------------------------

    (function_declaration
        name: (identifier) @function.name!
        parameters: (formal_parameters ("(") @function.parameters.cursor) @function.parameters
        body: (statement_block ("{") @function.body.cursor) @function.body)

    (function
        name: (identifier) @function.name!
        parameters: (formal_parameters ("(") @function.parameters.cursor) @function.parameters
        body: (statement_block ("{") @function.body.cursor) @function.body)

    (arrow_function
        parameters: (formal_parameters ("(") @function.parameters.cursor) @function.parameters
        body: (statement_block ("{") @function.body.cursor) @function.body)

    (class_declaration
        name: (_) @class.name!
        body: (class_body ("{") @class.body.cursor) @class.body)

    (arguments ("(") @arguments.cursor) @arguments

    ; Atomic intents
    ;--------------------------------

    (comment) @comment!
    (import_statement
        source: (string) @import.source!)

    (pair
        value: [
            (string (_)*)
            (template_string)
            (number)
            (identifier)
            (true)
            (false)
            (null)
            (undefined)
        ] @pair.value!)

    (arguments
        [
            (string (_)*)
            (template_string)
            (number)
            (identifier)
            (true)
            (false)
            (null)
            (undefined)
        ] @argument!)

    (formal_parameters) @parameters!
    (formal_parameters (_) @parameter!)

    (return_statement) @return_statement!
    (return_statement
        [
            (string (_)*)
            (template_string)
            (number)
            (identifier)
            (true)
            (false)
            (null)
            (undefined)
        ] @return_statement.value!)
`

const JSX_INTENTS_QUERY = dedent`
    ${JS_INTENTS_QUERY}

    (jsx_attribute (_) @jsx_attribute.value!)
`

const TS_INTENTS_QUERY = dedent`
    ${JS_INTENTS_QUERY}

    ; Cursor dependent intents
    ;--------------------------------

    (function_signature
        name: (identifier) @function.name!
        parameters: (formal_parameters ("(") @function.parameters.cursor) @function.parameters)

    (interface_declaration
        name: (type_identifier) @type_declaration.name!
        body: (object_type ("{") @type_declaration.body.cursor) @type_declaration.body)

    (type_alias_declaration
        name: (type_identifier) @type_declaration.name!
        value: (object_type ("{") @type_declaration.body.cursor) @type_declaration.body)
`

const TSX_INTENTS_QUERY = dedent`
    ${TS_INTENTS_QUERY}

    (jsx_attribute (_) @jsx_attribute.value!)
`

const TS_SINGLELINE_TRIGGERS_QUERY = dedent`
    (interface_declaration (object_type ("{") @block_start)) @trigger
    (type_alias_declaration (object_type ("{") @block_start)) @trigger
`

const JS_DOCUMENTABLE_NODES_QUERY = dedent`
    ; Identifiers
    ;--------------------------------
    (_
        name: (identifier) @identifier)

    ; Property Identifiers
    ;--------------------------------
    (method_definition
        name: (property_identifier) @identifier.property)
    (pair
        key: (property_identifier) @identifier.property)

    ; Exports
    ;--------------------------------
    ((export_statement) @export)
`

const TS_DOCUMENTABLE_NODES_QUERY = dedent`
    ${JS_DOCUMENTABLE_NODES_QUERY}

    ; Type Identifiers
    ;--------------------------------
    (_
        name: (type_identifier) @identifier)

    ; Type Signatures
    ;--------------------------------
    ((call_signature) @signature)
    (interface_declaration
        (object_type
            (property_signature
                name: (property_identifier) @signature.property)))
    (interface_declaration
        (object_type
            (method_signature
                name: (property_identifier) @signature.property)))
    (type_alias_declaration
        (object_type
            (property_signature
                name: (property_identifier) @signature.property)))
`

export const javascriptQueries = {
    [SupportedLanguage.JavaScript]: {
        singlelineTriggers: '',
        intents: JS_INTENTS_QUERY,
        documentableNodes: JS_DOCUMENTABLE_NODES_QUERY,
    },
    [SupportedLanguage.JSX]: {
        singlelineTriggers: '',
        intents: JSX_INTENTS_QUERY,
        documentableNodes: JS_DOCUMENTABLE_NODES_QUERY,
    },
    [SupportedLanguage.TypeScript]: {
        singlelineTriggers: TS_SINGLELINE_TRIGGERS_QUERY,
        intents: TS_INTENTS_QUERY,
        documentableNodes: TS_DOCUMENTABLE_NODES_QUERY,
    },
    [SupportedLanguage.TSX]: {
        singlelineTriggers: TS_SINGLELINE_TRIGGERS_QUERY,
        intents: TSX_INTENTS_QUERY,
        documentableNodes: TS_DOCUMENTABLE_NODES_QUERY,
    },
} satisfies Partial<Record<SupportedLanguage, Record<QueryName, string>>>
