(
  (comment)* @doc
  .
  (function_declaration
    name: (identifier) @name) @definition.function
  (#strip! @doc "^//\\s*")
  (#set-adjacent! @doc @definition.function)
)

(
  (comment)* @doc
  .
  (class_declaration
    name: (identifier) @name) @definition.class
    (#strip! @doc "^//\\s*")
    (#set-adjacent! @doc @definition.class)
)

(
  (comment)* @doc
  .
  (interface_declaration
    name: (identifier) @name) @definition.interface
    (#strip! @doc "^//\\s*")
    (#set-adjacent! @doc @definition.interface)
)

(
  (comment)* @doc
  .
  (object_declaration
    type: (type_identifier) @name) @reference.class
    (#strip! @doc "^//\\s*")
    (#set-adjacent! @doc @reference.class)
)

(type_declaration (type_spec name: (type_identifier) @name)) @definition.type