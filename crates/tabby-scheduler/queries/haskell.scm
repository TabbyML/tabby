(
  (comment)* @doc
  .
  (function_declaration
    name: (identifier) @name) @definition.function
  (#strip! @doc "^--\\s*")
  (#set-adjacent! @doc @definition.function)
)

(
  (data_declaration
    name: (type_identifier) @name) @definition.type
)

(
  (type_declaration
    name: (type_identifier) @name) @definition.type
)

(
  (class_declaration
    name: (type_identifier) @name) @definition.class
)

(
  (instance_declaration
    name: (type_identifier) @name) @definition.instance
)

(
  (module_declaration
    name: (module_identifier) @name) @definition.module
)

(
  (import_statement
    module: (module_identifier) @name) @reference.import
)
