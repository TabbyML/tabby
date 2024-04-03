(function_declaration
  name: (identifier) @name) @definition.function

(class_declaration
  name: (type_identifier) @name) @definition.class

(interface_declaration
  name: (type_identifier) @name) @definition.interface

(type_alias_declaration
  (type_identifier) @name) @definition.type

;; Top-level arrow function are definitions.
(program
  (lexical_declaration
    (variable_declarator
      name: (identifier) @name
      value: (arrow_function))) @definition.function)

;; Exported top-level arrow function are also definitions.
(program
  (export_statement
    (lexical_declaration
      (variable_declarator
        name: (identifier) @name
        value: (arrow_function))) @definition.function))