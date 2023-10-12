; Modified based on https://github.com/tree-sitter/tree-sitter-typescript/blob/master/queries/tags.scm

(function_signature
  name: (identifier) @name) @definition.function

(method_signature
  name: (property_identifier) @name) @definition.method

(abstract_method_signature
  name: (property_identifier) @name) @definition.method

(abstract_class_declaration
  name: (type_identifier) @name) @definition.class

(module
  name: (identifier) @name) @definition.module

(interface_declaration
  name: (type_identifier) @name) @definition.interface

(type_annotation
  (type_identifier) @name) @reference.type

(new_expression
  constructor: (identifier) @name) @reference.class

(call_expression
  function: (identifier) @name) @reference.call

(call_expression
  function: (member_expression property: (property_identifier) @name)) @reference.call