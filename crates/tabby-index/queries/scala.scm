
(class_definition
  name: (identifier) @name) @definition.class

(enum_definition
  name: (identifier) @name) @definition.enum

(object_definition
  name: (identifier) @name) @definition.object

(object_definition
  name: (identifier) @name) @definition.trait


(class_parameter
  name: (identifier) @name) @definition.class_parameter

(self_type (identifier) @name) @definition.parameter

(type_definition
  name: (type_identifier) @name) @definition.type


(val_definition
  pattern: (identifier) @name) @definition.val

(var_definition
  pattern: (identifier) @name) @definition.var

(val_declaration
  name: (identifier) @name) @definition.val_decl

(var_declaration
  name: (identifier) @name) @definition.var_decl



(call_expression
  function: (identifier) @name) @reference.call

(call_expression
  function: (operator_identifier) @name) @reference.call

(call_expression
  function: (field_expression
    field: (identifier) @name)) @reference.call

((call_expression
   function: (identifier) @name)
 (#match? @name "^[A-Z]")) @reference.constructor

(generic_function
  function: (identifier) @name) @reference.call

(interpolated_string_expression
  interpolator: (identifier) @name) @reference.call


(function_definition
  name: (identifier) @name) @definition.function


(function_declaration
      name: (identifier) @name) @definition.function

(function_definition
      name: (identifier) @name) @definition.function

