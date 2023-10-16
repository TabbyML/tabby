(function_definition
  name: (identifier) @name) @definition.function

(class_definition
  name: (identifier) @name) @definition.class

(import_statement
  module: (identifier) @name) @definition.module

(import_from_statement
  module: (identifier) @name) @definition.module

(if_statement) @control.if
(elif_statement) @control.elif
(else_statement) @control.else
(try_statement) @control.try
(except_clause) @control.except
(finally_clause) @control.finally
(for_statement) @control.for
(while_statement) @control.while
(return_statement) @control.return
(raise_statement) @control.raise

(async_function_definition
  name: (identifier) @name) @definition.function.async

(with_statement) @control.with

(yield_expression) @expression.yield

(call
  function: (identifier) @name) @reference.call

(comment) @comment
(#strip! @comment "^#\\s*")

(#set-adjacent! @comment @definition.function)
(#set-adjacent! @comment @definition.class)
