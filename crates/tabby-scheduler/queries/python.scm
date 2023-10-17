(function_definition
  name: (identifier) @name) @definition.function
(class_definition
  name: (identifier) @name) @definition.class

(import_statement
  (dotted_name (identifier) @name)) @definition.module

(assignment
  left: (_) @assignment.left
  right: (_) @assignment.right) @assignment

(if_statement
  condition: (_) @if-condition
  consequence: (_) @if-consequence)

(elif_clause
  condition: (_) @elif-condition
  consequence: (_) @elif-consequence)

(else_clause
  body: (_) @else-body)

(try_statement) @control.try

(except_clause) @control.except

(finally_clause) @control.finally

(for_statement) @control.for

(while_statement) @control.while

(return_statement) @control.return

(raise_statement) @control.raise

(with_statement) @control.with

(yield) @expression.yield

(pass_statement) @control.pass

(break_statement) @control.break

(continue_statement) @control.continue

(call
  function: (identifier) @name) @reference.call

(comment) @comment
(#strip! @comment "^#\\s*")

(#set-adjacent! @comment @definition.function)
(#set-adjacent! @comment @definition.class)

(await) @expression.await
