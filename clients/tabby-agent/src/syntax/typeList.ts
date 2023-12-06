// https://github.com/tree-sitter/tree-sitter-typescript/blob/master/src/node-types.json
export const typeList: Record<string, string[][]> = {
  tsx: [
    [
      "jsx_element",
      "jsx_self_closing_element",

      // exclude sentence level nodes for now
      // "expression_statement",
      // "lexical_declaration",

      "for_statement",
      "for_in_statement",
      "if_statement",
      "while_statement",
      "do_statement",
      "switch_statement",
      "try_statement",
      "with_statement",
      "labeled_statement",

      "class_declaration",
      "abstract_class_declaration",
      "interface_declaration",
      "enum_declaration",
      "type_alias_declaration",
      "function_declaration",
      "generator_function_declaration",
      "ambient_declaration",

      "method_definition",

      "import_statement",
      "export_statement",
      "module",
    ],
  ],

  // https://github.com/tree-sitter/tree-sitter-python/blob/master/src/node-types.json
  python: [
    [
      "for_statement",
      "if_statement",
      "while_statement",
      "match_statement",
      "try_statement",
      "with_statement",

      "function_definition",
      "decorated_definition",
      "class_definition",

      "import_statement",
      "import_from_statement",
    ],
  ],

  // https://github.com/tree-sitter/tree-sitter-go/blob/master/src/node-types.json
  go: [
    [
      "for_statement",
      "if_statement",
      "expression_switch_statement",
      "type_switch_statement",
      "select_statement",
      "labeled_statement",

      "function_declaration",
      "method_declaration",
      "type_declaration",

      "import_declaration",
      "package_clause",
    ],
  ],

  // https://github.com/tree-sitter/tree-sitter-rust/blob/master/src/node-types.json
  rust: [
    [
      "for_expression",
      "if_expression",
      "while_expression",
      "loop_expression",
      "match_expression",
      "try_expression",

      "function_item",
      "type_item",
      "enum_item",
      "struct_item",
      "union_item",
      "trait_item",
      "impl_item",

      "use_declaration",
    ],
  ],

  // https://github.com/tree-sitter/tree-sitter-ruby/blob/master/src/node-types.json
  ruby: [
    [
      "for",
      "if",
      "unless",
      "while",
      "until",
      "case",

      "class",
      "singleton_class",
      "method",
      "singleton_method",
      "module",
    ],
  ],
};
