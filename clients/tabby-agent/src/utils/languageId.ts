import * as path from "path";

// https://code.visualstudio.com/docs/languages/identifiers

export function getLanguageId(uri: string): string {
  const extensionToLanguageId: { [key: string]: string } = {
    ".abap": "abap",
    ".bat": "bat",
    ".bib": "bibtex",
    ".clj": "clojure",
    ".coffee": "coffeescript",
    ".c": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".css": "css",
    ".cu": "cuda-cpp",
    ".d": "d",
    ".dart": "dart",
    ".pas": "pascal",
    ".diff": "diff",
    ".dockerfile": "dockerfile",
    ".erl": "erlang",
    ".fs": "fsharp",
    ".go": "go",
    ".groovy": "groovy",
    ".hbs": "handlebars",
    ".haml": "haml",
    ".hs": "haskell",
    ".html": "html",
    ".ini": "ini",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascriptreact",
    ".json": "json",
    ".jsonc": "jsonc",
    ".jl": "julia",
    ".tex": "latex",
    ".less": "less",
    ".lua": "lua",
    ".m": "objective-c",
    ".mm": "objective-cpp",
    ".ml": "ocaml",
    ".pl": "perl",
    ".php": "php",
    ".txt": "plaintext",
    ".ps1": "powershell",
    ".pug": "pug",
    ".py": "python",
    ".r": "r",
    ".cshtml": "razor",
    ".rb": "ruby",
    ".rs": "rust",
    ".scss": "scss",
    ".sass": "sass",
    ".shader": "shaderlab",
    ".sh": "shellscript",
    ".slim": "slim",
    ".sql": "sql",
    ".styl": "stylus",
    ".svelte": "svelte",
    ".swift": "swift",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".vb": "vb",
    ".vue": "vue",
    ".xml": "xml",
    ".xsl": "xsl",
    ".yaml": "yaml",
    ".yml": "yaml",
    // Add more extensions as needed
  };

  const basenameToLanguageId: { [key: string]: string } = {
    Dockerfile: "dockerfile",
    Makefile: "makefile",
    "git-commit": "git-commit",
    "git-rebase": "git-rebase",
    // Add more special filenames as needed
  };

  const basename = path.basename(uri);
  if (basenameToLanguageId[basename]) {
    return basenameToLanguageId[basename];
  }

  const ext = path.extname(uri).toLowerCase();
  if (extensionToLanguageId[ext]) {
    return extensionToLanguageId[ext];
  }

  // Return extname without the dot as default
  return ext ? ext.slice(1) : "plaintext";
}
