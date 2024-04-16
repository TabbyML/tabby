const languages = {
  abap: {
    filenames: [],
    extnames: [`abap`]
  },
  actionscript: {
    filenames: [],
    extnames: [`as`]
  },
  ada: {
    filenames: [],
    extnames: [`ada`, `adb`, `ads`]
  },
  apacheconf: {
    filenames: [`.htaccess`, `apache2.conf`, `httpd.conf`],
    extnames: [`apacheconf`, `vhost`]
  },
  apl: {
    filenames: [],
    extnames: [`apl`, `dyalog`]
  },
  applescript: {
    filenames: [],
    extnames: [`applescript`, `scpt`]
  },
  arff: {
    filenames: [],
    extnames: [`arff`]
  },
  asciidoc: {
    filenames: [],
    extnames: [`asciidoc`, `adoc`, `asc`]
  },
  asm6502: {
    filenames: [],
    extnames: [`asm`]
  },
  autohotkey: {
    filenames: [],
    extnames: [`ahk`, `ahkl`]
  },
  autoit: {
    filenames: [],
    extnames: [`au3`]
  },
  bash: {
    filenames: [
      `.bash_history`,
      `.bash_logout`,
      `.bash_profile`,
      `.bashrc`,
      `.cshrc`,
      `.login`,
      `.profile`,
      `.zlogin`,
      `.zlogout`,
      `.zprofile`,
      `.zshenv`,
      `.zshrc`,
      `9fs`,
      `PKGBUILD`,
      `bash_logout`,
      `bash_profile`,
      `bashrc`,
      `cshrc`,
      `gradlew`,
      `login`,
      `man`,
      `profile`,
      `zlogin`,
      `zlogout`,
      `zprofile`,
      `zshenv`,
      `zshrc`
    ],
    extnames: [
      `sh`,
      `bash`,
      `bats`,
      `cgi`,
      `command`,
      `fcgi`,
      `ksh`,
      `tmux`,
      `tool`,
      `zsh`
    ]
  },
  basic: {
    filenames: [],
    extnames: [`vb`, `bas`, `cls`, `frm`, `frx`, `vba`, `vbhtml`, `vbs`]
  },
  batch: {
    filenames: [],
    extnames: [`bat`, `cmd`]
  },
  bison: {
    filenames: [],
    extnames: [`bison`]
  },
  brainfuck: {
    filenames: [],
    extnames: [`b`, `bf`]
  },
  bro: {
    filenames: [],
    extnames: [`bro`]
  },
  c: {
    filenames: [],
    extnames: [`c`, `cats`, `h`, `idc`]
  },
  csharp: {
    filenames: [],
    extnames: [`cs`, `cake`, `cshtml`, `csx`]
  },
  cpp: {
    filenames: [],
    extnames: [
      `cpp`,
      `c++`,
      `cc`,
      `cp`,
      `cxx`,
      `h`,
      `h++`,
      `hh`,
      `hpp`,
      `hxx`,
      `inc`,
      `inl`,
      `ino`,
      `ipp`,
      `re`,
      `tcc`,
      `tpp`
    ]
  },
  coffeescript: {
    filenames: [`Cakefile`],
    extnames: [`coffee`, `_coffee`, `cake`, `cjsx`, `iced`]
  },
  clojure: {
    filenames: [`riemann.config`],
    extnames: [
      `clj`,
      `boot`,
      `cl2`,
      `cljc`,
      `cljs`,
      `cljs.hl`,
      `cljscm`,
      `cljx`,
      `hic`
    ]
  },
  crystal: {
    filenames: [],
    extnames: [`cr`]
  },
  css: {
    filenames: [],
    extnames: [`css`]
  },
  d: {
    filenames: [],
    extnames: [`d`, `di`]
  },
  dart: {
    filenames: [],
    extnames: [`dart`]
  },
  diff: {
    filenames: [],
    extnames: [`diff`, `patch`]
  },
  django: {
    filenames: [],
    extnames: [`jinja`, `jinja2`, `mustache`, `njk`]
  },
  dockerfile: {
    filenames: [`Dockerfile`],
    extnames: [`dockerfile`]
  },
  eiffel: {
    filenames: [],
    extnames: [`e`]
  },
  elixir: {
    filenames: [`mix.lock`],
    extnames: [`ex`, `exs`]
  },
  elm: {
    filenames: [],
    extnames: [`elm`]
  },
  erb: {
    filenames: [],
    extnames: [`erb`]
  },
  erlang: {
    filenames: [`Emakefile`, `rebar.config`, `rebar.config.lock`, `rebar.lock`],
    extnames: [`erl`, `app.src`, `es`, `escript`, `hrl`, `xrl`, `yrl`]
  },
  fsharp: {
    filenames: [],
    extnames: [`fs`, `fsi`, `fsx`]
  },
  fortran: {
    filenames: [],
    extnames: [`f90`, `f`, `f03`, `f08`, `f77`, `f95`, `for`, `fpp`]
  },
  gedcom: {
    filenames: [],
    extnames: [`ged`]
  },
  gherkin: {
    filenames: [],
    extnames: [`feature`]
  },
  glsl: {
    filenames: [],
    extnames: [
      `glsl`,
      `fp`,
      `frag`,
      `frg`,
      `fs`,
      `fsh`,
      `fshader`,
      `geo`,
      `geom`,
      `glslv`,
      `gshader`,
      `shader`,
      `tesc`,
      `tese`,
      `vert`,
      `vrx`,
      `vsh`,
      `vshader`
    ]
  },
  go: {
    filenames: [],
    extnames: [`go`]
  },
  graphql: {
    filenames: [],
    extnames: [`graphql`, `gql`]
  },
  groovy: {
    filenames: [`Jenkinsfile`],
    extnames: [`groovy`, `grt`, `gtpl`, `gvy`]
  },
  haml: {
    filenames: [],
    extnames: [`haml`]
  },
  handlebars: {
    filenames: [],
    extnames: [`handlebars`, `hbs`]
  },
  haskell: {
    filenames: [],
    extnames: [`hs`, `hsc`]
  },
  haxe: {
    filenames: [],
    extnames: [`hx`, `hxsl`]
  },
  http: {
    filenames: [],
    extnames: [`http`]
  },
  icon: {
    filenames: [],
    extnames: [`icn`]
  },
  inform7: {
    filenames: [],
    extnames: [`ni`, `i7x`]
  },
  ini: {
    filenames: [`.editorconfig`, `.gitconfig`],
    extnames: [`ini`, `cfg`, `lektorproject`, `prefs`, `pro`, `properties`]
  },
  io: {
    filenames: [],
    extnames: [`io`]
  },
  j: {
    filenames: [],
    extnames: [`ijs`]
  },
  java: {
    filenames: [],
    extnames: [`java`]
  },
  javascript: {
    filenames: [`Jakefile`],
    extnames: [
      `js`,
      `_js`,
      `cjs`,
      `bones`,
      `es`,
      `es6`,
      `frag`,
      `gs`,
      `jake`,
      `jsb`,
      `jscad`,
      `jsfl`,
      `jsm`,
      `jss`,
      `mjs`,
      `njs`,
      `pac`,
      `sjs`,
      `ssjs`,
      `xsjs`,
      `xsjslib`
    ]
  },
  jolie: {
    filenames: [],
    extnames: [`ol`, `iol`]
  },
  json: {
    filenames: [
      `.arcconfig`,
      `.htmlhintrc`,
      `.tern-config`,
      `.tern-project`,
      `composer.lock`,
      `mcmod.info`
    ],
    extnames: [
      `json`,
      `avsc`,
      `geojson`,
      `gltf`,
      `JSON-tmLanguage`,
      `jsonl`,
      `tfstate`,
      `topojson`,
      `webapp`,
      `webmanifest`,
      `yy`,
      `yyp`
    ]
  },
  julia: {
    filenames: [],
    extnames: [`jl`]
  },
  keyman: {
    filenames: [],
    extnames: [`kmn`]
  },
  kotlin: {
    filenames: [],
    extnames: [`kt`, `ktm`, `kts`]
  },
  latex: {
    filenames: [],
    extnames: [
      `tex`,
      `aux`,
      `bbx`,
      `bib`,
      `cbx`,
      `cls`,
      `dtx`,
      `ins`,
      `lbx`,
      `ltx`,
      `mkii`,
      `mkiv`,
      `mkvi`,
      `sty`,
      `toc`
    ]
  },
  less: {
    filenames: [],
    extnames: [`less`]
  },
  liquid: {
    filenames: [],
    extnames: [`liquid`]
  },
  lisp: {
    filenames: [],
    extnames: [`lisp`, `asd`, `cl`, `l`, `lsp`, `ny`, `podsl`, `sexp`]
  },
  livescript: {
    filenames: [`Slakefile`],
    extnames: [`ls`, `_ls`]
  },
  lolcode: {
    filenames: [],
    extnames: [`lol`]
  },
  lua: {
    filenames: [],
    extnames: [`lua`, `fcgi`, `nse`, `p8`, `pd_lua`, `rbxs`, `wlua`]
  },
  cmake: {
    filenames: [
      `BSDmakefile`,
      `GNUmakefile`,
      `Kbuild`,
      `Makefile`,
      `Makefile.am`,
      `Makefile.boot`,
      `Makefile.frag`,
      `Makefile.in`,
      `Makefile.inc`,
      `Makefile.wat`,
      `makefile`,
      `makefile.sco`,
      `mkfile`
    ],
    extnames: [`mak`, `d`, `make`, `mk`, `mkfile`]
  },
  markdown: {
    filenames: [`contents.lr`, `LICENSE`],
    extnames: [
      `md`,
      `markdown`,
      `mdown`,
      `mdwn`,
      `mkd`,
      `mkdn`,
      `mkdown`,
      `ronn`,
      `workbook`
    ]
  },
  markup: {
    filenames: [],
    extnames: [
      `apib`,
      `blade`,
      `chem`,
      `ecr`,
      `eex`,
      `ejs`,
      `html`,
      `htm`,
      `ipynb`,
      `kit`,
      `latte`,
      `marko`,
      `mask`,
      `mtml`,
      `phtml`,
      `pic`,
      `raml`,
      `rhtml`,
      `vue`,
      `xht`,
      `xhtml`
    ]
  },
  matlab: {
    filenames: [],
    extnames: [`matlab`, `m`]
  },
  mel: {
    filenames: [],
    extnames: [`mel`]
  },
  mizar: {
    filenames: [],
    extnames: [`miz`, `voc`]
  },
  monkey: {
    filenames: [],
    extnames: [`monkey`, `monkey2`]
  },
  n4js: {
    filenames: [],
    extnames: [`n4jsd`]
  },
  nasm: {
    filenames: [],
    extnames: [`nasm`]
  },
  nginx: {
    filenames: [`nginx.conf`],
    extnames: [`nginxconf`, `vhost`]
  },
  nim: {
    filenames: [],
    extnames: [`nim`, `nimrod`]
  },
  nix: {
    filenames: [],
    extnames: [`nix`]
  },
  nsis: {
    filenames: [],
    extnames: [`nsi`, `nsh`]
  },
  objectivec: {
    filenames: [],
    extnames: [`m`, `h`]
  },
  ocaml: {
    filenames: [],
    extnames: [`ml`, `eliom`, `eliomi`, `ml4`, `mli`, `mll`, `mly`]
  },
  opencl: {
    filenames: [],
    extnames: [`opencl`, `cl`]
  },
  oz: {
    filenames: [],
    extnames: [`oz`]
  },
  pascal: {
    filenames: [],
    extnames: [`pas`, `dfm`, `dpr`, `inc`, `lpr`, `pascal`, `pp`]
  },
  perl: {
    filenames: [`Makefile.PL`, `Rexfile`, `ack`, `cpanfile`],
    extnames: [
      `pl`,
      `al`,
      `cgi`,
      `fcgi`,
      `perl`,
      `ph`,
      `plx`,
      `pm`,
      `psgi`,
      `t`
    ]
  },
  php: {
    filenames: [`.php`, `.php_cs`, `.php_cs.dist`, `Phakefile`],
    extnames: [
      `php`,
      `aw`,
      `ctp`,
      `fcgi`,
      `inc`,
      `php3`,
      `php4`,
      `php5`,
      `phps`,
      `phpt`
    ]
  },
  plsql: {
    filenames: [],
    extnames: [
      `pls`,
      `bdy`,
      `ddl`,
      `fnc`,
      `pck`,
      `pkb`,
      `pks`,
      `plb`,
      `plsql`,
      `prc`,
      `spc`,
      `tpb`,
      `tps`,
      `trg`,
      `vw`
    ]
  },
  powershell: {
    filenames: [],
    extnames: [`ps1`, `psd1`, `psm1`]
  },
  processing: {
    filenames: [],
    extnames: [`pde`]
  },
  prolog: {
    filenames: [],
    extnames: [`pl`, `pro`, `prolog`, `yap`]
  },
  properties: {
    filenames: [],
    extnames: [`properties`]
  },
  protobuf: {
    filenames: [],
    extnames: [`proto`]
  },
  pug: {
    filenames: [],
    extnames: [`jade`, `pug`]
  },
  puppet: {
    filenames: [`Modulefile`],
    extnames: [`pp`]
  },
  pure: {
    filenames: [],
    extnames: [`pure`]
  },
  python: {
    filenames: [
      `.gclient`,
      `BUCK`,
      `BUILD`,
      `BUILD.bazel`,
      `SConscript`,
      `SConstruct`,
      `Snakefile`,
      `WORKSPACE`,
      `wscript`
    ],
    extnames: [
      `py`,
      `bzl`,
      `cgi`,
      `fcgi`,
      `gyp`,
      `gypi`,
      `lmi`,
      `py3`,
      `pyde`,
      `pyi`,
      `pyp`,
      `pyt`,
      `pyw`,
      `rpy`,
      `spec`,
      `tac`,
      `wsgi`,
      `xpy`
    ]
  },
  q: {
    filenames: [],
    extnames: [`q`]
  },
  qore: {
    filenames: [],
    extnames: [`q`, `qm`, `qtest`]
  },
  r: {
    filenames: [`.Rprofile`, `expr-dist`],
    extnames: [`r`, `rd`, `rsx`]
  },
  jsx: {
    filenames: [],
    extnames: [`jsx`]
  },
  toml: {
    filenames: [],
    extnames: [`toml`]
  },
  tsx: {
    filenames: [],
    extnames: [`tsx`]
  },
  renpy: {
    filenames: [],
    extnames: [`rpy`]
  },
  reason: {
    filenames: [],
    extnames: [`re`, `rei`]
  },
  rest: {
    filenames: [],
    extnames: [`rst`, `rest`]
  },
  rip: {
    filenames: [],
    extnames: [`rip`]
  },
  ruby: {
    filenames: [
      `.irbrc`,
      `.pryrc`,
      `Appraisals`,
      `Berksfile`,
      `Brewfile`,
      `Buildfile`,
      `Capfile`,
      `Dangerfile`,
      `Deliverfile`,
      `Fastfile`,
      `Gemfile`,
      `Gemfile.lock`,
      `Guardfile`,
      `Jarfile`,
      `Mavenfile`,
      `Podfile`,
      `Puppetfile`,
      `Rakefile`,
      `Snapfile`,
      `Thorfile`,
      `Vagrantfile`,
      `buildfile`
    ],
    extnames: [
      `rb`,
      `builder`,
      `eye`,
      `fcgi`,
      `gemspec`,
      `god`,
      `jbuilder`,
      `mspec`,
      `pluginspec`,
      `podspec`,
      `rabl`,
      `rake`,
      `rbuild`,
      `rbw`,
      `rbx`,
      `ru`,
      `ruby`,
      `spec`,
      `thor`,
      `watchr`
    ]
  },
  rust: {
    filenames: [],
    extnames: [`rs`]
  },
  sas: {
    filenames: [],
    extnames: [`sas`]
  },
  sass: {
    filenames: [],
    extnames: [`sass`]
  },
  scss: {
    filenames: [],
    extnames: [`scss`]
  },
  scala: {
    filenames: [],
    extnames: [`scala`, `kojo`, `sbt`, `sc`]
  },
  scheme: {
    filenames: [],
    extnames: [`scm`, `sch`, `sld`, `sls`, `sps`, `ss`]
  },
  smalltalk: {
    filenames: [],
    extnames: [`st`, `cs`]
  },
  smarty: {
    filenames: [],
    extnames: [`tpl`]
  },
  sql: {
    filenames: [],
    extnames: [`sql`, `cql`, `ddl`, `inc`, `mysql`, `prc`, `tab`, `udf`, `viw`]
  },
  soy: {
    filenames: [],
    extnames: [`soy`]
  },
  stylus: {
    filenames: [],
    extnames: [`styl`]
  },
  swift: {
    filenames: [],
    extnames: [`swift`]
  },
  tcl: {
    filenames: [`owh`, `starfield`],
    extnames: [`tcl`, `adp`, `tm`]
  },
  textile: {
    filenames: [],
    extnames: [`textile`]
  },
  tt2: {
    filenames: [],
    extnames: [`pm`]
  },
  twig: {
    filenames: [],
    extnames: [`twig`]
  },
  typescript: {
    filenames: [],
    extnames: [`ts`]
  },
  velocity: {
    filenames: [],
    extnames: [`vm`]
  },
  verilog: {
    filenames: [],
    extnames: [`v`, `veo`]
  },
  vhdl: {
    filenames: [],
    extnames: [`vhdl`, `vhd`, `vhf`, `vhi`, `vho`, `vhs`, `vht`, `vhw`]
  },
  vim: {
    filenames: [
      `.gvimrc`,
      `.nvimrc`,
      `.vimrc`,
      `_vimrc`,
      `gvimrc`,
      `nvimrc`,
      `vimrc`
    ],
    extnames: [`vim`]
  },
  'visual-basic': {
    filenames: [],
    extnames: [`vb`, `bas`, `cls`, `frm`, `frx`, `vba`, `vbhtml`, `vbs`]
  },
  wasm: {
    filenames: [],
    extnames: [`wast`, `wat`]
  },
  xojo: {
    filenames: [],
    extnames: [
      `xojo_code`,
      `xojo_menu`,
      `xojo_report`,
      `xojo_script`,
      `xojo_toolbar`,
      `xojo_window`
    ]
  },
  xquery: {
    filenames: [],
    extnames: [`xquery`, `xq`, `xql`, `xqm`, `xqy`]
  },
  yaml: {
    filenames: [`.clang-format`, `.clang-tidy`, `.gemrc`, `glide.lock`],
    extnames: [
      `yml`,
      `mir`,
      `reek`,
      `rviz`,
      `sublime-syntax`,
      `syntax`,
      `yaml`,
      `yaml-tmlanguage`
    ]
  }
}

export default languages
